from __future__ import annotations

import argparse
import json
import math
import time
from datetime import datetime
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

import sys

THIS_DIR = Path(__file__).resolve().parent
ROOT_DIR = THIS_DIR.parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from ARX_Realenv.ROS2.arx_ros2_env import ARXRobotEnv  # noqa: E402
from Demo.shelf_search import target_point_prompt  # noqa: E402
from Demo.utils import (  # noqa: E402
    depth_to_meters,
    get_aligned_frames,
    pixel_to_base_point_safe,
    predict_multi_points_from_rgb,
)


def _choose_goal_point(
    points: list[tuple[float, float]],
    depth: np.ndarray,
    offset: float,
) -> tuple[tuple[int, int], float, np.ndarray]:
    valid_goals = []
    for point in points:
        pixel = (int(round(point[0])), int(round(point[1])))
        goal_pw = pixel_to_base_point_safe(
            pixel,
            depth,
            robot_part="center",
            offset=[0.0, offset, 0.0],
        )
        if goal_pw is None:
            continue
        depth_m = depth_to_meters(float(depth[pixel[1], pixel[0]]))
        valid_goals.append(
            (pixel, depth_m, np.asarray(goal_pw, dtype=np.float32)))

    if not valid_goals:
        raise RuntimeError("no valid point with usable depth")

    return min(
        valid_goals,
        key=lambda item: math.hypot(float(item[2][0]), float(item[2][1])),
    )


def _detect_goal_once(
    arx: ARXRobotEnv,
    prompt: str,
    depth_median_n: int,
    offset: float,
) -> tuple[np.ndarray, tuple[int, int], float, np.ndarray]:
    color, depth = get_aligned_frames(arx, depth_median_n=depth_median_n)
    if color is None or depth is None:
        raise RuntimeError("failed to read aligned color/depth frames")

    points, raw = predict_multi_points_from_rgb(
        image=color,
        text_prompt="",
        all_prompt=target_point_prompt(prompt),
        assume_bgr=False,
        temperature=0.0,
        return_raw=True,
    )
    if not points:
        raise RuntimeError(
            f"no point predicted for prompt: {prompt!r}, raw={raw!r}")

    pixel, depth_m, goal_pw = _choose_goal_point(points, depth, offset=offset)
    return color, pixel, depth_m, goal_pw


def _draw_status(
    image: np.ndarray,
    prompt: str,
    base_height: float,
    current_record: Optional[dict],
    saved_count: int,
) -> np.ndarray:
    canvas = image.copy()
    lines = [
        f"prompt: {prompt}",
        f"lift: {base_height:.2f}",
        f"saved: {saved_count}",
        "w/s lift, e detect, r clear, y save, q quit",
    ]

    if current_record is None:
        lines.append("current: none")
    else:
        pixel = tuple(current_record["pixel"])
        goal_pw = current_record["goal_pw"]
        cv2.circle(canvas, pixel, 6, (0, 255, 0), -1)
        lines.extend(
            [
                f"pixel: {pixel}",
                f"depth_m: {current_record['depth_m']:.4f}",
                f"goal_pw: {[round(v, 4) for v in goal_pw]}",
            ]
        )

    y = 28
    for line in lines:
        cv2.putText(
            canvas,
            line,
            (10, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 0, 255),
            2,
            cv2.LINE_AA,
        )
        y += 24
    return canvas


def _load_records(out_path: Path) -> list[dict]:
    if not out_path.exists():
        return []
    try:
        payload = json.loads(out_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return []
    return payload if isinstance(payload, list) else []


def _save_records(out_path: Path, records: list[dict]) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(
        json.dumps(records, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="ARX interactive height test collector")
    parser.add_argument("--prompt", type=str,
                        default="tennis ball")
    parser.add_argument(
        "--out",
        type=Path,
        default=THIS_DIR / "Testdata4Lift" / "height_test_manual.json",
        help="output json path",
    )
    parser.add_argument("--camera", type=str, default="camera_h")
    parser.add_argument("--img-w", type=int, default=640)
    parser.add_argument("--img-h", type=int, default=480)
    parser.add_argument("--height-step", type=float, default=1.0)
    parser.add_argument("--height-min", type=float, default=0.0)
    parser.add_argument("--height-max", type=float, default=20.0)
    parser.add_argument("--depth-median-n", type=int, default=5)
    parser.add_argument("--offset", type=float, default=0.5)
    parser.add_argument("--skip-home", action="store_true")
    args = parser.parse_args()

    env = ARXRobotEnv(
        min_steps=20,
        max_v_xyz=0.25,
        max_a_xyz=0.20,
        max_v_rpy=0.3,
        max_a_rpy=1.00,
        camera_type="all",
        camera_view=(args.camera,),
        img_size=(args.img_w, args.img_h),
    )

    if not args.skip_home:
        env.reset()

    time.sleep(1.0)
    obs0 = env.get_observation(
        include_arm=False, include_camera=False, include_base=True)
    base_height = float(
        obs0.get("base_height", np.array([0.0], dtype=np.float32))[0])

    records = _load_records(args.out)
    current_record: Optional[dict] = None
    win = "height_test_collect"

    cv2.namedWindow(win, cv2.WINDOW_NORMAL)
    print("操作说明: w上升, s下降, e检测, r清空当前检测, y保存到json, q/ESC退出")

    try:
        while True:
            color, _ = get_aligned_frames(env, depth_median_n=1)
            if color is None:
                cv2.waitKey(1)
                continue

            vis = _draw_status(
                image=color,
                prompt=args.prompt,
                base_height=base_height,
                current_record=current_record,
                saved_count=len(records),
            )
            cv2.imshow(win, vis)

            key = cv2.waitKey(1) & 0xFF
            if key in (27, ord("q")):
                break
            if key == ord("w"):
                base_height = min(
                    args.height_max, base_height + args.height_step)
                env.step_lift(base_height)
                print(f"lift -> {base_height:.2f}")
            elif key == ord("s"):
                base_height = max(
                    args.height_min, base_height - args.height_step)
                env.step_lift(base_height)
                print(f"lift -> {base_height:.2f}")
            elif key == ord("e"):
                try:
                    _, pixel, depth_m, goal_pw = _detect_goal_once(
                        arx=env,
                        prompt=args.prompt,
                        depth_median_n=args.depth_median_n,
                        offset=args.offset,
                    )
                    current_record = {
                        "timestamp": datetime.now().isoformat(timespec="milliseconds"),
                        "prompt": args.prompt,
                        "lift_height": float(base_height),
                        "offset": float(args.offset),
                        "pixel": [int(pixel[0]), int(pixel[1])],
                        "depth_m": float(depth_m),
                        "goal_pw": [float(goal_pw[0]), float(goal_pw[1]), float(goal_pw[2])],
                    }
                    print(current_record)
                except Exception as exc:
                    current_record = None
                    print(f"detection failed: {exc}")
            elif key == ord("r"):
                current_record = None
                print("current detection cleared")
            elif key == ord("y"):
                if current_record is None:
                    print("no current detection to save")
                    continue
                records.append(dict(current_record))
                _save_records(args.out, records)
                print(f"saved -> {args.out} (total={len(records)})")
    finally:
        cv2.destroyAllWindows()
        env.close()


if __name__ == "__main__":
    main()
