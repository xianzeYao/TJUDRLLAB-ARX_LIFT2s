from __future__ import annotations

import argparse
import json
import math
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

THIS_DIR = Path(__file__).resolve().parent
ROOT_DIR = THIS_DIR.parent.parent
ROS2_DIR = (THIS_DIR / "../ROS2").resolve()
DEMO_DIR = (ROOT_DIR / "Demo").resolve()
if str(ROS2_DIR) not in sys.path:
    sys.path.insert(0, str(ROS2_DIR))
if str(DEMO_DIR) not in sys.path:
    sys.path.insert(0, str(DEMO_DIR))

from base_calib_collect import (
    CENTER_TO_WHEEL_M,
    WHEEL_MAPPING,
    WHEEL_RADIUS_M,
    ensure_output_dir,
    query_robot_type,
)


DEFAULT_PROMPT = "tennis ball"
DEFAULT_OUT_DIR = THIS_DIR / "Testdata4Lift"
DEFAULT_SAMPLE_HZ = 1.0
DEFAULT_LIFT_RATE = 1.0
DEFAULT_START_HEIGHT = 0.0
DEFAULT_MAX_HEIGHT = 20.0
DEFAULT_OFFSET = 0.5
DEFAULT_DEPTH_MEDIAN_N = 5
DEFAULT_TARGET_SIZE = (640, 480)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Auto-collect fixed-object pw.z values while lifting the base."
    )
    parser.add_argument("--prompt", type=str, default=DEFAULT_PROMPT)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--sample-hz", type=float, default=DEFAULT_SAMPLE_HZ)
    parser.add_argument("--lift-rate", type=float, default=DEFAULT_LIFT_RATE)
    parser.add_argument("--start-height", type=float, default=DEFAULT_START_HEIGHT)
    parser.add_argument("--max-height", type=float, default=DEFAULT_MAX_HEIGHT)
    parser.add_argument("--offset", type=float, default=DEFAULT_OFFSET)
    parser.add_argument("--depth-median-n", type=int, default=DEFAULT_DEPTH_MEDIAN_N)
    parser.add_argument("--camera", type=str, default="camera_h")
    parser.add_argument("--img-w", type=int, default=DEFAULT_TARGET_SIZE[0])
    parser.add_argument("--img-h", type=int, default=DEFAULT_TARGET_SIZE[1])
    parser.add_argument("--skip-reset", action="store_true")
    return parser.parse_args()


def choose_goal_point(
    points: list[tuple[float, float]],
    depth,
    offset: float,
    prev_pixel: Optional[tuple[int, int]],
):
    from point2pos_utils import depth_to_meters, pixel_to_base_point_safe
    import numpy as np

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
            {
                "pixel": pixel,
                "depth_m": float(depth_m),
                "goal_pw": np.asarray(goal_pw, dtype=np.float32),
            }
        )

    if not valid_goals:
        raise RuntimeError("no valid point with usable depth")

    if prev_pixel is not None:
        return min(
            valid_goals,
            key=lambda item: math.hypot(
                float(item["pixel"][0] - prev_pixel[0]),
                float(item["pixel"][1] - prev_pixel[1]),
            ),
        )

    return min(
        valid_goals,
        key=lambda item: math.hypot(
            float(item["goal_pw"][0]),
            float(item["goal_pw"][1]),
        ),
    )


def detect_goal_once(
    arx,
    prompt: str,
    depth_median_n: int,
    offset: float,
    prev_pixel: Optional[tuple[int, int]],
):
    from arx_pointing import predict_multi_points_from_rgb
    from point2pos_utils import get_aligned_frames
    from shelf_search import target_point_prompt

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

    chosen = choose_goal_point(
        points=points,
        depth=depth,
        offset=offset,
        prev_pixel=prev_pixel,
    )
    return {
        "color_shape": list(color.shape),
        "predicted_points_count": len(points),
        "pixel": [int(chosen["pixel"][0]), int(chosen["pixel"][1])],
        "depth_m": float(chosen["depth_m"]),
        "goal_pw": [
            float(chosen["goal_pw"][0]),
            float(chosen["goal_pw"][1]),
            float(chosen["goal_pw"][2]),
        ],
    }


def build_record(
    *,
    sample_index: int,
    prompt: str,
    target_height: float,
    observed_height: Optional[float],
    robot_type: int,
    robot_type_source: str,
    detect_result: Optional[dict[str, Any]],
    error: Optional[str],
    sample_started_at: float,
    detect_started_at: float,
    prev_pixel: Optional[tuple[int, int]],
) -> dict[str, Any]:
    return {
        "sample_index": int(sample_index),
        "timestamp": datetime.fromtimestamp(sample_started_at).isoformat(timespec="milliseconds"),
        "timestamp_wall_s": float(sample_started_at),
        "prompt": prompt,
        "target_lift_height": float(target_height),
        "observed_lift_height": None if observed_height is None else float(observed_height),
        "robot_type": int(robot_type),
        "robot_type_source": robot_type_source,
        "detection_success": error is None,
        "detect_latency_s": float(time.time() - detect_started_at),
        "last_pixel_before_detect": None if prev_pixel is None else [int(prev_pixel[0]), int(prev_pixel[1])],
        "wheel_radius_m": WHEEL_RADIUS_M,
        "center_to_wheel_m": CENTER_TO_WHEEL_M,
        "wheel_mapping": WHEEL_MAPPING,
        "body_forward_definition": "arrow_up",
        "error": error,
        "detect_result": detect_result,
    }


def save_payload(out_dir: Path, payload: dict[str, Any]) -> Path:
    session_name = datetime.now().strftime("lift_pwz_%Y%m%d_%H%M%S.json")
    out_path = out_dir / session_name
    out_path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    return out_path


def main() -> None:
    args = parse_args()
    if args.sample_hz <= 0.0:
        raise ValueError("--sample-hz must be > 0")
    if args.lift_rate <= 0.0:
        raise ValueError("--lift-rate must be > 0")
    if args.max_height < args.start_height:
        raise ValueError("--max-height must be >= --start-height")
    if args.depth_median_n <= 0:
        raise ValueError("--depth-median-n must be > 0")

    from arx_ros2_env import ARXRobotEnv

    out_dir = ensure_output_dir(args.out)
    robot_type, robot_type_source = query_robot_type()
    lift_step = float(args.lift_rate) / float(args.sample_hz)

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

    records: list[dict[str, Any]] = []
    prev_pixel: Optional[tuple[int, int]] = None
    current_height = float(args.start_height)
    sample_index = 0
    started_at = time.time()

    try:
        if not args.skip_reset:
            env.reset()

        env.step_lift(current_height)
        time.sleep(1.0)

        while current_height <= float(args.max_height) + 1e-9:
            sample_started_at = time.time()
            env.step_lift(current_height)

            observed_height = None
            try:
                status = env.get_robot_status()
                base_status = status.get("base") if isinstance(status, dict) else None
                if base_status is not None:
                    observed_height = float(base_status.height)
            except Exception:
                observed_height = None

            detect_started_at = time.time()
            detect_result: Optional[dict[str, Any]] = None
            error: Optional[str] = None
            try:
                detect_result = detect_goal_once(
                    arx=env,
                    prompt=args.prompt,
                    depth_median_n=int(args.depth_median_n),
                    offset=float(args.offset),
                    prev_pixel=prev_pixel,
                )
                px = detect_result["pixel"]
                prev_pixel = (int(px[0]), int(px[1]))
            except Exception as exc:
                error = str(exc)

            records.append(
                build_record(
                    sample_index=sample_index,
                    prompt=args.prompt,
                    target_height=current_height,
                    observed_height=observed_height,
                    robot_type=robot_type,
                    robot_type_source=robot_type_source,
                    detect_result=detect_result,
                    error=error,
                    sample_started_at=sample_started_at,
                    detect_started_at=detect_started_at,
                    prev_pixel=prev_pixel,
                )
            )

            sample_index += 1
            if current_height >= float(args.max_height):
                break

            current_height = min(
                float(args.max_height),
                current_height + lift_step,
            )

        payload = {
            "created_at": datetime.now().isoformat(timespec="seconds"),
            "summary": {
                "prompt": args.prompt,
                "start_height": float(args.start_height),
                "max_height": float(args.max_height),
                "sample_hz": float(args.sample_hz),
                "lift_rate": float(args.lift_rate),
                "lift_step": float(lift_step),
                "offset": float(args.offset),
                "depth_median_n": int(args.depth_median_n),
                "robot_type": int(robot_type),
                "robot_type_source": robot_type_source,
                "duration_s": float(time.time() - started_at),
                "record_count": len(records),
                "goal": "measure pw.z of a fixed world target across lift heights",
            },
            "records": records,
        }
        out_path = save_payload(out_dir, payload)
        print(f"saved {len(records)} records -> {out_path}")
    finally:
        env.close()


if __name__ == "__main__":
    main()
