"""
交互式采集脚本：
- `w` / `s` 控制升降高度（上/下）
- 鼠标左键在图像上打点，实时显示红点和编号
- `y` 保存一次样本（无点图、有点图、json）
- `r` 清空当前点
- `q` 或 `ESC` 退出

每次保存会在输出目录下创建一个子文件夹，包含：
- image_raw.png
- image_marked.png
- meta.json
"""

from __future__ import annotations

import argparse
import json
import time
from datetime import datetime
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

import sys

THIS_DIR = Path(__file__).resolve().parent
ROOT_DIR = THIS_DIR.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from ARX_Realenv.ROS2.arx_ros2_env import ARXRobotEnv  # noqa: E402


def find_color_frame(obs: dict, prefer_cam: str = "camera_h") -> tuple[Optional[str], Optional[np.ndarray]]:
    candidates = [(k, v) for k, v in obs.items()
                  if isinstance(v, np.ndarray) and "color" in k]
    if not candidates:
        return None, None

    for key, img in candidates:
        if prefer_cam in key:
            return key, img
    return candidates[0]


def find_depth_frame(obs: dict, prefer_cam: str = "camera_h") -> tuple[Optional[str], Optional[np.ndarray]]:
    candidates = [(k, v) for k, v in obs.items()
                  if isinstance(v, np.ndarray) and "depth" in k]
    if not candidates:
        return None, None

    for key, img in candidates:
        if prefer_cam in key:
            return key, img
    return candidates[0]


def _depth_to_meters(depth: np.ndarray, unit: str) -> np.ndarray:
    depth_f = depth.astype(np.float32)
    if unit == "mm":
        return depth_f * 0.001
    if unit == "m":
        return depth_f
    valid = np.isfinite(depth_f) & (depth_f > 0)
    if np.any(valid):
        if float(np.percentile(depth_f[valid], 50)) > 20.0:
            return depth_f * 0.001
    return depth_f


def depth_to_colormap(depth: np.ndarray, vmin: float, vmax: float, unit: str) -> tuple[np.ndarray | None, dict]:
    depth_m = _depth_to_meters(depth, unit)
    valid = np.isfinite(depth_m) & (depth_m > 0)
    if not np.any(valid):
        return None, {}
    vals = depth_m[valid]
    if vmax <= vmin:
        vmax = vmin + 1e-6
    norm = (depth_m - vmin) / (vmax - vmin)
    norm = np.clip(norm, 0.0, 1.0)
    img8 = (norm * 255.0).astype(np.uint8)
    color = cv2.applyColorMap(img8, cv2.COLORMAP_JET)
    color[~valid] = 0
    stats = {
        "min": float(np.min(vals)),
        "med": float(np.median(vals)),
        "max": float(np.max(vals)),
        "above_max": float(np.mean(vals > vmax)),
    }
    return color, stats


def draw_points(image: np.ndarray, points: list[tuple[int, int]]) -> np.ndarray:
    canvas = image.copy()
    for idx, (x, y) in enumerate(points, start=1):
        cv2.circle(canvas, (x, y), 3, (0, 0, 255), -1)
        label = str(idx)
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.putText(
            canvas,
            label,
            (x - tw // 2, y + th // 2),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )
    return canvas


def save_sample(
    out_dir: Path,
    raw_img: np.ndarray,
    marked_img: np.ndarray,
    points: list[tuple[int, int]],
    base_height: float,
    camera_key: str,
) -> Path:
    ts = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    sample_dir = out_dir / f"sample_{ts}"
    sample_dir.mkdir(parents=True, exist_ok=False)

    raw_path = sample_dir / "image_raw.png"
    marked_path = sample_dir / "image_marked.png"
    cv2.imwrite(str(raw_path), raw_img)
    cv2.imwrite(str(marked_path), marked_img)

    h, w = raw_img.shape[:2]
    meta = {
        "timestamp": ts,
        "camera_key": camera_key,
        "image_size": {"width": int(w), "height": int(h)},
        "base_height": float(base_height),
        "points": [
            {"id": int(i), "x": int(x), "y": int(y)}
            for i, (x, y) in enumerate(points, start=1)
        ],
    }
    (sample_dir / "meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    return sample_dir


def main() -> None:
    parser = argparse.ArgumentParser(description="ARX 交互式打点采集")
    parser.add_argument("--out-dir", type=Path,
                        default=Path("../Testdata4Mani/collect4er"), help="保存根目录")
    parser.add_argument("--camera", type=str,
                        default="camera_h", help="优先使用的相机名")
    parser.add_argument("--img-w", type=int, default=640)
    parser.add_argument("--img-h", type=int, default=480)
    parser.add_argument("--height-step", type=float,
                        default=0.5, help="每次调高/调低的高度步长")
    parser.add_argument("--height-min", type=float, default=0.0)
    parser.add_argument("--height-max", type=float, default=20.0)
    parser.add_argument("--skip-home", action="store_true", help="不执行 reset()")
    parser.add_argument("--depth-min", type=float,
                        default=0.2, help="深度可视化最小值(米)")
    parser.add_argument("--depth-max", type=float,
                        default=2.5, help="深度可视化最大值(米)")
    parser.add_argument("--depth-unit", choices=["auto", "m", "mm"], default="auto",
                        help="深度单位(默认自动判断)")
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)

    env = ARXRobotEnv(
        min_steps=20,
        max_v_xyz=0.25, max_a_xyz=0.20,
        max_v_rpy=0.3, max_a_rpy=1.00,
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

    win = "collet4er"
    win_depth = "collet4er_depth"
    points: list[tuple[int, int]] = []

    def on_mouse(event, x, y, flags, param):
        del flags, param
        if event == cv2.EVENT_LBUTTONDOWN:
            points.append((int(x), int(y)))

    cv2.namedWindow(win, cv2.WINDOW_NORMAL)
    cv2.namedWindow(win_depth, cv2.WINDOW_NORMAL)
    cv2.setMouseCallback(win, on_mouse)

    print("操作说明: w上升, s下降, y保存, r清空点, q/ESC退出")

    try:
        while True:
            obs = env.get_observation(
                include_arm=False, include_base=True, include_camera=True)
            cam_key, frame = find_color_frame(obs, prefer_cam=args.camera)
            if frame is None or cam_key is None:
                cv2.waitKey(1)
                continue
            _, depth = find_depth_frame(obs, prefer_cam=args.camera)

            raw_img = frame.copy()
            marked = draw_points(raw_img, points)

            tip = f"h={base_height:.2f} | points={len(points)} | w/s height | y save"
            cv2.putText(marked, tip, (12, 28), cv2.FONT_HERSHEY_SIMPLEX,
                        0.7, (0, 0, 255), 2, cv2.LINE_AA)
            cv2.imshow(win, marked)
            if depth is not None:
                depth_vis, stats = depth_to_colormap(
                    depth, args.depth_min, args.depth_max, args.depth_unit)
                if depth_vis is not None:
                    if stats:
                        info = f"z(m) min={stats['min']:.2f} med={stats['med']:.2f} max={stats['max']:.2f}"
                        cv2.putText(depth_vis, info, (12, 24), cv2.FONT_HERSHEY_SIMPLEX,
                                    0.6, (255, 255, 255), 1, cv2.LINE_AA)
                        if stats["above_max"] > 0.5:
                            warn = f">50% > max ({args.depth_max:.2f}m). Increase --depth-max."
                            cv2.putText(depth_vis, warn, (12, 48), cv2.FONT_HERSHEY_SIMPLEX,
                                        0.6, (255, 255, 255), 1, cv2.LINE_AA)
                    cv2.imshow(win_depth, depth_vis)

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
            elif key == ord("y"):
                folder = save_sample(
                    out_dir=args.out_dir,
                    raw_img=raw_img,
                    marked_img=marked,
                    points=points,
                    base_height=base_height,
                    camera_key=cam_key,
                )
                points.clear()
                print(f"saved -> {folder}")
            elif key == ord("r"):
                points.clear()
                print("points cleared")

    finally:
        cv2.destroyAllWindows()
        env.close()


if __name__ == "__main__":
    main()
