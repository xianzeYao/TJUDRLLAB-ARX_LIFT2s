"""
采集眼在手外标定数据：
- 利用 ARXRobotEnv.get_observation() 同步取相机帧和末端姿态
- 每次按回车保存一组：camera.png（RGB） + meta.json(end_pos)
- 进入重力模式便于手拖动末端采样

约定：
end_pos 使用“初始法兰固定帧 R0”作为参考系（固定坐标系）。
"""
from __future__ import annotations

import time
import numpy as np
import cv2
from typing import Dict, Tuple
from pathlib import Path
import json
import argparse
import sys
sys.path.append("../ROS2")  # noqa
from arx_ros2_env import ARXRobotEnv


def rpy_to_matrix(roll: float, pitch: float, yaw: float) -> np.ndarray:
    """将 roll/pitch/yaw(ZYX) 转为旋转矩阵，方便后续位姿处理。"""
    cx, sx = np.cos(roll), np.sin(roll)
    cy, sy = np.cos(pitch), np.sin(pitch)
    cz, sz = np.cos(yaw), np.sin(yaw)
    rx = np.array([[1, 0, 0], [0, cx, -sx], [0, sx, cx]])
    ry = np.array([[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]])
    rz = np.array([[cz, -sz, 0], [sz, cz, 0], [0, 0, 1]])
    return rz @ ry @ rx


def save_sample(
    obs: Dict[str, np.ndarray],
    base_dir: Path,
    idx: int,
    side: str,
    target_size: Tuple[int, int] | None,
    save_depth: bool = False,
) -> bool:
    """保存单次观测数据（仅一张彩色相机帧 + end_pos）。"""
    sample_dir = base_dir / f"sample_{idx:04d}"
    sample_dir.mkdir(parents=True, exist_ok=True)

    meta = {}
    end_key = f"{side}_end_pos"
    if end_key in obs and obs[end_key] is not None:
        meta["end_pos"] = np.asarray(obs[end_key]).tolist()
    else:
        print(f"[{idx:04d}] 缺少 {end_key}，跳过保存。")
        return False

    # 仅保存一张彩色帧
    # 只挑彩色帧；若没有彩色帧则放弃本次
    color_candidates = [
        (key, img) for key, img in obs.items()
        if isinstance(img, np.ndarray) and "color" in key
    ]
    if not color_candidates:
        print(f"[{idx:04d}] 没有彩色帧，跳过保存。")
        return False
    color_key, img = color_candidates[0]
    # 强制保存为 RGB：输入通常来自 cv2/ROS，默认 BGR，这里转换成 RGB 再写盘
    # if img.ndim == 3 and img.shape[2] == 3:
    #     img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_path = sample_dir / "camera.png"
    cv2.imwrite(str(img_path), img)
    meta["color_key"] = color_key

    # 可选保存对齐深度
    if save_depth:
        depth_candidates = [
            (key, depth) for key, depth in obs.items()
            if isinstance(depth, np.ndarray) and "depth" in key
        ]
        if depth_candidates:
            depth_key, depth_img = depth_candidates[0]
            np.save(sample_dir / "depth.npy", depth_img)
            finite = depth_img[np.isfinite(depth_img)]
            if finite.size > 0:
                vmax = np.percentile(finite.astype(np.float32), 99)
                if vmax > 1e-6:
                    vis = cv2.convertScaleAbs(
                        depth_img.astype(np.float32), alpha=255.0 / vmax)
                    cv2.imwrite(str(sample_dir / "depth_vis.png"), vis)
            meta["depth_key"] = depth_key

    meta_path = sample_dir / "meta.json"
    meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(f"[{idx:04d}] 保存完成：{sample_dir}")
    return True


def main():
    parser = argparse.ArgumentParser(description="采集眼在手外标定数据")
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("new_right_calibration_data"),
        help="输出目录",
    )
    parser.add_argument(
        "--side",
        choices=["left", "right"],
        default="right",
        help="使用哪个末端的姿态作为 eef",
    )
    parser.add_argument(
        "--camera-view",
        nargs="+",
        default=["camera_h"],
        help="启用的相机 view（与 arx_env 一致）",
    )
    parser.add_argument(
        "--camera-type",
        choices=["color", "depth", "all"],
        default="color",
        help="采集彩色/深度/全部",
    )
    parser.add_argument(
        "--resize",
        nargs=2,
        type=int,
        metavar=("W", "H"),
        help="可选，保存前 resize 成指定尺寸",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=30,
        help="最多采集多少组，0 表示不限",
    )
    parser.add_argument(
        "--skip-home",
        action="store_true",
        help="跳过回 home，仅启动 IO 后就地采集",
    )
    parser.add_argument(
        "--save-depth",
        action="store_true",
        help="同时保存对齐深度（depth.npy 与 depth_vis.png）",
    )

    args = parser.parse_args()
    target_size = tuple(args.resize) if args.resize else (640, 480)
    args.out_dir.mkdir(parents=True, exist_ok=True)

    env = ARXRobotEnv(
        min_steps=20,
        max_v_xyz=0.25, max_a_xyz=0.20,
        max_v_rpy=0.3, max_a_rpy=1.00,
        camera_type=args.camera_type,
        camera_view=tuple(args.camera_view),
        dir=None,
        img_size=target_size,
    )
    time.sleep(2.0)  # 等待 IO 稳定
    if not args.skip_home:
        env.reset()
    else:
        print("已跳过回 home，仅启动通讯后采集。")

    # 启用重力模式，便于手动拖动末端采样
    env.set_special_mode(3)
    print("已切换至重力模式，可直接拖动末端调整姿态。")

    idx = 0
    print("按回车采集一帧，输入 q + 回车退出。")
    while True:
        if args.max_samples and idx >= args.max_samples:
            print("达到采集上限，退出。")
            break
        user_in = input(f"[{idx:04d}] > ")
        if user_in.strip().lower() in {"q", "quit", "exit"}:
            break
        obs = env.get_observation()
        if not obs:
            print("获取观测失败，已跳过。")
            continue
        saved = save_sample(
            obs=obs,
            base_dir=args.out_dir,
            idx=idx,
            side=args.side,
            target_size=target_size,
            save_depth=args.save_depth,
        )
        if saved:
            idx += 1

    env.close()
    print(f"采集结束，总计 {idx} 组，保存在 {args.out_dir}")


if __name__ == "__main__":
    main()
