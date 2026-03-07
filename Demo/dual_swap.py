import time
import sys
from typing import Optional

import cv2
import numpy as np

sys.path.append("../ARX_Realenv/ROS2")  # noqa

from arx_ros2_env import ARXRobotEnv  # noqa
from arx_pointing import predict_point_from_rgb
from motion_swap import build_swap_sequence
from point2pos_utils import (
    get_aligned_frames,
    pixel_to_base_point_safe,
)


def pick_tools(arx: ARXRobotEnv) -> None:
    open_action = {
        "left": np.array([0, 0, 0, 0, 0, 0, -3.4], dtype=np.float32),
        "right": np.array([0, 0, 0, 0, 0, 0, -3.4], dtype=np.float32),
    }
    arx.step(open_action)
    print("请放取扫把簸箕，5秒后开始夹取...")
    time.sleep(5.0)

    close_action = {
        "left": np.array([0, 0, 0, 0, 0, 0, 0.0], dtype=np.float32),
        "right": np.array([0, 0, 0, 0, 0, 0, 0.0], dtype=np.float32),
    }
    arx.step(close_action)
    time.sleep(5.0)

    lift_action = {
        "left": np.array([0, 0, 0.1, 0, 0, 0, 0.0], dtype=np.float32),
        "right": np.array([0, 0, 0.1, 0, 0, 0, 0.0], dtype=np.float32),
    }
    arx.step(lift_action)
    time.sleep(1.0)


def release_tools(arx: ARXRobotEnv) -> None:
    home_action = {
        "left": np.array([0, 0, 0, 0, 0, 0, 0.0], dtype=np.float32),
        "right": np.array([0, 0, 0, 0, 0, 0, 0.0], dtype=np.float32),
    }
    arx.step(home_action)
    time.sleep(1.0)

    open_action = {
        "left": np.array([0, 0, 0, 0, 0, 0, -3.4], dtype=np.float32),
        "right": np.array([0, 0, 0, 0, 0, 0, -3.4], dtype=np.float32),
    }
    arx.step(open_action)
    time.sleep(5.0)


def _detect_swap_target(
    arx: ARXRobotEnv,
    object_prompt: str,
    debug_raw: bool,
    depth_median_n: int,
) -> Optional[np.ndarray]:
    while True:
        color, depth = get_aligned_frames(arx, depth_median_n=depth_median_n)
        if color is None or depth is None:
            continue

        trash_point = predict_point_from_rgb(color, object_prompt)
        u, v = int(round(trash_point[0])), int(round(trash_point[1]))
        trash_base_point = pixel_to_base_point_safe(
            (u, v),
            depth,
            robot_part="center",
        )
        if trash_base_point is None:
            print(f"预测像素 {(u, v)} 深度无效或像素越界，自动刷新")
            continue

        if not debug_raw:
            return trash_base_point

        vis = color.copy()
        cv2.circle(vis, (u, v), 3, (0, 0, 255), -1)
        win = "dual_swap_detect"
        cv2.namedWindow(win, cv2.WINDOW_NORMAL)
        cv2.imshow(win, vis)
        key = cv2.waitKey(0)
        cv2.destroyWindow(win)
        if key == ord("r"):
            continue
        if key == ord("q"):
            return None
        return trash_base_point


def dual_swap(
    arx: ARXRobotEnv,
    object_prompt: str = "a white crumpled paper on the floor",
    debug_raw: bool = True,
    depth_median_n: int = 10,
) -> Optional[np.ndarray]:
    try:
        target_base_point = _detect_swap_target(
            arx,
            object_prompt=object_prompt,
            debug_raw=debug_raw,
            depth_median_n=depth_median_n,
        )
        if target_base_point is None:
            return None

        swap_seq = build_swap_sequence(target_base_point)
        for action in swap_seq:
            arx.step(action)
        return target_base_point
    finally:
        cv2.destroyAllWindows()


def main():
    arx = ARXRobotEnv(
        duration_per_step=1.0 / 20.0,
        min_steps=20,
        max_v_xyz=0.15,
        max_a_xyz=0.20,
        max_v_rpy=0.45,
        max_a_rpy=1.00,
        camera_type="all",
        camera_view=("camera_h",),
        img_size=(640, 480),
    )
    try:
        arx.reset()
        pick_tools(arx)
        dual_swap(
            arx,
            object_prompt="a white crumpled paper on the floor",
            debug_raw=True,
            depth_median_n=10,
        )
        release_tools(arx)
    finally:
        arx.close()


if __name__ == "__main__":
    main()
