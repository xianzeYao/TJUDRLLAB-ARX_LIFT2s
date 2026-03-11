import time
import sys
import select
import termios
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


def _get_key_nonblock():
    dr, _, _ = select.select([sys.stdin], [], [], 0)
    if dr:
        return sys.stdin.read(1)
    return None


def _init_keyboard():
    fd = sys.stdin.fileno()
    old_settings = termios.tcgetattr(fd)
    new_settings = termios.tcgetattr(fd)
    new_settings[3] = new_settings[3] & ~termios.ICANON & ~termios.ECHO
    termios.tcsetattr(fd, termios.TCSADRAIN, new_settings)
    return old_settings


def _restore_keyboard(old_settings):
    termios.tcsetattr(sys.stdin.fileno(), termios.TCSADRAIN, old_settings)


def _confirm_continue_swap(sweep_idx: int, max_sweeps: int) -> bool:
    win = "dual_swap_confirm"
    cv2.namedWindow(win, cv2.WINDOW_NORMAL)
    prompt = "Continue sweep or not?"
    try:
        while True:
            panel = np.zeros((120, 900, 3), dtype=np.uint8)
            cv2.putText(
                panel,
                prompt,
                (20, 55),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 255, 0),
                2,
            )
            cv2.putText(
                panel,
                "y: continue   n: stop",
                (20, 95),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255, 255, 255),
                2,
            )
            cv2.imshow(win, panel)

            key = _get_key_nonblock()
            if key in ("y", "Y"):
                return True
            if key in ("n", "N"):
                return False

            cv_key = cv2.waitKey(50) & 0xFF
            if cv_key in (ord("y"), ord("Y")):
                return True
            if cv_key in (ord("n"), ord("N"), 27):
                return False
    finally:
        cv2.destroyWindow(win)


def pick_tools(arx: ARXRobotEnv) -> None:
    open_action = {
        "left": np.array([0.05, 0, 0, 0, 0, 0, -3.4], dtype=np.float32),
        "right": np.array([0.05, 0, 0, 0, 0, 0, -3.4], dtype=np.float32),
    }
    arx.step(open_action)
    print("请放取扫把簸箕，5秒后开始夹取...")
    time.sleep(5.0)

    close_action = {
        "left": np.array([0.05, 0, 0, 0, 0, 0, 0.0], dtype=np.float32),
        "right": np.array([0.05, 0, 0, 0, 0, 0, 0.0], dtype=np.float32),
    }
    arx.step(close_action)
    time.sleep(5.0)

    lift_action = {
        "left": np.array([0.05, 0, 0.1, 0, 0, 0, 0.0], dtype=np.float32),
        "right": np.array([0.05, 0, 0.1, 0, 0, 0, 0.0], dtype=np.float32),
    }
    arx.step(lift_action)
    time.sleep(1.0)


def release_tools(arx: ARXRobotEnv) -> None:
    success, error_message = arx.set_special_mode(1)
    if not success:
        raise RuntimeError(f"Failed to home both arms: {error_message}")
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
        while True:
            key = _get_key_nonblock()
            if key == "q":
                cv2.destroyWindow(win)
                return None

            cv_key = cv2.waitKey(50)
            if cv_key < 0:
                continue

            cv2.destroyWindow(win)
            if cv_key == ord("r"):
                break
            if cv_key == ord("q"):
                return None
            return trash_base_point
        continue


def dual_swap(
    arx: ARXRobotEnv,
    object_prompt: str = "a white crumpled paper on the floor",
    debug_raw: bool = True,
    depth_median_n: int = 10,
) -> Optional[np.ndarray]:
    old_settings = _init_keyboard()
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
        if not swap_seq:
            return target_base_point

        arx.step(swap_seq[0])
        sweep_actions = swap_seq[1:]
        actions_per_sweep = 4
        max_sweeps = len(sweep_actions) // actions_per_sweep
        for sweep_idx in range(max_sweeps):
            start = sweep_idx * actions_per_sweep
            end = start + actions_per_sweep
            for action in sweep_actions[start:end]:
                arx.step(action)
            if sweep_idx == max_sweeps - 1:
                break
            if not _confirm_continue_swap(sweep_idx + 1, max_sweeps):
                break
        return target_base_point
    finally:
        cv2.destroyAllWindows()
        _restore_keyboard(old_settings)


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
            depth_median_n=5,
        )
        release_tools(arx)
    finally:
        arx.close()


if __name__ == "__main__":
    main()
