from nav_goal import nav_to_goal
from dual_swap import dual_swap, pick_tools, release_tools
import select
import sys
import termios
from typing import Optional
import numpy as np

sys.path.append("../ARX_Realenv/ROS2")  # noqa

from arx_ros2_env import ARXRobotEnv  # noqa


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


def nav_dual_sweep(
    arx: ARXRobotEnv,
    goal: str = "white paper balls",
    distance: float = 0.55,
    nav_lift_height: float = 5.0,
    nav_debug_raw: bool = False,
    swap_debug_raw: bool = False,
    nav_depth_median_n: int = 5,
    swap_depth_median_n: int = 10,
    vote_times: int = 5,
) -> None:
    old_settings = _init_keyboard()

    try:
        pick_tools(arx)

        while True:
            key = _get_key_nonblock()
            if key == "q":
                print("Stop signal received.")
                return

            key = _get_key_nonblock()
            if key == "q":
                print("Stop signal received.")
                return
            lift_action = {
                "left": np.array([0.05, 0, 0.1, 0, 0, 0, 0.0], dtype=np.float32),
                "right": np.array([0.05, 0, 0.1, 0, 0, 0, 0.0], dtype=np.float32),
            }
            arx.step_smooth_eef(lift_action)
            nav_result = nav_to_goal(
                arx,
                goal=goal,
                distance=distance,
                lift_height=nav_lift_height,
                continuous=False,
                debug_raw=nav_debug_raw,
                depth_median_n=nav_depth_median_n,
                vote_times=vote_times,
                rotate_search_on_miss=True,
            )
            if nav_result is None:
                print("nav_to_goal failed, retry next cycle")
                continue

            key = _get_key_nonblock()
            if key == "q":
                print("Stop signal received.")
                return
            arx.step_lift(0.0)
            swap_result: Optional[object] = dual_swap(
                arx,
                object_prompt=goal,
                debug_raw=swap_debug_raw,
                depth_median_n=swap_depth_median_n,
            )

            if swap_result is None:
                print("dual_swap failed, retry next cycle")
                continue
    finally:
        arx.step_lift(0.0)
        release_tools(arx)
        _restore_keyboard(old_settings)


def main():
    arx = ARXRobotEnv(
        duration_per_step=1.0 / 20.0,
        min_steps=20,
        max_v_xyz=0.2,
        max_a_xyz=0.25,
        max_v_rpy=0.5,
        max_a_rpy=0.8,
        camera_type="all",
        camera_view=("camera_h",),
        img_size=(640, 480),
    )
    try:
        arx.reset()
        nav_dual_sweep(
            arx,
            goal="nearest paper cup or nearest paper ball or nearest bottle on the floor",
            distance=0.5,
            nav_lift_height=0.0,
            nav_debug_raw=True,
            swap_debug_raw=True,
            nav_depth_median_n=2,
            swap_depth_median_n=5,
            vote_times=3,
        )
    finally:
        arx.close()


if __name__ == "__main__":
    main()
