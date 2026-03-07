import select
import sys
import termios
from typing import Optional

sys.path.append("../ARX_Realenv/ROS2")  # noqa

from arx_ros2_env import ARXRobotEnv  # noqa
from dual_swap import dual_swap, pick_tools, release_tools
from nav_goal import nav_to_goal


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
    nav_debug_raw: bool = False,
    swap_debug_raw: bool = False,
    nav_depth_median_n: int = 5,
    swap_depth_median_n: int = 10,
    vote_times: int = 5,
) -> None:
    old_settings = _init_keyboard()
    tools_picked = False

    try:
        while True:
            key = _get_key_nonblock()
            if key == "q":
                print("Stop signal received.")
                if tools_picked:
                    release_tools(arx)
                return

            pick_tools(arx)
            tools_picked = True

            key = _get_key_nonblock()
            if key == "q":
                print("Stop signal received.")
                release_tools(arx)
                return

            nav_result = nav_to_goal(
                arx,
                goal=goal,
                distance=distance,
                continuous=False,
                debug_raw=nav_debug_raw,
                depth_median_n=nav_depth_median_n,
                vote_times=vote_times,
            )
            if nav_result is None:
                print("nav_to_goal failed, release tools and retry")
                release_tools(arx)
                tools_picked = False
                continue

            key = _get_key_nonblock()
            if key == "q":
                print("Stop signal received.")
                release_tools(arx)
                return

            swap_result: Optional[object] = dual_swap(
                arx,
                object_prompt=goal,
                debug_raw=swap_debug_raw,
                depth_median_n=swap_depth_median_n,
            )
            release_tools(arx)
            tools_picked = False

            if swap_result is None:
                print("dual_swap failed, retry next cycle")
                continue
    finally:
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
        nav_dual_sweep(
            arx,
            goal="white paper balls",
            distance=0.55,
            nav_debug_raw=False,
            swap_debug_raw=True,
            nav_depth_median_n=5,
            swap_depth_median_n=10,
            vote_times=5,
        )
    finally:
        arx.close()


if __name__ == "__main__":
    main()
