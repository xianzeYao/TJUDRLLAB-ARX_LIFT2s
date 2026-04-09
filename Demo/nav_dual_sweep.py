from nav_goal import nav_to_goal
from dual_swap import dual_swap, pick_tools, release_tools
import sys
from typing import Optional
import numpy as np
from visualize_utils import (
    VisualizeContext,
    get_key_nonblock,
    init_keyboard,
    restore_keyboard,
    should_stop,
)
from sweep_prompt_parser import parse_human_sweep_request

sys.path.append("../ARX_Realenv/ROS2")  # noqa

from arx_ros2_env import ARXRobotEnv  # noqa


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
    visualize: Optional[VisualizeContext] = None,
) -> None:
    old_settings = init_keyboard()
    cycle_idx = 0
    sweep_plan = parse_human_sweep_request(goal)
    goal_prompt = sweep_plan.goal_prompt
    quit_requested = False
    base_stop_checker = None if visualize is None else visualize.stop_checker

    def _stop_checker() -> bool:
        nonlocal quit_requested
        if quit_requested:
            return True
        if base_stop_checker is not None:
            try:
                if base_stop_checker():
                    return True
            except Exception:
                pass
        if get_key_nonblock() == "q":
            quit_requested = True
            return True
        return False

    task_visualize = VisualizeContext(
        stop_checker=_stop_checker,
    )

    def _should_stop() -> bool:
        return should_stop(task_visualize)

    try:
        if sweep_plan.intent != "direct_prompt" or sweep_plan.raw_request != sweep_plan.goal_prompt:
            resolution_message = (
                "Resolved nav dual sweep request: "
                f"{sweep_plan.raw_request} -> {sweep_plan.goal_prompt}"
            )
            print(resolution_message)
        pick_tools(arx, visualize=task_visualize)

        while True:
            cycle_idx += 1
            if _should_stop():
                print("Stop signal received.")
                return

            if _should_stop():
                print("Stop signal received.")
                return
            lift_action = {
                "left": np.array([0.05, 0, 0.1, 0, 0, 0, 0.0], dtype=np.float32),
                "right": np.array([0.05, 0, 0.1, 0, 0, 0, 0.0], dtype=np.float32),
            }
            arx.step_smooth_eef(lift_action)
            nav_result = nav_to_goal(
                arx,
                goal=goal_prompt,
                distance=distance,
                lift_height=nav_lift_height,
                offset=0.47,
                continuous=False,
                debug_raw=nav_debug_raw,
                depth_median_n=nav_depth_median_n,
                vote_times=vote_times,
                rotate_search_on_miss=True,
                use_initial_search_roi=True,
                visualize=task_visualize,
            )
            if nav_result is None:
                if _should_stop():
                    print("Stop signal received.")
                    return
                print("nav_to_goal failed, retry next cycle")
                continue

            if _should_stop():
                print("Stop signal received.")
                return
            arx.step_lift(0.0)
            swap_result: Optional[object] = dual_swap(
                arx,
                object_prompt=goal_prompt,
                debug_raw=swap_debug_raw,
                depth_median_n=swap_depth_median_n,
                visualize=task_visualize,
            )

            if swap_result is None:
                if _should_stop():
                    print("Stop signal received.")
                    return
                print("dual_swap failed, retry next cycle")
                continue
    finally:
        arx.step_lift(0.0)
        release_tools(arx, visualize=task_visualize)
        restore_keyboard(old_settings)


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
            goal="paper cup or paper ball or bottle on the floor",
            distance=0.55,
            nav_lift_height=0.0,
            nav_debug_raw=True,
            swap_debug_raw=True,
            nav_depth_median_n=5,
            swap_depth_median_n=5,
            vote_times=3,
        )
    finally:
        arx.close()


if __name__ == "__main__":
    main()
