from __future__ import annotations

import sys
import time
from pathlib import Path

import cv2

from demo_utils import step_base_duration
from nav_goal import nav_to_goal
from single_arm_pick_place import single_arm_pick_place

ROOT_DIR = Path(__file__).resolve().parent.parent
ROS2_DIR = ROOT_DIR / "ARX_Realenv" / "ROS2"
if str(ROS2_DIR) not in sys.path:
    sys.path.append(str(ROS2_DIR))

from arx_ros2_env import ARXRobotEnv  # noqa: E402


def smart_shelf_search(
    arx: ARXRobotEnv,
    search_prompt: str,
    first_nav_height: int,
    second_nav_height: int,
    place_prompt: str,
    nav_debug: bool = True,
    debug_pick_place: bool = False,
    depth_median_n: int = 5,
):
    if not search_prompt:
        raise ValueError("search_prompt must be non-empty")
    if not place_prompt:
        raise ValueError("place_prompt must be non-empty")

    first_nav_result = None
    second_nav_result = None
    pick_arm = None
    place_arm = None

    try:
        first_nav_result = nav_to_goal(
            arx=arx,
            goal=search_prompt,
            distance=0.6,
            lift_height=first_nav_height,
            offset=0.25,
            use_goal_z_for_lift=True,
            continuous=False,
            debug_raw=nav_debug,
            depth_median_n=depth_median_n,
        )
        if first_nav_result is None:
            return {
                "success": False,
                "message": "first nav_to_goal failed or canceled",
                "first_nav_result": None,
                "second_nav_result": None,
                "pick_arm": None,
                "place_arm": None,
            }

        time.sleep(1.0)
        _, _, pick_arm = single_arm_pick_place(
            arx=arx,
            pick_prompt=search_prompt,
            place_prompt="",
            arm_side="fit",
            item_type="cup",
            debug=debug_pick_place,
            depth_median_n=depth_median_n,
        )
        if pick_arm is None:
            return {
                "success": False,
                "message": "pick canceled or failed",
                "first_nav_result": first_nav_result,
                "second_nav_result": None,
                "pick_arm": None,
                "place_arm": None,
            }

        step_base_duration(arx, 0.0, 0.0, -1.0, duration=10)
        second_nav_result = nav_to_goal(
            arx=arx,
            goal=place_prompt,
            distance=0.6,
            lift_height=second_nav_height,
            offset=0.25,
            use_goal_z_for_lift=True,
            continuous=False,
            debug_raw=nav_debug,
            depth_median_n=depth_median_n,
        )
        if second_nav_result is None:
            return {
                "success": False,
                "message": "second nav_to_goal failed or canceled",
                "first_nav_result": first_nav_result,
                "second_nav_result": None,
                "pick_arm": pick_arm,
                "place_arm": None,
            }

        time.sleep(1.0)
        _, _, place_arm = single_arm_pick_place(
            arx=arx,
            pick_prompt="",
            place_prompt=place_prompt,
            arm_side=pick_arm,
            item_type="cup",
            debug=debug_pick_place,
            depth_median_n=depth_median_n,
        )
        if place_arm is None:
            return {
                "success": False,
                "message": "place canceled or failed",
                "first_nav_result": first_nav_result,
                "second_nav_result": second_nav_result,
                "pick_arm": pick_arm,
                "place_arm": None,
            }

        return {
            "success": True,
            "message": "smart shelf search completed",
            "first_nav_result": first_nav_result,
            "second_nav_result": second_nav_result,
            "pick_arm": pick_arm,
            "place_arm": place_arm,
        }
    finally:
        cv2.destroyAllWindows()


def main() -> None:
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
        search_prompts = [
            "a tennis ball",
            "a blue box",
            "a brown horse",
        ]
        place_prompt = "a white square plate"

        for idx, search_prompt in enumerate(search_prompts):
            result = smart_shelf_search(
                arx=arx,
                first_nav_height=16,
                second_nav_height=20,
                search_prompt=search_prompt,
                place_prompt=place_prompt,
                nav_debug=True,
                debug_pick_place=True,
                depth_median_n=5,
            )
            print(f"{search_prompt}: {result}")

            step_base_duration(arx, 0.0, 0.0, -1.0, duration=10)
    finally:
        arx.close()


if __name__ == "__main__":
    main()
