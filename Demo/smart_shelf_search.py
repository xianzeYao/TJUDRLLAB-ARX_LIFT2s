from __future__ import annotations

import sys
import time
from pathlib import Path

import cv2

from demo_utils import run_push_away, step_base_duration
from nav_goal import nav_to_goal
from single_arm_pick_place import single_arm_pick_place
from move_away import move_away

ROOT_DIR = Path(__file__).resolve().parent.parent
ROS2_DIR = ROOT_DIR / "ARX_Realenv" / "ROS2"
if str(ROS2_DIR) not in sys.path:
    sys.path.append(str(ROS2_DIR))

from arx_ros2_env import ARXRobotEnv  # noqa: E402


def smart_shelf_search(
    arx: ARXRobotEnv,
    search_prompt: str,
    first_nav_height: int,
    nav_table_prompt: str,
    rotate_recover: bool,
    place_prompt: str,
    nav_debug: bool = True,
    debug_pick_place: bool = False,
    depth_median_n: int = 5,
):
    if not search_prompt:
        raise ValueError("search_prompt must be non-empty")
    if not nav_table_prompt:
        raise ValueError("nav_table_prompt must be non-empty")
    if not place_prompt:
        raise ValueError("place_prompt must be non-empty")

    first_nav_result = None
    second_nav_result = None
    pick_arm = None
    place_arm = None

    try:
        if "behind" in search_prompt.lower():
            distance = 0.4
        else:
            distance = 0.45
        first_nav_result = nav_to_goal(
            arx=arx,
            goal=search_prompt,
            distance=distance,
            lift_height=first_nav_height,
            rotate_recover=rotate_recover,
            offset=0.36,
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
        if "blue" in search_prompt.lower():
            result = move_away(
            arx=arx,
            pick_prompt="a blue box",
            debug_raw=True,
            depth_median_n=5,
            home_after_move = True,
            )
            # run_push_away(arx)
            search_prompt = "a blue box"
            _, _, pick_arm = single_arm_pick_place(
                arx=arx,
                pick_prompt=search_prompt,
                place_prompt="",
                arm_side=result.arm,
                item_type="normal object",
                debug=debug_pick_place,
                depth_median_n=depth_median_n,
                verify_completion=True,
                completion_retry_attempts=0,
            )
        else:    
            _, _, pick_arm = single_arm_pick_place(
                arx=arx,
                pick_prompt=search_prompt,
                place_prompt="",
                arm_side="fit",
                item_type="normal object",
                debug=debug_pick_place,
                depth_median_n=depth_median_n,
                verify_completion=True,
                completion_retry_attempts=0,
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

        step_base_duration(arx, 0.0, 0.0, -1.0, duration=7)
        current_height = arx.get_robot_status().get("base").height
        second_nav_result = nav_to_goal(
            arx=arx,
            goal=nav_table_prompt,
            distance=-0.1,
            rotate_recover=True,
            lift_height=current_height,
            offset=0.22,
            use_goal_z_for_lift=False,
            target_goal_z=0.1,
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
        step_base_duration(arx, 0.0, 0.0, -1.0, duration=3.1)
        # if "first" in place_prompt.lower():
        #     arx.step_lift(14.0)
        # elif "second" in place_prompt.lower():
        #     arx.step_lift(17.0)
        # elif "top" in place_prompt.lower():
        #     arx.step_lift(20.0)
        # else:
        #     arx.step_lift(14.0)
        if "white" in place_prompt.lower():
            arx.step_lift(0.0)
        else:
            arx.step_lift(14.0)
        resolved_place_prompt = place_prompt
        if "xx" in resolved_place_prompt and pick_arm in {"left", "right"}:
            location = "a little" +pick_arm
            resolved_place_prompt = resolved_place_prompt.replace("xx", location, 1)
        _, _, place_arm = single_arm_pick_place(
            arx=arx,
            pick_prompt="",
            place_prompt=resolved_place_prompt,
            arm_side=pick_arm,
            item_type="normal object",
            debug=debug_pick_place,
            depth_median_n=depth_median_n,

        )
        arx.set_special_mode(1)
        if place_arm is None:
            return {
                "success": False,
                "message": "place canceled or failed",
                "first_nav_result": first_nav_result,
                "second_nav_result": second_nav_result,
                "pick_arm": pick_arm,
                "place_arm": None,
            }
        step_base_duration(arx, 0.0, 0.0, -1.0, duration=5.1)
        step_base_duration(arx, 0.75, 0.0, 0.0, duration=6.2)
        step_base_duration(arx, 0.0, 0.0, -1.0, duration=5.1)
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
        camera_view=("camera_l", "camera_h", "camera_r"),
        img_size=(640, 480),
    )
    try:
        arx.reset()
        search_prompts = [
            "a blue object behind a rubik's cube",
            "a yellow glue",
            "a tennis ball",
        ]
        place_prompts = [
            "the blue plate",
            "the white plate",
            "the blue plate",
        ]

        for search_prompt, place_prompt in zip(search_prompts, place_prompts):
            result = smart_shelf_search(
                arx=arx,
                first_nav_height=14.0,
                search_prompt=search_prompt,
                nav_table_prompt="a brown coaster on the floor",
                place_prompt=place_prompt,
                rotate_recover=True,
                nav_debug=False,
                debug_pick_place=True,
                depth_median_n=10,
            )
            print(f"{search_prompt}: {result}")
            if not result["success"]:
                break
    finally:
        arx.close()


if __name__ == "__main__":
    main()
