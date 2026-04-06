from __future__ import annotations

import sys
from pathlib import Path
from typing import Optional

import cv2

from demo_utils import step_base_duration
from nav_goal import nav_to_goal
from move_away import move_away
from shelf_prompt_parser import parse_human_shelf_request
from single_arm_pick_place import single_arm_pick_place
from visualize_utils import (
    VisualizeContext,
    emit_event,
    emit_log,
    emit_result,
    emit_stage,
    should_stop,
)

ROOT_DIR = Path(__file__).resolve().parent.parent
ROS2_DIR = ROOT_DIR / "ARX_Realenv" / "ROS2"
if str(ROS2_DIR) not in sys.path:
    sys.path.append(str(ROS2_DIR))

from arx_ros2_env import ARXRobotEnv  # noqa: E402


def smart_shelf_search(
    arx: ARXRobotEnv,
    nav_pick_prompt: str,
    first_nav_height: int,
    nav_waypoint_prompt: str,
    rotate_recover: bool,
    place_target_prompt: str,
    nav_debug: bool = True,
    debug_pick_place: bool = False,
    depth_median_n: int = 5,
    visualize: Optional[VisualizeContext] = None,
):
    if not nav_pick_prompt:
        raise ValueError("nav_pick_prompt must be non-empty")
    if not nav_waypoint_prompt:
        raise ValueError("nav_waypoint_prompt must be non-empty")
    if not place_target_prompt:
        raise ValueError("place_target_prompt must be non-empty")

    first_nav_result = None
    second_nav_result = None
    pick_arm = None
    place_arm = None

    try:
        emit_stage(
            visualize,
            source="smart_shelf_search",
            stage="start",
            message="Start smart shelf search",
            nav_pick_prompt=nav_pick_prompt,
            nav_waypoint_prompt=nav_waypoint_prompt,
            place_target_prompt=place_target_prompt,
        )
        pick_prompt = nav_pick_prompt
        if "behind" in nav_pick_prompt.lower():
            distance = 0.425
        else:
            distance = 0.45
        if should_stop(visualize):
            return {
                "success": False,
                "message": "stopped before first navigation",
                "first_nav_result": None,
                "second_nav_result": None,
                "pick_arm": None,
                "place_arm": None,
            }
        first_nav_result = nav_to_goal(
            arx=arx,
            goal=nav_pick_prompt,
            distance=distance,
            lift_height=first_nav_height,
            rotate_recover=rotate_recover,
            offset=0.23,
            use_goal_z_for_lift=True,
            continuous=False,
            debug_raw=nav_debug,
            depth_median_n=depth_median_n,
            use_initial_search_roi=False,
            visualize=visualize,
        )
        if first_nav_result is None:
            emit_result(
                visualize,
                source="smart_shelf_search",
                status="failed",
                message="first nav_to_goal failed or canceled",
            )
            return {
                "success": False,
                "message": "first nav_to_goal failed or canceled",
                "first_nav_result": None,
                "second_nav_result": None,
                "pick_arm": None,
                "place_arm": None,
            }
        if "blue" in nav_pick_prompt.lower():
            emit_stage(
                visualize,
                source="smart_shelf_search",
                stage="move_away",
                message="Check and move away front blocker",
            )
            result = move_away(
                arx=arx,
                pick_prompt="a blue box",
                debug_raw=debug_pick_place and not (
                    visualize is not None and visualize.page_debug
                ),
                depth_median_n=5,
                home_after_move=True,
                visualize=visualize,
            )
            emit_log(
                visualize,
                source="smart_shelf_search",
                stage="move_away",
                message=(
                    f"move_away blocked={result.blocked}, "
                    f"moved_away={result.moved_away}, "
                    f"arm={result.arm}, message={result.message}"
                ),
            )
            pick_prompt = "center of a blue box"
            _, _, pick_arm = single_arm_pick_place(
                arx=arx,
                pick_prompt=pick_prompt,
                place_prompt="",
                arm_side=result.arm,
                item_type="normal object",
                debug=debug_pick_place,
                depth_median_n=depth_median_n,
                verify_completion=True,
                completion_retry_attempts=0,
                visualize=visualize,
            )
        else:
            _, _, pick_arm = single_arm_pick_place(
                arx=arx,
                pick_prompt=pick_prompt,
                place_prompt="",
                arm_side="fit",
                item_type="normal object",
                debug=debug_pick_place,
                depth_median_n=depth_median_n,
                verify_completion=True,
                completion_retry_attempts=0,
                visualize=visualize,
            )
        if pick_arm is None:
            emit_result(
                visualize,
                source="smart_shelf_search",
                status="failed",
                message="pick canceled or failed",
            )
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
        if should_stop(visualize):
            return {
                "success": False,
                "message": "stopped before second navigation",
                "first_nav_result": first_nav_result,
                "second_nav_result": None,
                "pick_arm": pick_arm,
                "place_arm": None,
            }
        second_nav_result = nav_to_goal(
            arx=arx,
            goal=nav_waypoint_prompt,
            distance=-0.1,
            rotate_recover=True,
            lift_height=current_height,
            offset=0.22,
            use_goal_z_for_lift=False,
            target_goal_z=0.1,
            continuous=False,
            debug_raw=nav_debug,
            depth_median_n=depth_median_n,
            use_initial_search_roi=False,
            visualize=visualize,
        )
        if second_nav_result is None:
            emit_result(
                visualize,
                source="smart_shelf_search",
                status="failed",
                message="second nav_to_goal failed or canceled",
            )
            return {
                "success": False,
                "message": "second nav_to_goal failed or canceled",
                "first_nav_result": first_nav_result,
                "second_nav_result": None,
                "pick_arm": pick_arm,
                "place_arm": None,
            }
        step_base_duration(arx, 0.0, 0.0, -1.0, duration=3.1)
        # if "first" in place_target_prompt.lower():
        #     arx.step_lift(14.0)
        # elif "second" in place_target_prompt.lower():
        #     arx.step_lift(17.0)
        # elif "top" in place_target_prompt.lower():
        #     arx.step_lift(20.0)
        # else:
        #     arx.step_lift(14.0)
        if "white" in place_target_prompt.lower():
            arx.step_lift(0.0)
        else:
            arx.step_lift(14.5)
        resolved_place_prompt = place_target_prompt
        if "xx" in resolved_place_prompt and pick_arm in {"left", "right"}:
            location = pick_arm
            resolved_place_prompt = resolved_place_prompt.replace(
                "xx", location, 1)
        _, _, place_arm = single_arm_pick_place(
            arx=arx,
            pick_prompt="",
            place_prompt=resolved_place_prompt,
            arm_side=pick_arm,
            item_type="normal object",
            debug=debug_pick_place,
            depth_median_n=depth_median_n,
            visualize=visualize,
        )
        arx.set_special_mode(1)
        if place_arm is None:
            emit_result(
                visualize,
                source="smart_shelf_search",
                status="failed",
                message="place canceled or failed",
            )
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
        emit_result(
            visualize,
            source="smart_shelf_search",
            status="success",
            message="smart shelf search completed",
            pick_arm=pick_arm,
            place_arm=place_arm,
        )
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


def smart_shelf_search_from_request(
    arx: ARXRobotEnv,
    request: str,
    first_nav_height: int,
    rotate_recover: bool,
    nav_debug: bool = True,
    debug_pick_place: bool = False,
    depth_median_n: int = 5,
    visualize: Optional[VisualizeContext] = None,
):
    plan = parse_human_shelf_request(request)
    emit_event(
        visualize,
        "parsed_request",
        source="smart_shelf_search",
        raw_request=plan.raw_request,
        nav_pick_prompt=plan.nav_pick_prompt,
        nav_waypoint_prompt=plan.nav_waypoint_prompt,
        place_target_prompt=plan.place_target_prompt,
        pick_target=plan.pick_target,
        used_default_place_target_prompt=plan.used_default_place_target_prompt,
        parser_source=plan.parser_source,
    )
    result = smart_shelf_search(
        arx=arx,
        nav_pick_prompt=plan.nav_pick_prompt,
        first_nav_height=first_nav_height,
        nav_waypoint_prompt=plan.nav_waypoint_prompt,
        rotate_recover=rotate_recover,
        place_target_prompt=plan.place_target_prompt,
        nav_debug=nav_debug,
        debug_pick_place=debug_pick_place,
        depth_median_n=depth_median_n,
        visualize=visualize,
    )
    result["resolved_prompts"] = {
        "raw_request": plan.raw_request,
        "nav_pick_prompt": plan.nav_pick_prompt,
        "nav_waypoint_prompt": plan.nav_waypoint_prompt,
        "place_target_prompt": plan.place_target_prompt,
        "pick_target": plan.pick_target,
        "used_default_place_target_prompt": plan.used_default_place_target_prompt,
        "parser_source": plan.parser_source,
    }
    return result


def main() -> None:
    arx = ARXRobotEnv(
        duration_per_step=1.0 / 20.0,
        min_steps=20,
        max_v_xyz=0.15,
        max_a_xyz=0.1,
        max_v_rpy=0.5,
        max_a_rpy=0.6,
        camera_type="all",
        camera_view=("camera_l", "camera_h", "camera_r"),
        img_size=(640, 480),
    )
    try:
        arx.reset()
        nav_pick_prompts = [
            "a blue box behind a rubik's cube",
            "a tennis ball",
            "a yellow screwdriver",
        ]
        place_target_prompts = [
            "the center part of the blue plate",
            "the center part of the white plate",
            "the center part of the blue plate",
        ]

        # for nav_pick_prompt, place_target_prompt in zip(
        #     nav_pick_prompts,
        #     place_target_prompts,
        # ):
        #     result = smart_shelf_search(
        #         arx=arx,
        #         nav_pick_prompt=nav_pick_prompt,
        #         first_nav_height=14.5,
        #         nav_waypoint_prompt="a brown coaster on the floor",
        #         place_target_prompt=place_target_prompt,
        #         rotate_recover=True,
        #         nav_debug=False,
        #         debug_pick_place=False,
        #         depth_median_n=10,
        #     )

        #     print(f"{nav_pick_prompt}: {result}")
        #     if not result["success"]:
        #         break

        result = smart_shelf_search_from_request(
            arx=arx,
            request=f"我要一个蓝色盒子",
            first_nav_height=14.5,
            rotate_recover=True,
            nav_debug=True,
            debug_pick_place=True,
            depth_median_n=10,
        )
        if not result["success"]:
            print(f"Request failed: {result}")
    finally:
        arx.close()


if __name__ == "__main__":
    main()
