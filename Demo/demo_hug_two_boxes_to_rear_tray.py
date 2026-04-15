#!/usr/bin/env python3
from __future__ import annotations

import sys
import time
from typing import Literal

import numpy as np

sys.path.append("../ARX_Realenv/ROS2")  # noqa

from arx_ros2_env import ARXRobotEnv
from arx_pointing import predict_point_from_rgb
from dual_arm_hug_pick_place import dual_arm_hug_pick, dual_arm_hug_place
from nav_goal import _vote_goal_presence
from point2pos_utils import get_aligned_frames, pixel_to_base_point_safe
from visualize_utils import (
    VisualizeContext,
    dispatch_debug_image,
    render_multi_points_debug_view,
    should_stop,
)


def _step_base_for_duration(
    arx: ARXRobotEnv,
    *,
    vx: float,
    vy: float,
    vz: float,
    duration_s: float,
) -> None:
    if duration_s <= 0:
        return
    arx.step_base(vx=float(vx), vy=float(vy), vz=float(vz))
    try:
        time.sleep(float(duration_s))
    finally:
        arx.step_base(0.0, 0.0, 0.0)


def recenter_base_by_point(
    arx: ARXRobotEnv,
    target_prompt: str,
    *,
    vy_cmd: float,
    lateral_speed_mps: float,
    depth_median_n: int,
    max_iters: int = 2,
    debug: bool = False,
    visualize: VisualizeContext | None = None,
) -> bool:
    """After stopping, point once, project to base, then move laterally (vy) to recenter."""
    if lateral_speed_mps <= 0:
        raise ValueError("lateral_speed_mps must be > 0")
    if vy_cmd <= 0:
        raise ValueError("vy_cmd must be > 0")
    if depth_median_n <= 0:
        raise ValueError("depth_median_n must be > 0")
    if max_iters <= 0:
        raise ValueError("max_iters must be > 0")

    for _ in range(int(max_iters)):
        if should_stop(visualize):
            return False

        color, depth = get_aligned_frames(arx, depth_median_n=depth_median_n)
        if color is None or depth is None:
            continue

        try:
            prompt = (
                f"Point to the center of this object: {target_prompt}. "
                "Return only JSON."
            )
            u, v = predict_point_from_rgb(
                color,
                text_prompt="",
                all_prompt=prompt,
                assume_bgr=False,
                temperature=0.0,
            )
            target_px = (int(round(u)), int(round(v)))
        except Exception:
            continue

        goal_pw = pixel_to_base_point_safe(
            target_px,
            depth,
            robot_part="center",
        )
        if goal_pw is None:
            continue

        goal_y = float(goal_pw[1])
        dy = goal_y

        if debug:
            vis = render_multi_points_debug_view(
                color=color,
                points=[target_px],
                title=f"Target: {target_prompt}",
            )
            debug_result = dispatch_debug_image(
                visualize,
                source="demo_hug_two_boxes_to_rear_tray",
                panel="nav",
                image=vis,
                window_name="recenter_base_by_point",
                target_prompt=target_prompt,
                goal_pw=[float(goal_pw[0]), goal_y, float(goal_pw[2])],
            )
            if debug_result is None:
                return False
            if not debug_result:
                # Re-point / re-project.
                continue

        # Only lateral correction: command vy is opposite to projected goal_y.
        vy = -float(vy_cmd) if dy > 0.0 else float(vy_cmd)
        _step_base_for_duration(
            arx,
            vx=0.0,
            vy=vy,
            vz=0.0,
            duration_s=abs(dy) / float(lateral_speed_mps),
        )

        # Re-check next iteration.
        time.sleep(0.1)

    return True


def shift_until_target_detected(
    arx: ARXRobotEnv,
    target_prompt: str,
    direction: Literal["left", "right"],
    speed: float,
    max_duration_s: float,
    check_interval_s: float,
    vote_times: int,
    debug: bool = False,
    visualize: VisualizeContext | None = None,
) -> bool:
    """Move laterally while repeatedly checking target presence; stop immediately once found."""
    if direction not in {"left", "right"}:
        raise ValueError("direction must be 'left' or 'right'")
    if speed <= 0:
        raise ValueError("speed must be > 0")
    if max_duration_s <= 0:
        raise ValueError("max_duration_s must be > 0")
    if check_interval_s <= 0:
        raise ValueError("check_interval_s must be > 0")
    if vote_times <= 0:
        raise ValueError("vote_times must be > 0")

    # Convention (per user): move right -> positive vy, move left -> negative vy.
    vy_cmd = abs(speed) if direction == "right" else -abs(speed)

    start = time.time()
    arx.step_base(vx=0.0, vy=vy_cmd, vz=0.0)
    try:
        while True:
            if should_stop(visualize):
                return False

            elapsed = time.time() - start
            if elapsed >= max_duration_s:
                return False

            frames = arx.get_camera(target_size=(640, 480), return_status=False)
            color = frames.get("camera_h_color") if isinstance(frames, dict) else None
            if color is not None and _vote_goal_presence(color, target_prompt, vote_times=vote_times):
                print(f"Target detected")
                # Use the same color frame (used for voting) to predict one point.
                try:
                    prompt = (
                        f"Point to the center of this object: {target_prompt}. "
                        "Return only JSON."
                    )
                    u, v = predict_point_from_rgb(
                        color,
                        text_prompt="",
                        all_prompt=prompt,
                        assume_bgr=False,
                        temperature=0.0,
                    )
                    target_px = (int(round(u)), int(round(v)))
                except Exception:
                    target_px = None
                if target_px is None:
                    # Presence is true but point is unavailable; keep scanning.
                    arx.step_base(vx=0.0, vy=vy_cmd, vz=0.0)
                    time.sleep(check_interval_s)
                    continue

                arx.step_base(0.0, 0.0, 0.0)

                if debug:
                    vis = render_multi_points_debug_view(
                        color=color,
                        points=[target_px],
                        title=f"Target: {target_prompt}",
                    )

                    debug_result = dispatch_debug_image(
                        visualize,
                        source="demo_hug_two_boxes_to_rear_tray",
                        panel="nav",
                        image=vis,
                        window_name="shift_until_target_detected",
                        target_prompt=target_prompt,
                        direction=direction,
                        vy_cmd=float(vy_cmd),
                    )
                    if debug_result is None:
                        return False
                    if not debug_result:
                        # Continue scanning.
                        arx.step_base(vx=0.0, vy=vy_cmd, vz=0.0)
                        time.sleep(check_interval_s)
                        continue

                # Post-stop recenter: point -> base coords -> move base.
                recentered = recenter_base_by_point(
                    arx,
                    target_prompt,
                    vy_cmd=0.75,
                    lateral_speed_mps=0.125,
                    depth_median_n=5,
                    max_iters=2,
                    debug=debug,
                    visualize=visualize,
                )
                if not recentered:
                    return False

                return True

            # Re-send command in loop to keep behavior consistent with scan-style control.
            arx.step_base(vx=0.0, vy=vy_cmd, vz=0.0)
            time.sleep(check_interval_s)
    finally:
        arx.step_base(0.0, 0.0, 0.0)


def reset_dual_arms(arx: ARXRobotEnv, *, open_gripper: bool = True) -> None:
    for arm in ("left", "right"):
        success, error_message = arx.set_special_mode(1, side=arm)
        if not success:
            raise RuntimeError(f"Failed to home {arm} arm: {error_message}")
        if open_gripper:
            home_open = np.array([0, 0, 0, 0, 0, 0, -3.4], dtype=np.float32)
            arx.step_smooth_eef({arm: home_open})
    time.sleep(1.0)




def main() -> None:
    """Program entry: execute only the first hug pick stage."""
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
        debug = True
        arx.reset()
        # Raise lift first so the head camera can observe the front stacked boxes.
        arx.step_lift(17.0)
        time.sleep(1.0)

        left_ref, right_ref = dual_arm_hug_pick(
            arx,
            object_prompt="the top box in front of the robot",
            item_type="hug",
            debug=debug,
            depth_median_n=5,
        )
        if left_ref is None or right_ref is None:
            print("first hug pick canceled or failed")
            return

        print("first hug pick completed")

        print("start lateral shift to place side")
        arx.step_lift(17.0)
        time.sleep(1.0)

        place_prompt = "a gray coaster"
        found_target = shift_until_target_detected(
            arx,
            target_prompt=f"{place_prompt}, if you think there are none, output FALSE",
            direction="right",
            speed=0.75,
            max_duration_s=6.0,
            check_interval_s=0.125,
            vote_times=2,
            debug=debug,
        )
        
        if not found_target:
            print("lateral shift finished but target was not detected")
            return

        print("lateral shift completed")

        # Place (put the hugged box down) onto the target.
        place_left_ref, place_right_ref = dual_arm_hug_place(
            arx,
            object_prompt=place_prompt,
            item_type="hug",
            debug=debug,
            depth_median_n=5,
        )
        if place_left_ref is None or place_right_ref is None:
            print("hug place canceled or failed")
            return

        print("hug place completed")

        # Reset dual arms, go back left to find the next box, align, then shift right to continue.
        reset_dual_arms(arx, open_gripper=False)
        arx.step_lift(17.0)
        time.sleep(1.0)
        _step_base_for_duration(
            arx,
            vx=0.0,
            vy=-0.75,
            vz=0.0,
            duration_s=3.0,
        )

        box_prompt = "a cardboard wood box"
        print("shift left until box is visible")
        found_box = shift_until_target_detected(
            arx,
            target_prompt=f"{box_prompt}, if you think there are none, output FALSE",
            direction="left",
            speed=0.75,
            max_duration_s=8.0,
            check_interval_s=0.125,
            vote_times=2,
            debug=debug,
        )
        if not found_box:
            print("shift left finished but box was not detected")
            return
        print("box detected")
        
        left_ref, right_ref = dual_arm_hug_pick(
            arx,
            object_prompt="the wood box in front of the robot",
            item_type="hug",
            debug=debug,
            depth_median_n=5,
        )
        print("pick the next box")

        print("start lateral shift to place side")
        arx.step_lift(19.0)
        time.sleep(1.0)

        place_prompt = "a gray coaster on top of wood box"
        found_target = shift_until_target_detected(
            arx,
            target_prompt=f"{place_prompt}, if you think there are none, output FALSE",
            direction="right",
            speed=0.75,
            max_duration_s=6.0,
            check_interval_s=0.125,
            vote_times=2,
            debug=debug,
        )
        if not found_target:
            print("lateral shift finished but target was not detected")
            return
        print("lateral shift completed")

        place_left_ref, place_right_ref = dual_arm_hug_place(
            arx,
            object_prompt=place_prompt,
            item_type="hug",
            debug=debug,
            depth_median_n=5,
        )
        if place_left_ref is None or place_right_ref is None:
            print("hug place canceled or failed")
            return
        print("hug place completed")

    finally:
        try:
            arx.close()
        except RuntimeError as exc:
            # Allow Ctrl+C / ROS shutdown to exit without crashing.
            print(f"close failed: {exc}")


if __name__ == "__main__":
    main()
