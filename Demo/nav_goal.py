from __future__ import annotations

import math
import sys
import time
from typing import List, Optional

import cv2
import numpy as np

sys.path.append("../ARX_Realenv/ROS2")  # noqa

from arx_pointing import predict_multi_points_from_rgb
from arx_ros2_env import ARXRobotEnv
from demo_utils import estimate_lift_from_goal_z
from nav_utils import execute_nav_actions, path_to_actions, recover_rotations
from point2pos_utils import get_aligned_frames, pixel_to_base_point_safe
from visualize_utils import (
    VisualizeContext,
    dispatch_debug_image,
    get_key_nonblock,
    init_keyboard,
    render_nav_goal_debug_view,
    restore_keyboard,
    should_stop,
)


def _build_presence_prompt(goal: str) -> str:
    return (
        f'Is there an object matching this description: "{goal}"? '
        "Output only True or False."
    )


def _vote_goal_presence(
    color,
    goal: str,
    vote_times: int,
) -> bool:
    if vote_times <= 0:
        raise ValueError(f"vote_times must be positive, got {vote_times}")

    prompt = _build_presence_prompt(goal)
    true_count = 0
    false_count = 0

    for _ in range(vote_times):
        _, generated_content = predict_multi_points_from_rgb(
            color,
            text_prompt=goal,
            all_prompt=prompt,
            assume_bgr=False,
            return_raw=True,
        )
        if generated_content == "True":
            true_count += 1
        elif generated_content == "False":
            false_count += 1

    return true_count > false_count


SEARCH_ROI_TOP_Y_RATIO = 1.0 / 3.0
SEARCH_ROI_TOP_WIDTH_RATIO = 1.0 / 2.0


def _build_search_roi_polygon(
    image_shape: tuple[int, ...],
) -> np.ndarray:
    height, width = image_shape[:2]
    top_y = int(round(height * SEARCH_ROI_TOP_Y_RATIO))
    bottom_y = height - 1
    top_width = width * SEARCH_ROI_TOP_WIDTH_RATIO
    top_margin = (width - top_width) / 2.0
    top_left_x = int(round(top_margin))
    top_right_x = int(round(width - 1 - top_margin))
    bottom_left_x = 0
    bottom_right_x = width - 1
    return np.array(
        [
            [top_left_x, top_y],
            [top_right_x, top_y],
            [bottom_right_x, bottom_y],
            [bottom_left_x, bottom_y],
        ],
        dtype=np.int32,
    )


def _apply_roi_focus(
    color: np.ndarray,
    roi_polygon: np.ndarray,
) -> np.ndarray:
    mask = np.zeros(color.shape[:2], dtype=np.uint8)
    cv2.fillConvexPoly(mask, roi_polygon, 255)
    focused = np.zeros_like(color)
    focused[mask == 255] = color[mask == 255]
    return focused


def _point_in_roi(
    point,
    roi_polygon: Optional[np.ndarray],
) -> bool:
    if roi_polygon is None:
        return True
    u, v = float(point[0]), float(point[1])
    return cv2.pointPolygonTest(
        roi_polygon.reshape((-1, 1, 2)),
        (u, v),
        False,
    ) >= 0


def _select_goal_point(
    points,
    depth,
    offset: float = 0.5,
    roi_polygon: Optional[np.ndarray] = None,
):
    valid_goals = []
    for point in points:
        if not _point_in_roi(point, roi_polygon):
            u, v = int(round(point[0])), int(round(point[1]))
            print(f"预测像素 {(u, v)} 不在 search ROI 内，自动刷新")
            continue
        goal_pw = pixel_to_base_point_safe(
            point, depth, robot_part="center", offset=[0.0, offset, 0.0])
        if goal_pw is None:
            u, v = int(round(point[0])), int(round(point[1]))
            print(f"预测像素 {(u, v)} 深度无效或像素越界，自动刷新")
            continue
        valid_goals.append((point, goal_pw))

    if not valid_goals:
        raise ValueError("no valid goal point with usable depth")

    return min(valid_goals, key=lambda item: math.hypot(item[1][0], item[1][1]))


def nav_to_goal(
    arx: ARXRobotEnv,
    goal: str = "white paper balls",
    distance: float = 0.55,
    lift_height: float = 0.0,
    offset: float = 0.5,
    use_goal_z_for_lift: bool = False,
    target_goal_z: float = 0.0,
    rotate_recover: bool = False,
    continuous: bool = False,
    debug_raw: bool = False,
    depth_median_n: int = 5,
    vote_times: int = 3,
    rotate_search_on_miss: bool = False,
    use_initial_search_roi: bool = False,
    visualize: Optional[VisualizeContext] = None,
):
    old_settings = init_keyboard()
    last_result = None
    rotating_search = False
    consecutive_true_count = 0
    consecutive_true_required = 3

    def _log(message: str, *, stage: Optional[str] = None, **payload) -> None:
        del stage, payload
        print(message)

    def _start_rotate_search() -> None:
        nonlocal rotating_search
        if rotating_search:
            return
        arx.step_base(0.0, 0.0, 1.0)
        rotating_search = True

    def _stop_rotate_search() -> None:
        nonlocal rotating_search
        if not rotating_search:
            return
        arx.step_base(0.0, 0.0, 0.0)
        rotating_search = False

    def _check_nav_emergency_stop() -> bool:
        if should_stop(visualize):
            return True
        key = get_key_nonblock()
        return key == "n"

    try:
        arx.step_lift(lift_height)
        while True:
            if should_stop(visualize):
                _stop_rotate_search()
                _log("Stop signal received.", stage="stopped")
                return last_result
            key = get_key_nonblock()
            if key == "n":
                _stop_rotate_search()
                arx.step_base(0.0, 0.0, 0.0)
                _log("Emergency stop received.", stage="stopped")
                return last_result

            frames = arx.get_camera(target_size=(
                640, 480), return_status=False)
            color = frames.get("camera_h_color")
            if color is None:
                if continuous:
                    time.sleep(0.2)
                    continue
                raise RuntimeError("failed to read camera_h color frame")

            search_roi_polygon = _build_search_roi_polygon(color.shape)
            presence_roi_polygon = (
                search_roi_polygon
                if rotating_search or use_initial_search_roi
                else None
            )
            presence_color = (
                _apply_roi_focus(color, presence_roi_polygon)
                if presence_roi_polygon is not None
                else color
            )
            detect_goal = _vote_goal_presence(
                presence_color, goal=goal, vote_times=vote_times)
            if not detect_goal:
                consecutive_true_count = 0
                _log(f"No goal detected for: {goal}", stage="presence_miss")
                if rotate_search_on_miss:
                    _start_rotate_search()
                    time.sleep(0.2)
                    continue
                if continuous:
                    time.sleep(0.2)
                    continue
                return None
            if rotating_search:
                consecutive_true_count += 1
                if consecutive_true_count < consecutive_true_required:
                    _log(
                        f"Goal detected during rotate search: "
                        f"{consecutive_true_count}/{consecutive_true_required}",
                        stage="rotate_confirm",
                    )
                    time.sleep(0.2)
                    continue
            else:
                consecutive_true_count = 0
            goal_point_roi_polygon = presence_roi_polygon
            _stop_rotate_search()
            consecutive_true_count = 0

            color, depth = get_aligned_frames(
                arx, depth_median_n=depth_median_n)
            if color is None or depth is None:
                if continuous:
                    time.sleep(0.2)
                    continue
                raise RuntimeError("failed to read camera_h color/depth frame")

            points, _ = predict_multi_points_from_rgb(
                color,
                text_prompt=goal,
                all_prompt=None,
                assume_bgr=False,
                return_raw=True,
            )
            if points is None or len(points) == 0:
                _log(
                    f"Goal detected but no point found for: {goal}",
                    stage="point_miss",
                )
                if rotate_search_on_miss:
                    _start_rotate_search()
                    time.sleep(0.2)
                    continue
                if continuous:
                    time.sleep(0.2)
                    continue
                return None

            try:
                goal_pixel, goal_pw = _select_goal_point(
                    points,
                    depth,
                    offset=offset,
                    roi_polygon=goal_point_roi_polygon,
                )
            except ValueError as exc:
                _log(str(exc), stage="point_invalid")
                if rotate_search_on_miss:
                    _start_rotate_search()
                    time.sleep(0.2)
                    continue
                if continuous:
                    time.sleep(0.2)
                    continue
                return None

            path = [(0.0, 0.0), (float(goal_pw[0]), float(-goal_pw[1]))]
            actions = path_to_actions(path)

            if debug_raw:
                debug_vis = render_nav_goal_debug_view(
                    color,
                    points,
                    goal_pixel,
                    roi_polygon=goal_point_roi_polygon,
                    nav_prompt=goal,
                )
                debug_result = dispatch_debug_image(
                    visualize,
                    source="nav_to_goal",
                    panel="nav",
                    image=debug_vis,
                    window_name="nav_goal_debug",
                    goal=goal,
                )
                if debug_result is None:
                    _stop_rotate_search()
                    _log("Stop signal received.", stage="stopped")
                    return last_result
                if not debug_result:
                    _log("Refresh requested.", stage="debug_refresh")
                    continue

            lift_height_target = None
            if use_goal_z_for_lift:
                base_status = arx.get_robot_status().get("base")
                current_lift = float(
                    base_status.height) if base_status is not None else float(lift_height)
                lift_height_target = float(estimate_lift_from_goal_z(
                    goal_z=float(goal_pw[2]),
                    current_lift=current_lift,
                    target_goal_z=target_goal_z,
                ))
                _log(
                    f"detected goal_pw[2]={float(goal_pw[2]):.4f}, "
                    f"current_lift={current_lift:.3f}, "
                    f"target_lift={lift_height_target:.3f}",
                    stage="lift_target",
                )
            executed_rotations, interrupted = execute_nav_actions(
                arx,
                actions,
                distance=distance,
                stop_checker=_check_nav_emergency_stop,
                lift_height_target=lift_height_target,
            )
            if interrupted:
                _stop_rotate_search()
                _log("Emergency stop received.", stage="stopped")
                return last_result
            if rotate_recover:
                recover_rotations(arx, executed_rotations)
            last_result = (goal_pw, actions)
            if not continuous:
                return last_result
    finally:
        _stop_rotate_search()
        cv2.destroyAllWindows()
        restore_keyboard(old_settings)


def main():
    arx = ARXRobotEnv(
        duration_per_step=1.0 / 20.0,
        min_steps=20,
        max_v_xyz=0.15,
        max_a_xyz=0.20,
        max_v_rpy=0.5,
        max_a_rpy=1.00,
        camera_type="all",
        camera_view=("camera_h",),
        img_size=(640, 480),
    )
    try:
        arx.reset()
        nav_to_goal(
            arx,
            goal="a yellow glue",
            distance=0.55,
            lift_height=15.0,
            offset=0.25,
            use_goal_z_for_lift=True,
            continuous=False,
            debug_raw=False,
            depth_median_n=5,
            vote_times=5,
        )
    finally:
        arx.close()


if __name__ == "__main__":
    main()
