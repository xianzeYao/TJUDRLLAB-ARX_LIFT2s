from __future__ import annotations

import math
import select
import sys
import termios
import time
from typing import List, Literal, Tuple

import cv2

sys.path.append("../ARX_Realenv/ROS2")  # noqa

from arx_pointing import predict_multi_points_from_rgb
from arx_ros2_env import ARXRobotEnv
from demo_utils import step_base_duration
from nav_utils import path_to_actions
from point2pos_utils import get_aligned_frames, pixel_to_base_point_safe

BASE_FORWARD_SPEED = 0.06
BASE_ROTATE_SPEED = math.pi / 20.6


def _build_presence_prompt(goal: str) -> str:
    return (
        f'Is there an object matching this description: "{goal}"? '
        "Output only True or False."
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


def _select_goal_point(points, depth):
    valid_goals = []
    for point in points:
        goal_pw = pixel_to_base_point_safe(
            point, depth, robot_part="center", offset=[0.0, 0.5, 0.0])
        if goal_pw is None:
            u, v = int(round(point[0])), int(round(point[1]))
            print(f"预测像素 {(u, v)} 深度无效或像素越界，自动刷新")
            continue
        valid_goals.append((point, goal_pw))

    if not valid_goals:
        raise ValueError("no valid goal point with usable depth")

    return min(valid_goals, key=lambda item: math.hypot(item[1][0], item[1][1]))


def _confirm_debug_view(
    color,
    points,
    goal_pixel,
) -> Literal[True, False, None]:
    vis = color.copy()
    for point in points:
        cv2.circle(
            vis,
            center=(int(round(point[0])), int(round(point[1]))),
            radius=5,
            color=(0, 0, 255),
            thickness=-1,
        )

    cv2.circle(
        vis,
        center=(int(round(goal_pixel[0])), int(round(goal_pixel[1]))),
        radius=9,
        color=(0, 255, 0),
        thickness=2,
    )

    win = "nav_goal_debug"
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
            return False
        return True


def _execute_nav_actions(
    arx: ARXRobotEnv,
    actions: List[Tuple[str, float]],
    distance: float,
) -> None:
    for action, value in actions:
        if action == "rotate":
            if value <= 0:
                duration = max(-value - 0.1, 0.1) / BASE_ROTATE_SPEED
                step_base_duration(arx, vx=0.0, vy=0.0, vz=-
                                   0.5, duration=float(duration))
            else:
                duration = value / BASE_ROTATE_SPEED
                step_base_duration(arx, vx=0.0, vy=0.0,
                                   vz=0.5, duration=float(duration))
        elif action == "forward":
            remaining = value - distance
            if remaining <= 0:
                continue
            step_base_duration(
                arx,
                vx=0.5,
                vy=0.0,
                vz=0.0,
                duration=float(remaining / BASE_FORWARD_SPEED),
            )


def nav_to_goal(
    arx: ARXRobotEnv,
    goal: str = "white paper balls",
    distance: float = 0.55,
    continuous: bool = False,
    debug_raw: bool = False,
    depth_median_n: int = 5,
    vote_times: int = 5,
):
    old_settings = _init_keyboard()
    last_result = None

    try:
        arx.step_lift(12.0)
        while True:
            key = _get_key_nonblock()
            if key == "q":
                print("Stop signal received.")
                return last_result

            frames = arx.get_camera(target_size=(
                640, 480), return_status=False)
            color = frames.get("camera_h_color")
            if color is None:
                if continuous:
                    time.sleep(0.2)
                    continue
                raise RuntimeError("failed to read camera_h color frame")

            detect_goal = _vote_goal_presence(
                color, goal=goal, vote_times=vote_times)
            if not detect_goal:
                print(f"No goal detected for: {goal}")
                if continuous:
                    time.sleep(0.2)
                    continue
                return None

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
                print(f"Goal detected but no point found for: {goal}")
                if continuous:
                    time.sleep(0.2)
                    continue
                return None

            try:
                goal_pixel, goal_pw = _select_goal_point(points, depth)
            except ValueError as exc:
                print(exc)
                if continuous:
                    time.sleep(0.2)
                    continue
                return None

            path = [(0.0, 0.0), (float(goal_pw[0]), float(-goal_pw[1]))]
            actions = path_to_actions(path)

            if debug_raw:
                debug_result = _confirm_debug_view(color, points, goal_pixel)
                if debug_result is None:
                    print("Stop signal received.")
                    return last_result
                if not debug_result:
                    print("Refresh requested.")
                    continue

            _execute_nav_actions(arx, actions, distance=distance)
            last_result = (goal_pw, actions)
            if not continuous:
                return last_result
    finally:
        cv2.destroyAllWindows()
        _restore_keyboard(old_settings)


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
            goal="a white crumpled paper",
            distance=0.55,
            continuous=False,
            debug_raw=False,
            depth_median_n=5,
            vote_times=5,
        )
    finally:
        arx.close()


if __name__ == "__main__":
    main()
