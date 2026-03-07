from __future__ import annotations

from typing import Optional, Tuple, Literal

import cv2
import numpy as np

from arx_pointing import predict_multi_points_from_rgb
from demo_utils import (
    draw_text_lines,
    execute_pick_place_cup_sequence,
    execute_pick_place_straw_sequence,
)
from point2pos_utils import (
    get_aligned_frames,
    pixel_to_ref_point_safe,
)
import time
import sys

sys.path.append("../ARX_Realenv/ROS2")  # noqa
from arx_ros2_env import ARXRobotEnv  # noqa


def _release_gripper_at_current_eef(
    arx: ARXRobotEnv,
    arm: str,
    delay_s: float = 5.0,
) -> None:
    status = arx.get_robot_status().get(arm)
    if status is None or not hasattr(status, "end_pos"):
        print(f"{arm} arm status unavailable, skip delayed gripper open")
        return

    eef = np.asarray(status.end_pos, dtype=np.float32)
    if eef.size < 6:
        print(f"{arm} arm end_pos invalid, skip delayed gripper open")
        return

    time.sleep(delay_s)
    open_gripper = -3.4
    action = {
        arm: np.array(
            [eef[0], eef[1], eef[2], eef[3], eef[4], eef[5], open_gripper],
            dtype=np.float32,
        )
    }
    arx.step(action)


def _predict_one_point(color: np.ndarray, base_prompt: str) -> Tuple[int, int]:
    full_prompt = (
        "Provide exactly one point coordinate of objects region this sentence describes: "
        f"{base_prompt} "
        'The answer should be presented in JSON format as follows: [{"point_2d": [x, y]}]. '
        "Return only JSON."
    )
    points = predict_multi_points_from_rgb(
        color,
        text_prompt="",
        all_prompt=full_prompt,
        assume_bgr=False,
        temperature=0.0,
    )
    if not points:
        raise RuntimeError("未解析到坐标")
    u, v = points[0]
    return int(round(u)), int(round(v))


def _predict_two_points(
    color: np.ndarray, pick_prompt: str, place_prompt: str
) -> Tuple[Tuple[int, int], Tuple[int, int]]:
    full_prompt = (
        "Provide exactly two points coordinate of objects region this sentence describes: "
        f"{pick_prompt} and {place_prompt}. "
        'The answer should be presented in JSON format as follows: [{"point_2d": [x, y]}]. '
        "Return only JSON. First point is pick, second point is place."
    )
    points = predict_multi_points_from_rgb(
        color,
        text_prompt="",
        all_prompt=full_prompt,
        assume_bgr=False,
        temperature=0.0,
    )
    if len(points) < 2:
        raise RuntimeError("未解析到足够坐标")
    pick = (int(round(points[0][0])), int(round(points[0][1])))
    place = (int(round(points[1][0])), int(round(points[1][1])))
    return pick, place


def single_arm_pick_place(
    arx: ARXRobotEnv,
    pick_prompt: str,
    place_prompt: str,
    arm: str = "left",
    item_type: Literal["cup", "straw"] = "cup",
    reset_robot: bool = True,
    close_robot: bool = True,
    debug: bool = True,
    depth_median_n: int = 10,
    release_after_pick: bool = False,
) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
    try:
        if reset_robot:
            arx.reset()

        while True:
            do_pick = bool(pick_prompt)
            do_place = bool(place_prompt)
            if not do_pick and not do_place:
                raise ValueError("pick_prompt 和 place_prompt 不能同时为空")
            time.sleep(1.5)
            color, depth = get_aligned_frames(arx, depth_median_n=depth_median_n)
            if color is None or depth is None:
                continue
            pick_px = None
            place_px = None
            if do_pick and do_place:
                pick_px, place_px = _predict_two_points(
                    color, pick_prompt, place_prompt
                )
            elif do_pick:
                pick_px = _predict_one_point(color, pick_prompt)
            else:
                place_px = _predict_one_point(color, place_prompt)

            pick_ref = None
            place_ref = None
            if pick_px is not None:
                pick_ref = pixel_to_ref_point_safe(
                    pick_px,
                    depth,
                    robot_part=arm,
                )
                if pick_ref is None:
                    print(f"预测像素 {pick_px} 深度无效或像素越界，自动刷新")
                    continue
            if place_px is not None:
                place_ref = pixel_to_ref_point_safe(
                    place_px,
                    depth,
                    robot_part=arm,
                )
                if place_ref is None:
                    print(f"预测像素 {place_px} 深度无效或像素越界，自动刷新")
                    continue

            vis = color.copy()
            lines = []
            if pick_px is not None:
                cv2.circle(vis, pick_px, 3,  (0, 0, 255), -1)
                lines.append(f"Pick: {pick_prompt}")
            if place_px is not None:
                cv2.circle(vis, place_px, 3,  (255, 0, 0), -1)
                lines.append(f"Place: {place_prompt}")
            if lines:
                draw_text_lines(
                    vis,
                    lines,
                    origin=(10, 25),
                    line_height=22,
                    color=(0, 0, 255),
                    scale=0.6,
                    thickness=2,
                )

            if debug:
                win = "single_arm_pick_place"
                cv2.namedWindow(win, cv2.WINDOW_NORMAL)
                cv2.imshow(win, vis)
                key = cv2.waitKey(0)
                if key == ord("r"):
                    continue
                if key == ord("q"):
                    return None, None
                # 默认确认
            else:
                cv2.destroyAllWindows()

            if item_type == "cup":
                execute_pick_place_cup_sequence(
                    arx=arx,
                    pick_ref=pick_ref,
                    place_ref=place_ref,
                    arm=arm,
                    do_pick=do_pick,
                    do_place=do_place,
                )
            elif item_type == "straw":
                execute_pick_place_straw_sequence(
                    arx=arx,
                    pick_ref=pick_ref,
                    place_ref=place_ref,
                    arm=arm,
                    do_pick=do_pick,
                    do_place=do_place,
                )
            else:
                raise ValueError(f"unknown item_type: {item_type!r}")
            if do_pick and release_after_pick:
                _release_gripper_at_current_eef(
                    arx=arx,
                    arm=arm,
                )
            return pick_ref, place_ref
    finally:
        cv2.destroyAllWindows()
        if close_robot:
            arx.close()


def main():
    arx = ARXRobotEnv(
        duration_per_step=1.0 / 20.0,
        min_steps=20,
        max_v_xyz=0.15, max_a_xyz=0.20,
        max_v_rpy=0.45, max_a_rpy=1.00,
        camera_type="all",
        camera_view=("camera_h",),
        img_size=(640, 480),
    )
    arx.reset()
    arx.step_lift(18.0)
    # right_open_action = {"right": np.array(
    #     [0, 0, 0, 0, 0, 0, -3.4], dtype=np.float32)}
    # arx.step(right_open_action)
    # time.sleep(5.0)
    # right_close_action = {"right": np.array(
    #     [0, 0, 0, 0, 0, 0, -2.05], dtype=np.float32)}
    # arx.step(right_close_action)
    # place_prompt = "the center part of the brown coaster on the right side"
    # single_arm_pick_place(arx, reset_robot=False, pick_prompt="", place_prompt=place_prompt, arm="right",
    #                       debug=True, depth_median_n=10)
    pick_prompt = "the cup on the left brown coaster"
    single_arm_pick_place(arx, reset_robot=False, pick_prompt=pick_prompt, place_prompt="",
                          debug=True, depth_median_n=15)


if __name__ == "__main__":
    main()
