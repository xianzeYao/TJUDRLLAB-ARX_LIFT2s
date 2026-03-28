from __future__ import annotations

from typing import Mapping, Optional, Tuple, Literal

import cv2
import numpy as np

from arx_pointing import predict_multi_points_from_rgb, predict_point_from_rgb
from demo_utils import (
    draw_text_lines,
    execute_pick_place_cup_sequence,
    execute_pick_place_deepbox_sequence,
    execute_pick_place_normal_object_sequence,
    execute_pick_place_straw_sequence,
)
from point2pos_utils import (
    get_aligned_frames,
    pixel_to_ref_point_safe,
)
from task_completion_detector import run_task_completion_check
import time
import sys

sys.path.append("../ARX_Realenv/ROS2")  # noqa
from arx_ros2_env import ARXRobotEnv  # noqa


def _set_gripper_at_current_eef(
    arx: ARXRobotEnv,
    arm: str,
    gripper: float,
    delay_s: float = 5.0,
) -> None:
    status = arx.get_robot_status().get(arm)
    if status is None or not hasattr(status, "end_pos"):
        print(f"{arm} arm status unavailable, skip gripper action")
        return

    eef = np.asarray(status.end_pos, dtype=np.float32)
    if eef.size < 6:
        print(f"{arm} arm end_pos invalid, skip gripper action")
        return

    time.sleep(delay_s)
    action = {
        arm: np.array(
            [eef[0], eef[1], eef[2], eef[3], eef[4], eef[5], gripper],
            dtype=np.float32,
        )
    }
    arx.step_smooth_eef(action)


def _release_gripper_at_current_eef(
    arx: ARXRobotEnv,
    arm: str,
    delay_s: float = 5.0,
) -> None:
    _set_gripper_at_current_eef(
        arx=arx,
        arm=arm,
        gripper=-3.4,
        delay_s=delay_s,
    )


def _open_gripper_at_current_eef(
    arx: ARXRobotEnv,
    arm: str,
) -> None:
    _set_gripper_at_current_eef(
        arx=arx,
        arm=arm,
        gripper=-3.4,
        delay_s=0.0,
    )


def _close_gripper_at_current_eef(
    arx: ARXRobotEnv,
    arm: str,
) -> None:
    _set_gripper_at_current_eef(
        arx=arx,
        arm=arm,
        gripper=0.0,
        delay_s=0.0,
    )


def _predict_one_point(color: np.ndarray, base_prompt: str) -> Tuple[int, int]:
    u, v = predict_point_from_rgb(
        color,
        text_prompt=base_prompt,
        assume_bgr=False,
        temperature=0.0,
    )
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


def _select_arm_from_pixel(
    pixel: Tuple[int, int],
    image_width: int,
) -> Literal["left", "right"]:
    return "left" if pixel[0] < (image_width / 2.0) else "right"


def _single_arm_pick_place_once(
    arx: ARXRobotEnv,
    pick_prompt: str,
    place_prompt: str,
    arm_side: Literal["left", "right", "fit"] = "left",
    debug: bool = True,
    depth_median_n: int = 10,
) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], Optional[Literal["left", "right"]]]:
    while True:
        do_pick = bool(pick_prompt)
        do_place = bool(place_prompt)
        if not do_pick and not do_place:
            raise ValueError("pick_prompt 和 place_prompt 不能同时为空")
        time.sleep(1.5)
        color, depth = get_aligned_frames(
            arx, depth_median_n=depth_median_n)
        if color is None or depth is None:
            continue
        pick_px = None
        place_px = None
        try:
            if do_pick and do_place:
                pick_px, place_px = _predict_two_points(
                    color, pick_prompt, place_prompt
                )
            elif do_pick:
                pick_px = _predict_one_point(color, pick_prompt)
            else:
                place_px = _predict_one_point(color, place_prompt)
        except RuntimeError as exc:
            print(f"点位预测失败，自动刷新：{exc}")
            continue

        if arm_side == "fit":
            ref_pixel = pick_px if pick_px is not None else place_px
            if ref_pixel is None:
                raise RuntimeError(
                    "fit arm_side requires at least one predicted point")
            arm = _select_arm_from_pixel(ref_pixel, color.shape[1])
        else:
            arm = arm_side

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
        lines = [f"Arm: {arm}"]
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
                return None, None, None
        else:
            cv2.destroyAllWindows()
        return pick_ref, place_ref, arm


def _execute_item_sequence(
    arx: ARXRobotEnv,
    item_type: Literal["cup", "straw", "deepbox", "normal object"],
    pick_ref: Optional[np.ndarray],
    place_ref: Optional[np.ndarray],
    arm: Literal["left", "right"],
    do_pick: bool,
    do_place: bool,
) -> None:
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
    elif item_type == "deepbox":
        execute_pick_place_deepbox_sequence(
            arx=arx,
            pick_ref=pick_ref,
            place_ref=place_ref,
            arm=arm,
            do_pick=do_pick,
            do_place=do_place,
        )
    elif item_type == "normal object":
        execute_pick_place_normal_object_sequence(
            arx=arx,
            pick_ref=pick_ref,
            place_ref=place_ref,
            arm=arm,
            do_pick=do_pick,
            do_place=do_place,
        )
    else:
        raise ValueError(f"unknown item_type: {item_type!r}")


def single_arm_pick_place(
    arx: ARXRobotEnv,
    pick_prompt: str,
    place_prompt: str,
    arm_side: Literal["left", "right", "fit"] = "fit",
    item_type: Literal["cup", "straw", "deepbox",
                       "normal object"] = "normal object",
    debug: bool = True,
    depth_median_n: int = 10,
    release_after_pick: bool = False,
    verify_completion: bool = False,
    completion_retry_attempts: int = 1,
    completion_check_mode: Literal["single_image",
                                   "multi_image"] = "single_image",
    completion_third_camera_view: str = "camera_h",
    completion_hand_camera_by_arm: Optional[Mapping[str, str]] = None,
    completion_settle_s: float = 1.0,
    completion_capture_retries: int = 1,
    completion_capture_retry_sleep_s: float = 0.2,
) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], Optional[Literal["left", "right"]]]:
    try:
        do_pick = bool(pick_prompt)
        do_place = bool(place_prompt)
        last_result: Tuple[
            Optional[np.ndarray],
            Optional[np.ndarray],
            Optional[Literal["left", "right"]],
        ] = (None, None, None)
        previous_arm: Optional[Literal["left", "right"]] = None
        retry_limit: Optional[int] = None
        if verify_completion:
            normalized_retry_attempts = max(0, int(completion_retry_attempts))
            retry_limit = None if normalized_retry_attempts == 0 else normalized_retry_attempts
        retry_idx = 0

        while True:
            last_result = _single_arm_pick_place_once(
                arx=arx,
                pick_prompt=pick_prompt,
                place_prompt=place_prompt,
                arm_side=arm_side,
                debug=debug,
                depth_median_n=depth_median_n,
            )
            _, _, arm = last_result
            if arm is None:
                return last_result
            if not verify_completion or not do_pick:
                _execute_item_sequence(
                    arx=arx,
                    item_type=item_type,
                    pick_ref=last_result[0],
                    place_ref=last_result[1],
                    arm=arm,
                    do_pick=do_pick,
                    do_place=do_place,
                )
                if do_pick and (not do_place) and release_after_pick:
                    _release_gripper_at_current_eef(
                        arx=arx,
                        arm=arm,
                    )
                return last_result
            if previous_arm is not None and previous_arm != arm:
                success, error_message = arx.set_special_mode(
                    1, side=previous_arm)
                if not success and error_message:
                    print(
                        f"set_special_mode(1, side={previous_arm!r}) returned: {error_message}"
                    )
            previous_arm = arm

            try:
                if do_pick:
                    _execute_item_sequence(
                        arx=arx,
                        item_type=item_type,
                        pick_ref=last_result[0],
                        place_ref=last_result[1],
                        arm=arm,
                        do_pick=True,
                        do_place=False,
                    )
                check_start = time.time()
                check_result = run_task_completion_check(
                    arx=arx,
                    arm=arm,
                    pick_prompt=pick_prompt,
                    item_type=item_type,
                    mode=completion_check_mode,
                    third_camera_view=completion_third_camera_view,
                    hand_camera_by_arm=completion_hand_camera_by_arm,
                    settle_s=completion_settle_s,
                    max_retries=completion_capture_retries,
                    retry_sleep_s=completion_capture_retry_sleep_s,
                )
            except Exception as exc:
                print(f"任务检测失败，跳过自动重试: {exc}")
                return last_result
            check_elapsed_s = time.time() - check_start

            print(
                f"任务检测结果: {check_result.status}, "
                f"desc:{check_result.description}, "
                f"elapsed_s={check_elapsed_s:.3f}"
            )
            if check_result.third_description is not None:
                print(f"第三视角描述: {check_result.third_description}")
            if check_result.wrist_description is not None:
                print(f"腕部视角描述: {check_result.wrist_description}")
            if check_result.status == "success":
                if do_place:
                    _execute_item_sequence(
                        arx=arx,
                        item_type=item_type,
                        pick_ref=last_result[0],
                        place_ref=last_result[1],
                        arm=arm,
                        do_pick=False,
                        do_place=True,
                    )
                elif release_after_pick:
                    _release_gripper_at_current_eef(
                        arx=arx,
                        arm=arm,
                    )
                return last_result
            if (
                check_result.third_status == "success"
                and check_result.wrist_status != "success"
            ):
                print("失去物体视野，退出自动重试")
                return last_result
            if retry_limit is not None and retry_idx >= retry_limit:
                return last_result
            _open_gripper_at_current_eef(
                arx=arx,
                arm=arm,
            )
            time.sleep(1.0)
            _close_gripper_at_current_eef(
                arx=arx,
                arm=arm,
            )
            retry_idx += 1
            if retry_limit is None:
                print(f"任务未完成，开始 retry {retry_idx}/inf")
            else:
                print(f"任务未完成，开始 retry {retry_idx}/{retry_limit}")
        return last_result
    finally:
        cv2.destroyAllWindows()


def main():
    arx = ARXRobotEnv(
        duration_per_step=1.0 / 20.0,
        min_steps=20,
        max_v_xyz=0.15, max_a_xyz=0.20,
        max_v_rpy=0.45, max_a_rpy=1.00,
        camera_type="all",
        camera_view=("camera_l", "camera_h", "camera_r"),
        img_size=(640, 480),
    )
    arx.reset()
    # arx.step_lift(18.0)
    # right_open_action = {"right": np.array(
    #     [0, 0, 0, 0, 0, 0, -3.4], dtype=np.float32)}
    # arx.step_smooth_eef(right_open_action)
    # time.sleep(5.0)
    # right_close_action = {"right": np.array(
    #     [0, 0, 0, 0, 0, 0, -2.05], dtype=np.float32)}
    # arx.step_smooth_eef(right_close_action)
    # place_prompt = "the center part of the brown coaster on the right side"
    # single_arm_pick_place(arx, pick_prompt="", place_prompt=place_prompt, arm_side="right",
    #                       debug=True, depth_median_n=10)
    pick_prompt = "a tennis ball"
    try:
        single_arm_pick_place(arx, pick_prompt=pick_prompt, place_prompt="", arm_side="fit",
                              debug=True, depth_median_n=10,
                              item_type="normal object",
                              verify_completion=True,
                              completion_retry_attempts=2,)
    finally:
        arx.close()


if __name__ == "__main__":
    main()
