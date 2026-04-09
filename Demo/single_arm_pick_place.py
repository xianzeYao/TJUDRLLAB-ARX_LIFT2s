from __future__ import annotations

from typing import Mapping, Optional, Tuple, Literal

import cv2
import numpy as np

from arx_pointing import predict_multi_points_from_rgb, predict_point_from_rgb
from demo_utils import (
    execute_pick_place_cup_sequence,
    execute_pick_place_deepbox_sequence,
    execute_pick_place_normal_object_sequence,
    execute_pick_place_straw_sequence,
)
from point2pos_utils import (
    get_aligned_frames,
    pixel_to_ref_point_safe,
)
from task_completion_detector import (
    prepare_task_completion_check,
    run_task_completion_check,
)
from visualize_utils import (
    VisualizeContext,
    dispatch_debug_image,
    render_pick_place_debug_view,
    should_stop,
)
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


def _predict_pick_one_point(
    color: np.ndarray,
    base_prompt: str,
) -> Tuple[int, int]:
    u, v = predict_point_from_rgb(
        color,
        text_prompt=base_prompt,
        assume_bgr=False,
        temperature=0.0,
    )
    return int(round(u)), int(round(v))


def _predict_place_one_point(
    color: np.ndarray,
    base_prompt: str,
    all_prompt: str = "",
) -> Tuple[int, int]:
    if all_prompt:
        u, v = predict_point_from_rgb(
            color,
            text_prompt="",
            all_prompt=all_prompt,
            assume_bgr=False,
            temperature=0.0,
        )
    else:
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
    place_all_prompt: str = "",
    arm_side: Literal["left", "right", "fit"] = "left",
    debug: bool = True,
    depth_median_n: int = 10,
    visualize: Optional[VisualizeContext] = None,
) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], Optional[Literal["left", "right"]]]:
    while True:
        if should_stop(visualize):
            return None, None, None
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
                pick_px = _predict_pick_one_point(color, pick_prompt)
            else:
                place_px = _predict_place_one_point(
                    color,
                    place_prompt,
                    all_prompt=place_all_prompt,
                )
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

        vis = render_pick_place_debug_view(
            color,
            arm=arm,
            pick_prompt=pick_prompt,
            place_prompt=place_prompt,
            pick_px=pick_px,
            place_px=place_px,
        )

        if debug:
            debug_result = dispatch_debug_image(
                visualize,
                source="single_arm_pick_place",
                panel="manip",
                image=vis,
                window_name="single_arm_pick_place",
                arm=arm,
                pick_prompt=pick_prompt,
                place_prompt=place_prompt,
            )
            if not debug_result:
                continue
            if debug_result is None:
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


def _finish_pick_place_success(
    arx: ARXRobotEnv,
    *,
    last_result: Tuple[
        Optional[np.ndarray],
        Optional[np.ndarray],
        Optional[Literal["left", "right"]],
    ],
    item_type: Literal["cup", "straw", "deepbox", "normal object"],
    arm: Literal["left", "right"],
    do_place: bool,
    release_after_pick: bool,
    pick_prompt: str,
    place_prompt: str,
    visualize: Optional[VisualizeContext],
) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], Optional[Literal["left", "right"]]]:
    del visualize, pick_prompt, place_prompt
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


def single_arm_pick_place(
    arx: ARXRobotEnv,
    pick_prompt: str,
    place_prompt: str,
    place_all_prompt: str = "",
    arm_side: Literal["left", "right", "fit"] = "fit",
    item_type: Literal["cup", "straw", "deepbox",
                       "normal object"] = "normal object",
    debug: bool = True,
    depth_median_n: int = 10,
    release_after_pick: bool = False,
    verify_completion: bool = False,
    completion_retry_attempts: int = 1,
    completion_check_mode: Literal["plus", "single_image",
                                   "multi_image"] = "plus",
    completion_third_camera_view: str = "camera_h",
    completion_hand_camera_by_arm: Optional[Mapping[str, str]] = None,
    completion_settle_s: float = 1.0,
    completion_capture_retries: int = 1,
    completion_capture_retry_sleep_s: float = 0.2,
    visualize: Optional[VisualizeContext] = None,
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
            if should_stop(visualize):
                return last_result
            last_result = _single_arm_pick_place_once(
                arx=arx,
                pick_prompt=pick_prompt,
                place_prompt=place_prompt,
                place_all_prompt=place_all_prompt,
                arm_side=arm_side,
                debug=debug,
                depth_median_n=depth_median_n,
                visualize=visualize,
            )
            _, _, arm = last_result
            if arm is None:
                return last_result
            should_verify_pick = verify_completion and do_pick
            if should_verify_pick and previous_arm is not None and previous_arm != arm:
                success, error_message = arx.set_special_mode(
                    1, side=previous_arm)
                if not success and error_message:
                    print(
                        f"set_special_mode(1, side={previous_arm!r}) returned: {error_message}"
                    )
            if should_verify_pick:
                previous_arm = arm
            check_context = None
            if should_verify_pick:
                try:
                    check_context = prepare_task_completion_check(
                        arx=arx,
                        arm=arm,
                        mode=completion_check_mode,
                        third_camera_view=completion_third_camera_view,
                        hand_camera_by_arm=completion_hand_camera_by_arm,
                        pre_pick_settle_s=0.0,
                        max_retries=completion_capture_retries,
                        retry_sleep_s=completion_capture_retry_sleep_s,
                    )
                except Exception as exc:
                    print(f"pick 前任务检测准备失败，跳过自动重试: {exc}")
                    return last_result

            if do_pick:
                if should_stop(visualize):
                    return last_result
                if should_verify_pick:
                    try:
                        _execute_item_sequence(
                            arx=arx,
                            item_type=item_type,
                            pick_ref=last_result[0],
                            place_ref=last_result[1],
                            arm=arm,
                            do_pick=True,
                            do_place=False,
                        )
                    except Exception as exc:
                        print(f"pick 执行失败，跳过自动重试: {exc}")
                        return last_result
                else:
                    _execute_item_sequence(
                        arx=arx,
                        item_type=item_type,
                        pick_ref=last_result[0],
                        place_ref=last_result[1],
                        arm=arm,
                        do_pick=True,
                        do_place=False,
                    )

            if not should_verify_pick:
                return _finish_pick_place_success(
                    arx=arx,
                    last_result=last_result,
                    item_type=item_type,
                    arm=arm,
                    do_place=do_place,
                    release_after_pick=release_after_pick,
                    pick_prompt=pick_prompt,
                    place_prompt=place_prompt,
                    visualize=visualize,
                )

            try:
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
                    check_context=check_context,
                )
            except Exception as exc:
                print(f"任务检测失败，跳过自动重试: {exc}")
                return last_result
            check_elapsed_s = time.time() - check_start

            if check_result.status == "success":
                return _finish_pick_place_success(
                    arx=arx,
                    last_result=last_result,
                    item_type=item_type,
                    arm=arm,
                    do_place=do_place,
                    release_after_pick=release_after_pick,
                    pick_prompt=pick_prompt,
                    place_prompt=place_prompt,
                    visualize=visualize,
                )

            if check_result.next_action == "exit":
                return last_result[0], last_result[1], None

            if retry_limit is not None and retry_idx >= retry_limit:
                print(f"[pick_place][{completion_check_mode}] retry limit reached, stop retry")
                return last_result

            if check_result.next_action == "open_close_retry":
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
    try:
        pick_prompt = "a tennis ball"
        place_prompt = "the center part of the third floor on the shelf"
        arx.step_lift(17.0)
        single_arm_pick_place(arx, pick_prompt=pick_prompt, place_prompt=place_prompt, arm_side="right",
                              item_type="normal object",
                              debug=True, depth_median_n=10)
        arx.step_lift(14.0)
        single_arm_pick_place(arx, pick_prompt=pick_prompt, place_prompt="", arm_side="right",
                              debug=True, depth_median_n=10,

                              verify_completion=False,
                              completion_retry_attempts=2,)
    finally:
        arx.close()


if __name__ == "__main__":
    main()
