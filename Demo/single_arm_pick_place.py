from __future__ import annotations

from typing import Mapping, Optional, Tuple, Literal

import cv2
import numpy as np

from arx_pointing import predict_multi_points_from_rgb, predict_point_from_rgb
from demo_utils import (
    execute_pick_place_cup_sequence,
    execute_pick_place_deepbox_sequence,
    execute_pick_place_normal_object_sequence,
    execute_return_to_source_sequence,
    execute_pick_place_straw_sequence,
    get_pick_close_target,
)
from point2pos_utils import (
    get_aligned_frames,
    pixel_to_ref_point_safe,
)
from task_completion_detector import (
    capture_hand_check_frame,
    capture_third_check_frame,
    predict_third_person_target_check,
    predict_wrist_target_check,
    run_task_completion_check,
)
from visualize_utils import (
    VisualizeContext,
    dispatch_debug_image,
    emit_event,
    emit_log,
    emit_result,
    emit_stage,
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


def _read_current_gripper(arx: ARXRobotEnv, arm: str) -> Optional[float]:
    status = arx.get_robot_status().get(arm)
    if status is None or not hasattr(status, "joint_pos"):
        print(f"{arm} arm joint status unavailable, skip gripper read")
        return None

    joint_pos = np.asarray(status.joint_pos, dtype=np.float32).reshape(-1)
    if joint_pos.size < 7:
        print(f"{arm} arm joint_pos invalid, skip gripper read")
        return None
    return float(joint_pos[6])


def _print_decision(title: str, parts: list[tuple[str, object]]) -> None:
    chunks = []
    for key, value in parts:
        if value is None:
            continue
        chunks.append(f"{key}={value}")
    print(f"{title} " + " | ".join(chunks))


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
) -> Tuple[int, int]:
    all_prompt = (
        f"Provide ONE 2D point for: {base_prompt}\n\n"
        "        Rules:\n"
        '            Output JSON only: [{"point_2d":[x,y]}]\n'
        "            x,y must be in [0,1000]\n"
        "            The point MUST NOT be on any other object already inside or placed on the "
        f" {base_prompt}\n"
        "        Return JSON only."
    )
    u, v = predict_point_from_rgb(
        color,
        text_prompt="",
        all_prompt=all_prompt,
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
                place_px = _predict_place_one_point(color, place_prompt)
        except RuntimeError as exc:
            print(f"点位预测失败，自动刷新：{exc}")
            emit_log(
                visualize,
                source="single_arm_pick_place",
                stage="predict",
                message=f"点位预测失败，自动刷新：{exc}",
            )
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
                emit_log(
                    visualize,
                    source="single_arm_pick_place",
                    stage="predict",
                    message=f"预测像素 {pick_px} 深度无效或像素越界，自动刷新",
                )
                continue
        if place_px is not None:
            place_ref = pixel_to_ref_point_safe(
                place_px,
                depth,
                robot_part=arm,
            )
            if place_ref is None:
                print(f"预测像素 {place_px} 深度无效或像素越界，自动刷新")
                emit_log(
                    visualize,
                    source="single_arm_pick_place",
                    stage="predict",
                    message=f"预测像素 {place_px} 深度无效或像素越界，自动刷新",
                )
                continue

        vis = render_pick_place_debug_view(
            color,
            arm=arm,
            pick_prompt=pick_prompt,
            place_prompt=place_prompt,
            pick_px=pick_px,
            place_px=place_px,
        )

        emit_event(
            visualize,
            "pick_place_plan",
            source="single_arm_pick_place",
            arm=arm,
            pick_prompt=pick_prompt,
            place_prompt=place_prompt,
            pick_pixel=pick_px,
            place_pixel=place_px,
            pick_ref=(
                None if pick_ref is None
                else np.asarray(pick_ref, dtype=np.float32).tolist()
            ),
            place_ref=(
                None if place_ref is None
                else np.asarray(place_ref, dtype=np.float32).tolist()
            ),
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
    if do_place:
        emit_stage(
            visualize,
            source="single_arm_pick_place",
            stage="place",
            message=f"Execute place with {arm} arm",
            arm=arm,
        )
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
    emit_result(
        visualize,
        source="single_arm_pick_place",
        status="success",
        message="single arm pick/place completed",
        arm=arm,
        pick_prompt=pick_prompt,
        place_prompt=place_prompt,
    )
    return last_result


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
        emit_stage(
            visualize,
            source="single_arm_pick_place",
            stage="start",
            message="Start single arm pick/place",
            pick_prompt=pick_prompt,
            place_prompt=place_prompt,
        )
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
                emit_result(
                    visualize,
                    source="single_arm_pick_place",
                    status="stopped",
                    message="single arm pick/place stopped",
                    pick_prompt=pick_prompt,
                    place_prompt=place_prompt,
                )
                return last_result
            last_result = _single_arm_pick_place_once(
                arx=arx,
                pick_prompt=pick_prompt,
                place_prompt=place_prompt,
                arm_side=arm_side,
                debug=debug,
                depth_median_n=depth_median_n,
                visualize=visualize,
            )
            _, _, arm = last_result
            if arm is None:
                emit_result(
                    visualize,
                    source="single_arm_pick_place",
                    status="canceled",
                    message="single arm pick/place canceled",
                    pick_prompt=pick_prompt,
                    place_prompt=place_prompt,
                )
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
                emit_result(
                    visualize,
                    source="single_arm_pick_place",
                    status="success",
                    message="single arm pick/place completed",
                    arm=arm,
                    pick_prompt=pick_prompt,
                    place_prompt=place_prompt,
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
                    emit_stage(
                        visualize,
                        source="single_arm_pick_place",
                        stage="pick",
                        message=f"Execute pick with {arm} arm",
                        arm=arm,
                    )
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
                emit_log(
                    visualize,
                    source="single_arm_pick_place",
                    stage="pick",
                    message=f"pick 执行失败，跳过自动重试: {exc}",
                )
                return last_result

            if completion_check_mode == "plus":
                close_target = get_pick_close_target(item_type)
                actual_gripper = _read_current_gripper(arx, arm)
                emit_event(
                    visualize,
                    "completion_check",
                    source="single_arm_pick_place",
                    mode="plus",
                    arm=arm,
                    close_target=close_target,
                    actual_gripper=actual_gripper,
                )
                if actual_gripper is not None and actual_gripper >= close_target:
                    _print_decision(
                        "[pick_place][plus][夹爪]",
                        [
                            ("actual", actual_gripper),
                            ("target", f"{close_target:.3f}"),
                            ("result", "empty"),
                            ("next", "retry"),
                        ],
                    )
                    emit_log(
                        visualize,
                        source="single_arm_pick_place",
                        stage="completion_check",
                        message=(
                            "夹爪接近闭合位，视为未夹到物体，"
                            f"actual_gripper={actual_gripper:.3f}, "
                            f"close_target={close_target:.3f}"
                        ),
                    )
                    if retry_limit is not None and retry_idx >= retry_limit:
                        print(
                            "[pick_place][plus] retry limit reached after empty-gripper branch"
                        )
                        return last_result
                    retry_idx += 1
                    if retry_limit is None:
                        print(f"任务未完成，开始 retry {retry_idx}/inf")
                    else:
                        print(f"任务未完成，开始 retry {retry_idx}/{retry_limit}")
                    continue
                _print_decision(
                    "[pick_place][plus][夹爪]",
                    [
                        ("actual", actual_gripper),
                        ("target", f"{close_target:.3f}"),
                        ("result", "has_object"),
                        ("next", "check_wrist"),
                    ],
                )
                try:
                    check_start = time.time()
                    hand_image, hand_camera_key = capture_hand_check_frame(
                        arx=arx,
                        arm=arm,
                        hand_camera_by_arm=completion_hand_camera_by_arm,
                        settle_s=completion_settle_s,
                        max_retries=completion_capture_retries,
                        retry_sleep_s=completion_capture_retry_sleep_s,
                    )
                    wrist_result = predict_wrist_target_check(
                        hand_image=hand_image,
                        pick_prompt=pick_prompt,
                    )
                except Exception as exc:
                    print(f"腕部检测失败，跳过自动重试: {exc}")
                    emit_log(
                        visualize,
                        source="single_arm_pick_place",
                        stage="completion_check",
                        message=f"腕部检测失败，跳过自动重试: {exc}",
                    )
                    return last_result
                check_elapsed_s = time.time() - check_start
                _print_decision(
                    "[pick_place][plus][腕部]",
                    [
                        ("result", wrist_result.status),
                        ("desc", wrist_result.description),
                        ("elapsed_s", f"{check_elapsed_s:.3f}"),
                    ],
                )
                emit_event(
                    visualize,
                    "completion_check",
                    source="single_arm_pick_place",
                    mode="plus",
                    status=wrist_result.status,
                    description=wrist_result.description,
                    wrist_description=wrist_result.description,
                    wrist_status=wrist_result.status,
                    elapsed_s=check_elapsed_s,
                    arm=arm,
                    actual_gripper=actual_gripper,
                    close_target=close_target,
                    hand_camera_key=hand_camera_key,
                )
                if wrist_result.status == "success":
                    _print_decision(
                        "[pick_place][plus][最终]",
                        [
                            ("wrist", "success"),
                            ("final", "success"),
                            ("next", "continue"),
                        ],
                    )
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
                    third_start = time.time()
                    third_image, third_camera_key = capture_third_check_frame(
                        arx=arx,
                        third_camera_view=completion_third_camera_view,
                        settle_s=completion_settle_s,
                        max_retries=completion_capture_retries,
                        retry_sleep_s=completion_capture_retry_sleep_s,
                    )
                    third_result = predict_third_person_target_check(
                        third_image=third_image,
                        pick_prompt=pick_prompt,
                    )
                except Exception as exc:
                    print(f"第三视角检测失败，跳过自动重试: {exc}")
                    emit_log(
                        visualize,
                        source="single_arm_pick_place",
                        stage="completion_check",
                        message=f"第三视角检测失败，跳过自动重试: {exc}",
                    )
                    return last_result
                third_elapsed_s = time.time() - third_start
                _print_decision(
                    "[pick_place][plus][第三视角]",
                    [
                        ("result", third_result.status),
                        ("desc", third_result.description),
                        ("elapsed_s", f"{third_elapsed_s:.3f}"),
                    ],
                )
                emit_event(
                    visualize,
                    "completion_check",
                    source="single_arm_pick_place",
                    mode="plus_third_after_wrist_fail",
                    status=third_result.status,
                    description=third_result.description,
                    wrist_description=wrist_result.description,
                    wrist_status=wrist_result.status,
                    third_description=third_result.description,
                    third_status=third_result.status,
                    elapsed_s=third_elapsed_s,
                    arm=arm,
                    third_camera_key=third_camera_key,
                    actual_gripper=actual_gripper,
                    close_target=close_target,
                )
                if third_result.status == "success":
                    _print_decision(
                        "[pick_place][plus][最终]",
                        [
                            ("wrist", "fail"),
                            ("third", "success"),
                            ("final", "success"),
                            ("next", "continue"),
                        ],
                    )
                    emit_log(
                        visualize,
                        source="single_arm_pick_place",
                        stage="completion_check",
                        message=(
                            "腕部检测失败，但第三视角未见目标，"
                            "按成功继续执行"
                        ),
                    )
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
                    _print_decision(
                        "[pick_place][plus][最终]",
                        [
                            ("wrist", "fail"),
                            ("third", "fail"),
                            ("final", "retry"),
                            ("next", "return_home_retry"),
                        ],
                    )
                    execute_return_to_source_sequence(
                        arx=arx,
                        pick_ref=last_result[0],
                        arm=arm,
                        item_type=item_type,
                    )
                    success, error_message = arx.set_special_mode(1, side=arm)
                    if not success:
                        raise RuntimeError(
                            f"failed to reset {arm} arm after return_to_source: "
                            f"{error_message}"
                        )
                except Exception as exc:
                    print(f"放回源位失败，跳过自动重试: {exc}")
                    emit_log(
                        visualize,
                        source="single_arm_pick_place",
                        stage="completion_check",
                        message=f"放回源位失败，跳过自动重试: {exc}",
                    )
                    return last_result
                emit_log(
                    visualize,
                    source="single_arm_pick_place",
                    stage="completion_check",
                    message=(
                        "腕部检测失败，且第三视角仍见目标，"
                        "已放回源位并复位，准备重试"
                    ),
                )
                if retry_limit is not None and retry_idx >= retry_limit:
                    print(
                        "[pick_place][plus] retry limit reached after return-to-source branch"
                    )
                    return last_result
                retry_idx += 1
                if retry_limit is None:
                    print(f"腕部检测失败，已放回源位，开始 retry {retry_idx}/inf")
                else:
                    print(
                        f"腕部检测失败，已放回源位，开始 retry {retry_idx}/{retry_limit}"
                    )
                continue

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
                )
            except Exception as exc:
                print(f"任务检测失败，跳过自动重试: {exc}")
                emit_log(
                    visualize,
                    source="single_arm_pick_place",
                    stage="completion_check",
                    message=f"任务检测失败，跳过自动重试: {exc}",
                )
                return last_result
            check_elapsed_s = time.time() - check_start

            _print_decision(
                "[pick_place][legacy_check]",
                [
                    ("third", check_result.third_status),
                    ("wrist", check_result.wrist_status),
                    ("final", check_result.status),
                    ("elapsed_s", f"{check_elapsed_s:.3f}"),
                ],
            )
            emit_event(
                visualize,
                "completion_check",
                source="single_arm_pick_place",
                status=check_result.status,
                description=check_result.description,
                third_description=check_result.third_description,
                wrist_description=check_result.wrist_description,
                third_status=check_result.third_status,
                wrist_status=check_result.wrist_status,
                elapsed_s=check_elapsed_s,
                arm=arm,
            )
            if check_result.status == "success":
                _print_decision(
                    "[pick_place][legacy_check][最终]",
                    [
                        ("final", "success"),
                        ("next", "continue"),
                    ],
                )
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
            if retry_limit is not None and retry_idx >= retry_limit:
                print("[pick_place][legacy_check] retry limit reached, stop retry")
                return last_result
            _print_decision(
                "[pick_place][legacy_check][最终]",
                [
                    ("final", "fail"),
                    ("next", "open_close_retry"),
                ],
            )
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
