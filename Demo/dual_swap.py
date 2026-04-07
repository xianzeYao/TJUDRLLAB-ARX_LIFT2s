import time
import sys
import re
from dataclasses import dataclass
from typing import Optional

import cv2
import numpy as np

sys.path.append("../ARX_Realenv/ROS2")  # noqa

from arx_ros2_env import ARXRobotEnv  # noqa
from arx_pointing import (
    predict_multi_points_from_multi_image,
    predict_multi_points_from_rgb,
    predict_point_from_rgb,
)
from motion_swap import build_swap_sequence
from point2pos_utils import (
    get_aligned_frames,
    pixel_to_base_point_safe,
)
from visualize_utils import (
    VisualizeContext,
    dispatch_debug_image,
    emit_event,
    emit_log,
    emit_result,
    emit_stage,
    init_keyboard,
    render_dual_swap_debug_view,
    restore_keyboard,
    should_stop,
)


@dataclass
class SweepJudgeResult:
    continue_sweep: bool
    floor_has_target: bool
    dustpan_gained_target: bool
    reason: str


def _parse_bool(text: Optional[str]) -> Optional[bool]:
    if text is None:
        return None
    normalized = text.strip().lower()
    if not normalized:
        return None
    if normalized == "true":
        return True
    if normalized == "false":
        return False
    matches = re.findall(r"\b(true|false)\b", normalized)
    if not matches:
        return None
    return matches[0] == "true"


def _vote_single_image_bool(
    color: np.ndarray,
    prompt: str,
    vote_times: int,
) -> bool:
    if vote_times <= 0:
        raise ValueError(f"vote_times must be positive, got {vote_times}")

    true_count = 0
    false_count = 0
    for _ in range(vote_times):
        _, raw = predict_multi_points_from_rgb(
            image=color,
            text_prompt="",
            all_prompt=prompt,
            assume_bgr=False,
            temperature=0.0,
            return_raw=True,
        )
        parsed = _parse_bool(raw)
        if parsed is True:
            true_count += 1
        elif parsed is False:
            false_count += 1

    if true_count == 0 and false_count == 0:
        raise RuntimeError(f"bool parsing failed. prompt={prompt!r}")
    return true_count > false_count


def _vote_multi_image_bool(
    images: list[np.ndarray],
    prompt: str,
    vote_times: int,
) -> bool:
    if vote_times <= 0:
        raise ValueError(f"vote_times must be positive, got {vote_times}")

    true_count = 0
    false_count = 0
    for _ in range(vote_times):
        _, raw = predict_multi_points_from_multi_image(
            images=images,
            text_prompt="",
            all_prompt=prompt,
            assume_bgr=(False, False),
            temperature=0.0,
            return_raw=True,
        )
        parsed = _parse_bool(raw)
        if parsed is True:
            true_count += 1
        elif parsed is False:
            false_count += 1

    if true_count == 0 and false_count == 0:
        raise RuntimeError(f"bool parsing failed. prompt={prompt!r}")
    return true_count > false_count


def _build_floor_presence_prompt(object_prompt: str) -> str:
    return (
        "This is the current top-camera image during dual sweep. "
        "The left arm holds the dustpan. The right arm holds the broom. "
        f'Focus only on objects matching "{object_prompt}". '
        "Check whether at least one matching object is still on the floor outside the dustpan. "
        "Ignore any matching object already inside the dustpan, at the dustpan mouth, "
        "or being lifted / blocked by the broom during the sweep motion. "
        "Answer True only when a matching object clearly remains on the floor and still needs another sweep. "
        "Output only True or False."
    )


def _build_dustpan_gain_prompt(object_prompt: str) -> str:
    return (
        "Image 1 is the reference image captured after the robot reached the pre-sweep pose. "
        "Image 2 is the current image after the latest sweep. "
        "The left arm holds the dustpan. The right arm holds the broom. "
        f'Focus only on objects matching "{object_prompt}". '
        "Compare only the dustpan interior and dustpan mouth region between the two images. "
        "Ignore any matching object that was already inside the dustpan in image 1. "
        "Do not answer True just because both images already contain trash in the dustpan. "
        "Answer True only if image 2 shows at least one additional matching object that is newly inside "
        "the dustpan or clearly crossing into the dustpan mouth compared with image 1. "
        "Output only True or False."
    )


def _get_top_color_frame(
    arx: ARXRobotEnv,
    target_size: tuple[int, int] = (640, 480),
    max_retries: int = 3,
    retry_sleep_s: float = 0.1,
) -> np.ndarray:
    frames = None
    for _ in range(max(1, int(max_retries))):
        frames = arx.get_camera(target_size=target_size, return_status=False)
        color = frames.get("camera_h_color") if frames else None
        if color is not None:
            return color
        time.sleep(max(0.0, float(retry_sleep_s)))

    available = sorted(frames.keys()) if frames else []
    raise RuntimeError(
        "failed to fetch camera_h_color frame, "
        f"available keys: {available}"
    )


def _apply_dual_swap_detection_mask(color: np.ndarray) -> np.ndarray:
    """Keep only the bottom trapezoid region for dual-sweep target pointing."""
    if color.ndim < 2:
        raise ValueError(f"unexpected image shape: {color.shape}")

    height, width = color.shape[:2]
    top_y = height // 2
    half_top_width = width // 4
    center_x = width // 2
    polygon = np.array(
        [
            [0, height - 1],
            [width - 1, height - 1],
            [center_x + half_top_width, top_y],
            [center_x - half_top_width, top_y],
        ],
        dtype=np.int32,
    )

    mask = np.zeros((height, width), dtype=np.uint8)
    cv2.fillConvexPoly(mask, polygon, 255)

    masked = color.copy()
    masked[mask == 0] = 0
    return masked


def _judge_continue_swap(
    arx: ARXRobotEnv,
    object_prompt: str,
    reference_color: np.ndarray,
    vote_times: int,
    verify_retry_s: float = 0.2,
) -> SweepJudgeResult:
    floor_prompt = _build_floor_presence_prompt(object_prompt)
    dustpan_prompt = _build_dustpan_gain_prompt(object_prompt)

    def _run_once(color: np.ndarray) -> tuple[bool, bool]:
        floor_has_target = _vote_single_image_bool(
            color=color,
            prompt=floor_prompt,
            vote_times=vote_times,
        )
        dustpan_gained_target = _vote_multi_image_bool(
            images=[reference_color, color],
            prompt=dustpan_prompt,
            vote_times=vote_times,
        )
        return floor_has_target, dustpan_gained_target

    current_color = _get_top_color_frame(arx)
    floor_has_target, dustpan_gained_target = _run_once(current_color)
    if floor_has_target:
        return SweepJudgeResult(
            continue_sweep=True,
            floor_has_target=True,
            dustpan_gained_target=dustpan_gained_target,
            reason="target still on floor",
        )
    if dustpan_gained_target:
        return SweepJudgeResult(
            continue_sweep=False,
            floor_has_target=False,
            dustpan_gained_target=True,
            reason="target no longer on floor and dustpan gained target",
        )

    time.sleep(max(0.0, float(verify_retry_s)))
    retry_color = _get_top_color_frame(arx)
    floor_has_target_retry, dustpan_gained_target_retry = _run_once(
        retry_color)
    if floor_has_target_retry:
        return SweepJudgeResult(
            continue_sweep=True,
            floor_has_target=True,
            dustpan_gained_target=dustpan_gained_target_retry,
            reason="retry check still sees target on floor",
        )
    if dustpan_gained_target_retry:
        return SweepJudgeResult(
            continue_sweep=False,
            floor_has_target=False,
            dustpan_gained_target=True,
            reason="retry check confirms dustpan gained target",
        )
    return SweepJudgeResult(
        continue_sweep=True,
        floor_has_target=False,
        dustpan_gained_target=False,
        reason="result uncertain, continue sweep conservatively",
    )


def pick_tools(
    arx: ARXRobotEnv,
    visualize: Optional[VisualizeContext] = None,
) -> None:
    open_action = {
        "left": np.array([0.05, 0, 0, 0, 0, 0, -3.4], dtype=np.float32),
        "right": np.array([0.05, 0, 0, 0, 0, 0, -3.4], dtype=np.float32),
    }
    arx.step_smooth_eef(open_action)
    print("请放取扫把簸箕，5秒后开始夹取...")
    emit_log(
        visualize,
        source="dual_swap",
        stage="pick_tools",
        message="请放取扫把簸箕，5秒后开始夹取...",
    )
    time.sleep(5.0)

    close_action = {
        "left": np.array([0.05, 0, 0, 0, 0, 0, 0.0], dtype=np.float32),
        "right": np.array([0.05, 0, 0, 0, 0, 0, 0.0], dtype=np.float32),
    }
    arx.step_smooth_eef(close_action)
    time.sleep(5.0)

    lift_action = {
        "left": np.array([0.05, 0, 0.1, 0, 0, 0, 0.0], dtype=np.float32),
        "right": np.array([0.05, 0, 0.1, 0, 0, 0, 0.0], dtype=np.float32),
    }
    arx.step_smooth_eef(lift_action)
    time.sleep(1.0)


def release_tools(
    arx: ARXRobotEnv,
    visualize: Optional[VisualizeContext] = None,
) -> None:
    success, error_message = arx.set_special_mode(1)
    if not success:
        raise RuntimeError(f"Failed to home both arms: {error_message}")
    emit_log(
        visualize,
        source="dual_swap",
        stage="release_tools",
        message="Release sweep tools and home both arms.",
    )
    time.sleep(1.0)

    open_action = {
        "left": np.array([0, 0, 0, 0, 0, 0, -3.4], dtype=np.float32),
        "right": np.array([0, 0, 0, 0, 0, 0, -3.4], dtype=np.float32),
    }
    arx.step_smooth_eef(open_action)
    time.sleep(5.0)


def _detect_swap_target(
    arx: ARXRobotEnv,
    object_prompt: str,
    debug_raw: bool,
    depth_median_n: int,
    visualize: Optional[VisualizeContext] = None,
) -> Optional[np.ndarray]:
    while True:
        if should_stop(visualize):
            return None
        color, depth = get_aligned_frames(arx, depth_median_n=depth_median_n)
        if color is None or depth is None:
            continue

        masked_color = _apply_dual_swap_detection_mask(color)
        trash_point = predict_point_from_rgb(masked_color, object_prompt)
        u, v = int(round(trash_point[0])), int(round(trash_point[1]))
        trash_base_point = pixel_to_base_point_safe(
            (u, v),
            depth,
            robot_part="center",
        )
        if trash_base_point is None:
            print(f"预测像素 {(u, v)} 深度无效或像素越界，自动刷新")
            emit_log(
                visualize,
                source="dual_swap",
                stage="detect_target",
                message=f"预测像素 {(u, v)} 深度无效或像素越界，自动刷新",
            )
            continue

        if not debug_raw:
            emit_event(
                visualize,
                "swap_target",
                source="dual_swap",
                object_prompt=object_prompt,
                pixel=(u, v),
                target_base_point=np.asarray(
                    trash_base_point, dtype=np.float32).tolist(),
            )
            return trash_base_point

        vis = render_dual_swap_debug_view(
            color,
            (u, v),
            object_prompt=object_prompt,
        )
        emit_event(
            visualize,
            "swap_target",
            source="dual_swap",
            object_prompt=object_prompt,
            pixel=(u, v),
            target_base_point=np.asarray(
                trash_base_point, dtype=np.float32).tolist(),
        )
        debug_result = dispatch_debug_image(
            visualize,
            source="dual_swap",
            panel="manip",
            image=vis,
            window_name="dual_swap_detect",
            object_prompt=object_prompt,
            pixel=(u, v),
        )
        if debug_result is None:
            return None
        if debug_result:
            return trash_base_point
        continue


def dual_swap(
    arx: ARXRobotEnv,
    object_prompt: str = "a white crumpled paper on the floor",
    debug_raw: bool = True,
    depth_median_n: int = 10,
    judge_vote_times: int = 3,
    required_consecutive_dustpan_gain: int = 2,
    visualize: Optional[VisualizeContext] = None,
) -> Optional[np.ndarray]:
    old_settings = init_keyboard()
    try:
        emit_stage(
            visualize,
            source="dual_swap",
            stage="start",
            message=f"Start dual sweep for {object_prompt}",
            object_prompt=object_prompt,
        )
        target_base_point = _detect_swap_target(
            arx,
            object_prompt=object_prompt,
            debug_raw=debug_raw,
            depth_median_n=depth_median_n,
            visualize=visualize,
        )
        if target_base_point is None:
            emit_result(
                visualize,
                source="dual_swap",
                status="canceled",
                message="dual sweep canceled",
                object_prompt=object_prompt,
            )
            return None

        swap_seq = build_swap_sequence(target_base_point)
        if not swap_seq:
            return target_base_point

        arx.step_smooth_eef(swap_seq[0])
        reference_color = _get_top_color_frame(arx)
        sweep_actions = swap_seq[1:]
        actions_per_sweep = 4
        max_sweeps = len(sweep_actions) // actions_per_sweep
        consecutive_dustpan_gain_count = 0
        for sweep_idx in range(max_sweeps):
            if should_stop(visualize):
                emit_result(
                    visualize,
                    source="dual_swap",
                    status="stopped",
                    message="dual sweep stopped",
                    object_prompt=object_prompt,
                )
                return None
            start = sweep_idx * actions_per_sweep
            end = start + actions_per_sweep
            emit_stage(
                visualize,
                source="dual_swap",
                stage="sweep",
                message=f"Execute sweep {sweep_idx + 1}/{max_sweeps}",
                sweep_index=sweep_idx + 1,
                sweep_total=max_sweeps,
            )
            for action in sweep_actions[start:end]:
                if should_stop(visualize):
                    emit_result(
                        visualize,
                        source="dual_swap",
                        status="stopped",
                        message="dual sweep stopped",
                        object_prompt=object_prompt,
                    )
                    return None
                arx.step_smooth_eef(action)
            if sweep_idx == max_sweeps - 1:
                break
            try:
                judge_result = _judge_continue_swap(
                    arx,
                    object_prompt=object_prompt,
                    reference_color=reference_color,
                    vote_times=judge_vote_times,
                )
            except Exception as exc:
                consecutive_dustpan_gain_count = 0
                print(
                    f"[auto-judge] failed after sweep {sweep_idx + 1}: {exc}. "
                    "Continue sweep conservatively."
                )
                emit_log(
                    visualize,
                    source="dual_swap",
                    stage="judge",
                    message=(
                        f"[auto-judge] failed after sweep {sweep_idx + 1}: "
                        f"{exc}. Continue sweep conservatively."
                    ),
                )
                continue
            print(
                f"[auto-judge] sweep {sweep_idx + 1}/{max_sweeps}: "
                f"floor_has_target={judge_result.floor_has_target}, "
                f"dustpan_gained_target={judge_result.dustpan_gained_target}, "
                f"continue={judge_result.continue_sweep}, "
                f"reason={judge_result.reason}"
            )
            emit_event(
                visualize,
                "judge_result",
                source="dual_swap",
                sweep_index=sweep_idx + 1,
                sweep_total=max_sweeps,
                floor_has_target=judge_result.floor_has_target,
                dustpan_gained_target=judge_result.dustpan_gained_target,
                continue_sweep=judge_result.continue_sweep,
                reason=judge_result.reason,
            )
            if judge_result.dustpan_gained_target:
                consecutive_dustpan_gain_count += 1
            else:
                consecutive_dustpan_gain_count = 0
            if not judge_result.continue_sweep:
                print("[auto-judge] original stop condition met, retract sweep.")
                break
            if consecutive_dustpan_gain_count >= required_consecutive_dustpan_gain:
                print(
                    "[auto-judge] dustpan gained target in consecutive checks, "
                    "retract sweep."
                )
                break
            if judge_result.dustpan_gained_target:
                print(
                    "[auto-judge] dustpan gained target once, "
                    "wait for one more confirmation before retracting."
                )
        emit_result(
            visualize,
            source="dual_swap",
            status="success",
            message="dual sweep completed",
            object_prompt=object_prompt,
            target_base_point=np.asarray(
                target_base_point, dtype=np.float32).tolist(),
        )
        return target_base_point
    finally:
        cv2.destroyAllWindows()
        restore_keyboard(old_settings)


def main():
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
        pick_tools(arx)
        dual_swap(
            arx,
            object_prompt="a white crumpled paper on the floor",
            debug_raw=True,
            depth_median_n=5,
        )
        release_tools(arx)
    finally:
        arx.close()


if __name__ == "__main__":
    main()
