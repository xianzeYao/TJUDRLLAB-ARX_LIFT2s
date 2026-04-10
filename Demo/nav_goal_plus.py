from __future__ import annotations

import math
import sys
import time
from dataclasses import dataclass
from pathlib import Path
import threading

import cv2
import numpy as np

CURRENT_DIR = Path(__file__).resolve().parent
ROOT_DIR = CURRENT_DIR.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.append(str(ROOT_DIR))

from ARX_Realenv.ROS2.arx_ros2_env import ARXRobotEnv
from nav_goal import (
    _apply_roi_focus,
    _build_search_roi_polygon,
    _confirm_debug_view,
    _get_key_nonblock,
    _init_keyboard,
    _restore_keyboard,
    _select_goal_point,
    _vote_goal_presence,
)
from utils import (
    BASE_FORWARD_SPEED,
    BASE_ROTATE_SPEED,
    FORWARD_VX_CMD,
    ROTATE_VZ_CMD,
    estimate_lift_from_goal_z,
    get_aligned_frames,
    predict_multi_points_from_rgb,
    recover_rotations,
    step_base_lift_duration,
)


@dataclass
class NavGoalPlusSnapshot:
    captured_at: float
    color: np.ndarray
    depth: np.ndarray | None
    rotating_search: bool
    use_initial_search_roi: bool


@dataclass
class NavGoalPlusInference:
    captured_at: float
    finished_at: float
    color: np.ndarray
    roi_polygon: np.ndarray | None
    points: list[tuple[float, float]]
    goal_pixel: tuple[float, float] | None
    goal_pw: np.ndarray | None
    goal_detected: bool
    error: str | None


def _estimate_forward_speed(vx_cmd: float) -> float:
    if abs(float(vx_cmd)) <= 1e-6:
        return 0.0
    return BASE_FORWARD_SPEED * abs(float(vx_cmd)) / float(FORWARD_VX_CMD)


def _estimate_rotate_speed(vz_cmd: float) -> float:
    if abs(float(vz_cmd)) <= 1e-6:
        return 0.0
    return BASE_ROTATE_SPEED * abs(float(vz_cmd)) / float(ROTATE_VZ_CMD)


PLUS_LOOP_SLEEP_S = 0.05
PLUS_FORWARD_VX_CMD = FORWARD_VX_CMD
PLUS_ROTATE_VZ_CMD = ROTATE_VZ_CMD
PLUS_SEARCH_VZ_CMD = ROTATE_VZ_CMD
PLUS_FORWARD_STAGE_RATIOS = (1.0 / 3.0, 0.5, 1.0)
PLUS_NEGATIVE_ROTATE_COMP_RAD = 0.1


def _infer_goal_snapshot(
    snapshot: NavGoalPlusSnapshot,
    goal: str,
    offset: float,
    vote_times: int,
) -> NavGoalPlusInference:
    color = snapshot.color
    depth = snapshot.depth
    roi_polygon = _build_search_roi_polygon(color.shape)
    active_roi = (
        roi_polygon
        if snapshot.rotating_search or snapshot.use_initial_search_roi
        else None
    )

    if vote_times > 0:
        presence_color = (
            _apply_roi_focus(color, active_roi)
            if active_roi is not None
            else color
        )
        detect_goal = _vote_goal_presence(
            presence_color,
            goal=goal,
            vote_times=vote_times,
        )
        if not detect_goal:
            return NavGoalPlusInference(
                captured_at=snapshot.captured_at,
                finished_at=time.monotonic(),
                color=color,
                roi_polygon=active_roi,
                points=[],
                goal_pixel=None,
                goal_pw=None,
                goal_detected=False,
                error=None,
            )
    else:
        detect_goal = True

    if depth is None:
        return NavGoalPlusInference(
            captured_at=snapshot.captured_at,
            finished_at=time.monotonic(),
            color=color,
            roi_polygon=active_roi,
            points=[],
            goal_pixel=None,
            goal_pw=None,
            goal_detected=detect_goal,
            error="depth frame unavailable",
        )

    try:
        points, _ = predict_multi_points_from_rgb(
            color,
            text_prompt=goal,
            all_prompt=None,
            assume_bgr=False,
            return_raw=True,
        )
    except Exception as exc:
        return NavGoalPlusInference(
            captured_at=snapshot.captured_at,
            finished_at=time.monotonic(),
            color=color,
            roi_polygon=active_roi,
            points=[],
            goal_pixel=None,
            goal_pw=None,
            goal_detected=detect_goal,
            error=f"point prediction failed: {exc}",
        )

    if points is None or len(points) == 0:
        return NavGoalPlusInference(
            captured_at=snapshot.captured_at,
            finished_at=time.monotonic(),
            color=color,
            roi_polygon=active_roi,
            points=[],
            goal_pixel=None,
            goal_pw=None,
            goal_detected=False if vote_times <= 0 else detect_goal,
            error="goal detected but no point found",
        )

    try:
        goal_pixel, goal_pw = _select_goal_point(
            points,
            depth,
            offset=offset,
            roi_polygon=active_roi,
        )
    except ValueError as exc:
        return NavGoalPlusInference(
            captured_at=snapshot.captured_at,
            finished_at=time.monotonic(),
            color=color,
            roi_polygon=active_roi,
            points=[(float(p[0]), float(p[1])) for p in points],
            goal_pixel=None,
            goal_pw=None,
            goal_detected=detect_goal,
            error=str(exc),
        )

    return NavGoalPlusInference(
        captured_at=snapshot.captured_at,
        finished_at=time.monotonic(),
        color=color,
        roi_polygon=active_roi,
        points=[(float(p[0]), float(p[1])) for p in points],
        goal_pixel=(float(goal_pixel[0]), float(goal_pixel[1])),
        goal_pw=np.asarray(goal_pw, dtype=np.float32).copy(),
        goal_detected=True,
        error=None,
    )


def nav_to_goal_plus(
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
):
    old_settings = _init_keyboard()
    debug_checked_once = False
    rotating_search = False
    inference_count = 0
    last_printed_state: str | None = None
    last_result = None
    rotation_history: list[tuple[float, float]] = []
    forward_stage_idx = 0
    last_motion_end_at: float | None = None
    require_post_motion_capture_after: float | None = None
    latest_candidate: NavGoalPlusInference | None = None
    latest_candidate_capture_at: float | None = None
    last_consumed_capture_at: float | None = None
    worker_lock = threading.Lock()
    worker_stop = threading.Event()

    def _publish_candidate(candidate: NavGoalPlusInference) -> None:
        nonlocal latest_candidate, latest_candidate_capture_at, inference_count
        with worker_lock:
            latest_candidate = candidate
            latest_candidate_capture_at = candidate.captured_at
            inference_count += 1

    def _read_latest_candidate() -> tuple[
        NavGoalPlusInference | None,
        float | None,
        int,
    ]:
        with worker_lock:
            return latest_candidate, latest_candidate_capture_at, int(inference_count)

    def _read_search_flags() -> tuple[bool, bool]:
        with worker_lock:
            return bool(rotating_search), bool(use_initial_search_roi)

    def _inference_worker() -> None:
        while not worker_stop.is_set():
            color, depth = get_aligned_frames(
                arx, depth_median_n=depth_median_n)
            if color is None:
                time.sleep(PLUS_LOOP_SLEEP_S)
                continue
            worker_rotating_search, worker_use_initial_roi = _read_search_flags()
            snapshot = NavGoalPlusSnapshot(
                captured_at=time.monotonic(),
                color=color.copy(),
                depth=None if depth is None else depth.copy(),
                rotating_search=worker_rotating_search,
                use_initial_search_roi=worker_use_initial_roi,
            )
            candidate = _infer_goal_snapshot(
                snapshot,
                goal,
                float(offset),
                int(vote_times),
            )
            _publish_candidate(candidate)

    def _recover_rotations_if_needed() -> None:
        if not rotate_recover or not rotation_history:
            return
        total_duration = sum(duration for _, duration in rotation_history)
        _set_state(
            f"recovering {len(rotation_history)} rotation segments, total={total_duration:.2f}s"
        )
        try:
            arx.step_base(0.0, 0.0, 0.0)
            recover_rotations(arx, rotation_history)
        finally:
            rotation_history.clear()

    def _start_rotate_search() -> None:
        nonlocal rotating_search
        if rotating_search:
            return
        with worker_lock:
            rotating_search = True
        arx.step_base(0.0, 0.0, float(PLUS_SEARCH_VZ_CMD))

    def _stop_rotate_search() -> None:
        nonlocal rotating_search
        if not rotating_search:
            return
        with worker_lock:
            rotating_search = False
        arx.step_base(0.0, 0.0, 0.0)

    def _run_discrete_motion(
        *,
        vx: float,
        vz: float,
        duration: float,
        record_rotation: bool,
        label: str,
        lift_height_target: float | None = None,
    ) -> bool:
        nonlocal last_motion_end_at
        if duration <= 1e-4:
            return True
        state_msg = f"{label}: duration={duration:.2f}s, vx={vx:.2f}, vz={vz:.2f}"
        if lift_height_target is not None:
            state_msg += f", lift={lift_height_target:.2f}"
        _set_state(state_msg)
        _stop_rotate_search()
        def stop_checker(): return _get_key_nonblock() == "n"
        completed = step_base_lift_duration(
            arx,
            vx=float(vx),
            vy=0.0,
            vz=float(vz),
            height=lift_height_target,
            duration=float(duration),
            should_stop=stop_checker,
        )
        if not completed:
            last_motion_end_at = time.monotonic()
            print("Emergency stop received.")
            return False
        last_motion_end_at = time.monotonic()
        if record_rotation and abs(vz) > 1e-6:
            rotation_history.append((float(vz), float(duration)))
        return True

    def _set_state(message: str) -> None:
        nonlocal last_printed_state
        if message == last_printed_state:
            return
        print(f"[nav_goal_plus] {message}")
        last_printed_state = message

    try:
        worker_thread = threading.Thread(
            target=_inference_worker,
            name="nav-goal-plus-infer",
            daemon=True,
        )
        worker_thread.start()
        arx.step_lift(lift_height)

        while True:
            key = _get_key_nonblock()
            if key == "q":
                _stop_rotate_search()
                return last_result
            if key == "n":
                _stop_rotate_search()
                arx.step_base(0.0, 0.0, 0.0)
                print("Emergency stop received.")
                return last_result

            candidate, candidate_capture_at, current_inference_count = _read_latest_candidate()
            if (
                candidate is None
                or candidate_capture_at is None
                or (
                    last_consumed_capture_at is not None
                    and candidate_capture_at <= last_consumed_capture_at
                )
            ):
                if rotating_search:
                    _set_state("searching by rotation")
                else:
                    _set_state("waiting for a valid goal result")
                time.sleep(PLUS_LOOP_SLEEP_S)
                continue

            if (
                require_post_motion_capture_after is not None
                and candidate_capture_at <= require_post_motion_capture_after
            ):
                last_consumed_capture_at = candidate_capture_at
                _set_state("waiting for post-motion goal result")
                time.sleep(PLUS_LOOP_SLEEP_S)
                continue

            last_consumed_capture_at = candidate_capture_at
            require_post_motion_capture_after = None

            if candidate.goal_pw is None:
                if candidate.error:
                    _set_state(candidate.error)
                elif not candidate.goal_detected:
                    _set_state(f"no goal detected for: {goal}")
                if rotate_search_on_miss:
                    _start_rotate_search()
                    _set_state("searching by rotation")
                    time.sleep(PLUS_LOOP_SLEEP_S)
                    continue
                if continuous:
                    time.sleep(PLUS_LOOP_SLEEP_S)
                    continue
                return None

            _stop_rotate_search()

            if debug_raw and not debug_checked_once:
                debug_result = _confirm_debug_view(
                    candidate.color,
                    candidate.points,
                    candidate.goal_pixel,
                    roi_polygon=candidate.roi_polygon,
                    nav_prompt=goal,
                )
                if debug_result is None:
                    return last_result
                if not debug_result:
                    _set_state("debug refresh requested")
                    continue
                debug_checked_once = True

            goal_pw = candidate.goal_pw.copy()
            goal_x = float(goal_pw[0])
            goal_y = float(goal_pw[1])
            radial_dist = math.hypot(goal_x, goal_y)
            yaw_err = math.atan2(-goal_y, goal_x)
            remaining = radial_dist - float(distance)
            motion_lift_target = None
            if use_goal_z_for_lift:
                base_status = arx.get_robot_status().get("base")
                current_lift = float(
                    base_status.height
                ) if base_status is not None else float(lift_height)
                motion_lift_target = float(
                    estimate_lift_from_goal_z(
                        goal_z=float(goal_pw[2]),
                        current_lift=current_lift,
                        target_goal_z=target_goal_z,
                    )
                )
            last_result = (
                goal_pw.copy(),
                {
                    "goal": goal,
                    "goal_pixel": candidate.goal_pixel,
                    "inference_count": current_inference_count,
                    "captured_at": candidate.captured_at,
                },
            )

            if remaining <= 0.0:
                arx.step_base(0.0, 0.0, 0.0)
                _recover_rotations_if_needed()
                rotation_history.clear()
                _set_state(
                    f"arrived: dist={radial_dist:.3f}, yaw_err={math.degrees(yaw_err):.1f} deg"
                )
                if not continuous:
                    return last_result
                forward_stage_idx = 0
                continue

            current_stage_idx = min(
                forward_stage_idx,
                len(PLUS_FORWARD_STAGE_RATIOS) - 1,
            )
            is_final_stage = current_stage_idx >= len(
                PLUS_FORWARD_STAGE_RATIOS) - 1

            if abs(yaw_err) > 1e-3:
                rotate_speed = _estimate_rotate_speed(
                    float(PLUS_ROTATE_VZ_CMD))
                if rotate_speed <= 1e-6:
                    raise ValueError(
                        "PLUS_ROTATE_VZ_CMD must be non-zero for rotate stage"
                    )
                rotate_angle = abs(float(yaw_err))
                rotate_duration = rotate_angle / rotate_speed
                if yaw_err > 0.0:
                    rotate_duration = max(
                        rotate_angle - float(PLUS_NEGATIVE_ROTATE_COMP_RAD),
                        float(PLUS_NEGATIVE_ROTATE_COMP_RAD),
                    ) / rotate_speed
                    rotate_cmd = -float(PLUS_ROTATE_VZ_CMD)
                else:
                    rotate_cmd = float(PLUS_ROTATE_VZ_CMD)
                if not _run_discrete_motion(
                    vx=0.0,
                    vz=rotate_cmd,
                    duration=rotate_duration,
                    record_rotation=True,
                    lift_height_target=motion_lift_target,
                    label=(
                        "rotate-full "
                        f"(yaw_err={math.degrees(yaw_err):.1f} deg)"
                    ),
                ):
                    return last_result
                if is_final_stage and last_motion_end_at is not None:
                    require_post_motion_capture_after = last_motion_end_at
                    _set_state(
                        "waiting for post-rotate goal result before final stage")
                    continue

            forward_speed = _estimate_forward_speed(float(PLUS_FORWARD_VX_CMD))
            if forward_speed <= 1e-6:
                raise ValueError(
                    "PLUS_FORWARD_VX_CMD must be non-zero for forward stage"
                )
            forward_ratio = float(PLUS_FORWARD_STAGE_RATIOS[current_stage_idx])
            forward_distance = max(float(remaining) * forward_ratio, 0.0)
            if current_stage_idx >= len(PLUS_FORWARD_STAGE_RATIOS) - 1:
                forward_distance = max(float(remaining), 0.0)
            forward_duration = forward_distance / forward_speed
            if not _run_discrete_motion(
                vx=float(PLUS_FORWARD_VX_CMD),
                vz=0.0,
                duration=forward_duration,
                record_rotation=False,
                lift_height_target=motion_lift_target,
                label=(
                    f"forward-stage-{current_stage_idx + 1}/3 "
                    f"(remaining={remaining:.3f}m, execute={forward_distance:.3f}m)"
                ),
            ):
                return last_result
            if forward_stage_idx < len(PLUS_FORWARD_STAGE_RATIOS) - 1:
                forward_stage_idx += 1
                if (
                    forward_stage_idx >= len(PLUS_FORWARD_STAGE_RATIOS) - 1
                    and last_motion_end_at is not None
                ):
                    require_post_motion_capture_after = last_motion_end_at
            continue
    finally:
        try:
            worker_stop.set()
            if "worker_thread" in locals():
                worker_thread.join(timeout=1.0)
            _stop_rotate_search()
            arx.step_base(0.0, 0.0, 0.0)
        except Exception:
            pass
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
        result = nav_to_goal_plus(
            arx,
            goal="a tennis ball",
            distance=0.45,
            lift_height=14.5,
            offset=0.23,
            use_goal_z_for_lift=True,
            rotate_recover=True,
            continuous=False,
            debug_raw=True,
            depth_median_n=5,
            rotate_search_on_miss=False,
            vote_times=0,
        )
        print(f"[nav_goal_plus_result] {result}")
    finally:
        arx.close()


if __name__ == "__main__":
    main()
