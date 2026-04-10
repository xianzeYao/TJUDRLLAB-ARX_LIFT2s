from __future__ import annotations

"""底盘路径点到旋转/前进动作的简单转换和执行工具。"""

import math
import time
from typing import Callable, List, Optional, Tuple

from .demo_utils import step_base_duration

BASE_FORWARD_SPEED = 0.155
BASE_ROTATE_SPEED = (2.0 * math.pi) / 20.11
FORWARD_VX_CMD = 0.75
ROTATE_VZ_CMD = 1.0

Action = Tuple[str, float]
ExecutedRotation = Tuple[float, float]


def step_base_lift_duration(
    arx,
    vx: float,
    vy: float,
    vz: float,
    height: Optional[float],
    duration: float,
    should_stop: Optional[Callable[[], bool]] = None,
    poll_interval_s: float = 0.01,
    lift_step: float = 0.03,
) -> bool:
    """在固定时长内持续发送底盘+升降命令。"""
    if duration <= 0:
        print("base+lift move duration must be positive")
        return True

    base_status = arx.get_robot_status().get("base")
    curr_height = float(base_status.height) if base_status is not None else 0.0
    target_height = curr_height if height is None else float(height)
    completed = True
    end_time = time.monotonic() + float(duration)
    poll_interval_s = max(0.0, float(poll_interval_s))
    lift_step = abs(float(lift_step))

    try:
        while True:
            remaining = end_time - time.monotonic()
            if remaining <= 0.0:
                break
            if should_stop is not None and should_stop():
                completed = False
                break

            if abs(target_height - curr_height) > 1e-6:
                if abs(target_height - curr_height) <= lift_step:
                    curr_height = target_height
                else:
                    curr_height += lift_step if target_height > curr_height else -lift_step

            arx.step_base_lift(
                float(vx),
                float(vy),
                float(vz),
                float(curr_height),
            )
            time.sleep(min(poll_interval_s, remaining))
    finally:
        arx.step_base_lift(0.0, 0.0, 0.0, float(curr_height))

    return completed


def path_to_actions(
    path: List[Tuple[float, float]],
    init_yaw: float = 0.0,
) -> List[Action]:
    """把二维路径点转换为依次执行的 rotate/forward 动作列表。"""
    actions: List[Action] = []
    cur_yaw = init_yaw

    for i in range(1, len(path)):
        x0, y0 = path[i - 1]
        x1, y1 = path[i]

        dx = x1 - x0
        dy = y1 - y0

        target_yaw = math.atan2(dy, dx)
        d_yaw = target_yaw - cur_yaw

        while d_yaw > math.pi:
            d_yaw -= 2 * math.pi
        while d_yaw < -math.pi:
            d_yaw += 2 * math.pi

        dist = math.hypot(dx, dy)

        if abs(d_yaw) > 1e-3:
            actions.append(("rotate", -d_yaw))
            cur_yaw = target_yaw

        if dist > 1e-3:
            actions.append(("forward", dist))

    return actions


def execute_nav_actions(
    arx,
    actions: List[Action],
    distance: float,
    stop_checker: Optional[Callable[[], bool]] = None,
    lift_height_target: Optional[float] = None,
) -> Tuple[List[ExecutedRotation], bool]:
    """执行底盘动作列表，并记录实际做过的旋转动作用于回退。"""
    executed_rotations: List[ExecutedRotation] = []
    interrupted = False

    def _run_action(vx: float, vz: float, duration: float) -> bool:
        return step_base_lift_duration(
            arx,
            vx=float(vx),
            vy=0.0,
            vz=float(vz),
            height=lift_height_target,
            duration=float(duration),
            should_stop=stop_checker,
        )

    for action, value in actions:
        if action == "rotate":
            if value <= 0:
                duration = max(-value - 0.1, 0.1) / BASE_ROTATE_SPEED
                vz = -ROTATE_VZ_CMD
            else:
                duration = value / BASE_ROTATE_SPEED
                vz = ROTATE_VZ_CMD

            completed = _run_action(
                vx=0.0, vz=float(vz), duration=float(duration))
            if not completed:
                interrupted = True
                break
            executed_rotations.append((float(vz), float(duration)))
        elif action == "forward":
            remaining = value - distance
            if remaining <= 0:
                continue
            completed = _run_action(
                vx=FORWARD_VX_CMD,
                vz=0.0,
                duration=float(remaining / BASE_FORWARD_SPEED),
            )
            if not completed:
                interrupted = True
                break

    if not interrupted and lift_height_target is not None:
        base_status = arx.get_robot_status().get("base")
        actual_height = (
            float(base_status.height)
            if base_status is not None else float(lift_height_target)
        )
        if abs(float(lift_height_target) - actual_height) > 1e-3:
            arx.step_lift(float(lift_height_target))

    return executed_rotations, interrupted


def recover_rotations(
    arx,
    executed_rotations: List[ExecutedRotation],
) -> None:
    """按逆序撤销之前记录下来的旋转动作。"""
    for vz, duration in reversed(executed_rotations):
        step_base_duration(
            arx,
            vx=0.0,
            vy=0.0,
            vz=float(-vz),
            duration=float(duration),
        )
