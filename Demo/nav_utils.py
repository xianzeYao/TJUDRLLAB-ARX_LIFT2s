from __future__ import annotations

import math
from typing import List, Tuple

from demo_utils import step_base_duration

BASE_FORWARD_SPEED = 0.155
BASE_ROTATE_SPEED = (2.0 * math.pi) / 20.11
FORWARD_VX_CMD = 0.75
ROTATE_VZ_CMD = 1.0

Action = Tuple[str, float]
ExecutedRotation = Tuple[float, float]


def path_to_actions(
    path: List[Tuple[float, float]],
    init_yaw: float = 0.0,
) -> List[Action]:
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
) -> List[ExecutedRotation]:
    executed_rotations: List[ExecutedRotation] = []
    for action, value in actions:
        if action == "rotate":
            if value <= 0:
                duration = max(-value - 0.1, 0.1) / BASE_ROTATE_SPEED
                vz = -ROTATE_VZ_CMD
            else:
                duration = value / BASE_ROTATE_SPEED
                vz = ROTATE_VZ_CMD

            step_base_duration(
                arx,
                vx=0.0,
                vy=0.0,
                vz=float(vz),
                duration=float(duration),
            )
            executed_rotations.append((float(vz), float(duration)))
        elif action == "forward":
            remaining = value - distance
            if remaining <= 0:
                continue
            step_base_duration(
                arx,
                vx=FORWARD_VX_CMD,
                vy=0.0,
                vz=0.0,
                duration=float(remaining / BASE_FORWARD_SPEED),
            )
    return executed_rotations


def recover_rotations(
    arx,
    executed_rotations: List[ExecutedRotation],
) -> None:
    for vz, duration in reversed(executed_rotations):
        step_base_duration(
            arx,
            vx=0.0,
            vy=0.0,
            vz=float(-vz),
            duration=float(duration),
        )
