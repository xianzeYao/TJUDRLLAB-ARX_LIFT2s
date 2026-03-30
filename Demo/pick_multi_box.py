from __future__ import annotations
from demo_utils import step_base_duration

import heapq
import re
import sys
import time
from pathlib import Path
from typing import Dict, List, Literal, TYPE_CHECKING, Tuple


CURRENT_DIR = Path(__file__).resolve().parent
ROOT_DIR = CURRENT_DIR.parent
ROS2_DIR = ROOT_DIR / "ARX_Realenv" / "ROS2"

if str(CURRENT_DIR) not in sys.path:
    sys.path.append(str(CURRENT_DIR))
if str(ROS2_DIR) not in sys.path:
    sys.path.append(str(ROS2_DIR))

if TYPE_CHECKING:
    from arx_ros2_env import ARXRobotEnv


ColorName = Literal["green", "yellow", "blue"]
PositionName = Literal["start", "green", "yellow", "blue"]
PromptStep = Tuple[ColorName, int]

COLOR_ORDER: tuple[ColorName, ...] = ("green", "yellow", "blue")
ALL_POSITIONS: tuple[PositionName, ...] = ("start", "green", "yellow", "blue")
COLOR_TO_LIFT_HEIGHT: Dict[ColorName, float] = {
    "green": 18.0,
    "yellow": 0.0,
    "blue": 18.0,
}
START_LIFT_HEIGHT = 0.0
MOVE_VX = 0.6
MOVE_VY = 0.55
START_MOVE_DURATION = 3.0
BETWEEN_MOVE_DURATION = 0.7
SETTLE_S = 0.5
PICK_ARM_SIDE = "fit"

BASE_MOTION_BY_TRANSITION: Dict[Tuple[PositionName, PositionName], Tuple[float, float, float, float]] = {
    ("start", "green"): (MOVE_VX, -MOVE_VY, 0.0, START_MOVE_DURATION),
    ("start", "yellow"): (MOVE_VX, 0.0, 0.0, START_MOVE_DURATION),
    ("start", "blue"): (MOVE_VX, MOVE_VY, 0.0, START_MOVE_DURATION),
    ("yellow", "green"): (MOVE_VX, -MOVE_VY, 0.0, BETWEEN_MOVE_DURATION),
    ("yellow", "blue"): (MOVE_VX, MOVE_VY, 0.0, BETWEEN_MOVE_DURATION),
}

PROMPT_ITEM_PATTERN = re.compile(
    r"([1-6])\s*(green|yellow|blue)\b", re.IGNORECASE)
PROMPT_SEPARATOR_PATTERN = re.compile(r"[,，;；/]+")
COLOR_TO_PICK_PROMPT: Dict[ColorName, str] = {
    "green": "the nearest green object",
    "yellow": "the nearest yellow object",
    "blue": "the nearest blue object",
}


def _normalize_prompt(prompt: str) -> str:
    normalized = prompt.strip().lower()
    normalized = PROMPT_SEPARATOR_PATTERN.sub(" ", normalized)
    normalized = re.sub(r"\s+", " ", normalized)
    return normalized


def parse_prompt_steps(prompt: str) -> List[PromptStep]:
    normalized = _normalize_prompt(prompt)
    if not normalized:
        raise ValueError("prompt is empty")

    matches = list(PROMPT_ITEM_PATTERN.finditer(normalized))
    if not matches:
        raise ValueError(
            "prompt format invalid, expected items like '1 blue 2 yellow' with num in 1-6"
        )

    remainder = PROMPT_ITEM_PATTERN.sub(" ", normalized)
    remainder = re.sub(r"\s+", " ", remainder).strip()
    if remainder:
        raise ValueError(f"prompt contains unsupported text: {remainder!r}")

    counts: Dict[ColorName, int] = {color: 0 for color in COLOR_ORDER}
    steps: List[PromptStep] = []
    for match in matches:
        count = int(match.group(1))
        color = match.group(2).lower()
        counts[color] += count
        steps.append((color, count))

    for color, count in counts.items():
        if count > 6:
            raise ValueError(
                f"color {color!r} total count must be <= 6, got {count}"
            )

    if not steps:
        raise ValueError("no valid color target found in prompt")
    return steps


def _format_plan(steps: List[PromptStep]) -> str:
    return ", ".join(f"{color} x{count}" for color, count in steps)


def _reverse_motion(
    motion: Tuple[float, float, float, float],
) -> Tuple[float, float, float, float]:
    vx, vy, vz, duration = motion
    return (-vx, -vy, -vz, duration)


def _get_direct_motion(
    src: PositionName,
    dst: PositionName,
) -> Tuple[float, float, float, float] | None:
    motion = BASE_MOTION_BY_TRANSITION.get((src, dst))
    if motion is not None:
        return motion
    reverse_motion = BASE_MOTION_BY_TRANSITION.get((dst, src))
    if reverse_motion is not None:
        return _reverse_motion(reverse_motion)
    return None


def _plan_transition(
    current_position: PositionName,
    target_position: PositionName,
) -> List[Tuple[PositionName, PositionName, Tuple[float, float, float, float]]]:
    if current_position == target_position:
        return []

    dist: Dict[PositionName, float] = {current_position: 0.0}
    prev: Dict[PositionName, PositionName] = {}
    heap: List[Tuple[float, PositionName]] = [(0.0, current_position)]

    while heap:
        cost, node = heapq.heappop(heap)
        if cost > dist.get(node, float("inf")):
            continue
        if node == target_position:
            break
        for neighbor in ALL_POSITIONS:
            if neighbor == node:
                continue
            motion = _get_direct_motion(node, neighbor)
            if motion is None:
                continue
            next_cost = cost + float(motion[3])
            if next_cost >= dist.get(neighbor, float("inf")):
                continue
            dist[neighbor] = next_cost
            prev[neighbor] = node
            heapq.heappush(heap, (next_cost, neighbor))

    if target_position not in dist:
        raise ValueError(
            f"unsupported transition: {current_position!r} -> {target_position!r}"
        )

    path: List[PositionName] = [target_position]
    while path[-1] != current_position:
        parent = prev.get(path[-1])
        if parent is None:
            raise ValueError(
                f"failed to build transition path: {current_position!r} -> {target_position!r}"
            )
        path.append(parent)
    path.reverse()

    motions: List[Tuple[PositionName, PositionName, Tuple[float, float, float, float]]] = []
    for src, dst in zip(path[:-1], path[1:]):
        motion = _get_direct_motion(src, dst)
        if motion is None:
            raise ValueError(
                f"missing direct motion for path segment: {src!r} -> {dst!r}"
            )
        motions.append((src, dst, motion))
    return motions


def move_to_position(
    arx: ARXRobotEnv,
    current_position: PositionName,
    target_position: PositionName,
    *,
    settle_s: float = SETTLE_S,
    apply_lift: bool = True,
) -> PositionName:
    motions = _plan_transition(current_position, target_position)
    if not motions:
        print(f"[pick_multi_box] already at {target_position}, no base move")
    for src, dst, (vx, vy, vz, duration) in motions:
        print(
            "[pick_multi_box] "
            f"move {src} -> {dst}, "
            f"vx={vx:.2f}, vy={vy:.2f}, vz={vz:.2f}, duration={duration:.1f}s"
        )
        step_base_duration(arx, vx=vx, vy=vy, vz=vz, duration=duration)

    if apply_lift and target_position in COLOR_TO_LIFT_HEIGHT:
        target_height = COLOR_TO_LIFT_HEIGHT[target_position]
        print(f"[pick_multi_box] lift -> {target_position} ({target_height:.1f})")
        arx.step_lift(target_height)

    if settle_s > 0.0:
        time.sleep(settle_s)
    return target_position


def _pick_color_boxes(
    arx: ARXRobotEnv,
    color: ColorName,
    count: int,
    *,
    debug: bool,
    depth_median_n: int,
) -> bool:
    from single_arm_pick_place import single_arm_pick_place

    pick_prompt = COLOR_TO_PICK_PROMPT[color]
    for idx in range(1, count + 1):
        print(
            "[pick_multi_box] "
            f"pick {idx}/{count} at {color}: {pick_prompt}"
        )
        _, _, arm_used = single_arm_pick_place(
            arx,
            pick_prompt=pick_prompt,
            place_prompt="",
            arm_side=PICK_ARM_SIDE,
            item_type="deepbox",
            debug=debug,
            depth_median_n=depth_median_n,
            release_after_pick=False,
        )
        if arm_used is None:
            print(
                "[pick_multi_box] "
                f"pick canceled at {color}, stop remaining picks"
            )
            return False

        print(f"[pick_multi_box] set special mode after pick: home {arm_used}")
        success, error_message = arx.set_special_mode(1, side=arm_used)
        if not success and error_message:
            print(
                f"set_special_mode(1, side={arm_used!r}) returned: {error_message}"
            )
    return True


def multi_box(
    arx: ARXRobotEnv,
    prompt: str,
    debug: bool = True,
    depth_median_n: int = 10,
    start_position: PositionName = "start",
) -> None:
    steps = parse_prompt_steps(prompt)
    current_position: PositionName = start_position
    start_height = COLOR_TO_LIFT_HEIGHT.get(start_position, START_LIFT_HEIGHT)
    print(
        "[pick_multi_box] "
        f"plan: {_format_plan(steps)}"
    )
    print(
        "[pick_multi_box] "
        f"start at {current_position}, lift={start_height:.1f}, "
        "green=left, yellow=center, blue=right, start=in front of yellow"
    )
    arx.step_lift(start_height)

    stop_requested = False
    for color, count in steps:
        print(f"[pick_multi_box] target {color}, count={count}")
        current_position = move_to_position(
            arx,
            current_position=current_position,
            target_position=color,
            settle_s=SETTLE_S,
        )
        if not _pick_color_boxes(
            arx,
            color=color,
            count=count,
            debug=debug,
            depth_median_n=depth_median_n,
        ):
            stop_requested = True
            break

    print(f"[pick_multi_box] return to start from {current_position}")
    move_to_position(
        arx,
        current_position=current_position,
        target_position=start_position,
        settle_s=SETTLE_S,
        apply_lift=(start_position in COLOR_TO_LIFT_HEIGHT),
    )
    if stop_requested:
        print("[pick_multi_box] stopped early after user canceled one pick")


def main() -> None:
    from arx_ros2_env import ARXRobotEnv

    arx = ARXRobotEnv(
        duration_per_step=1.0 / 20.0,
        min_steps=20,
        max_v_xyz=0.15,
        max_a_xyz=0.2,
        max_v_rpy=1.0,
        max_a_rpy=1.0,
        camera_type="all",
        camera_view=("camera_l", "camera_h", "camera_r"),
        img_size=(640, 480),
    )
    try:
        arx.reset()
        # Example prompt: "1 blue 2 yellow 1 green"
        # execution follows the prompt order
        multi_box(
            arx,
            prompt="2 blue 5 yellow 1 green",
            debug=True,
            depth_median_n=10,
            start_position="start",
        )
    finally:
        arx.close()


if __name__ == "__main__":
    main()
