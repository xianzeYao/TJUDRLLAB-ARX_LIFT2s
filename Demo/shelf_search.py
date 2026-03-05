from __future__ import annotations

import re
import sys
import time
from pathlib import Path
from typing import Optional

import numpy as np

from arx_pointing import predict_multi_points_from_rgb

ROOT_DIR = Path(__file__).resolve().parent.parent
ROS2_DIR = ROOT_DIR / "ARX_Realenv" / "ROS2"
if str(ROS2_DIR) not in sys.path:
    sys.path.append(str(ROS2_DIR))

from arx_ros2_env import ARXRobotEnv  # noqa: E402


RIGHT_EDGE_PROMPT = (
    "After moving right, is there still shelf space on the right to continue searching? Output only True or False."
)
LEFT_EDGE_PROMPT = (
    "After moving left, is there still shelf space on the left to continue searching? Output only True or False."
)


def get_color_frame(arx: ARXRobotEnv) -> np.ndarray:
    target_size = (640, 480)
    camera_key = "camera_h_color"
    while True:
        frames = arx.node.get_camera(target_size=target_size, return_status=False)
        color = frames.get(camera_key)
        if color is not None:
            return color
        time.sleep(0.05)


def parse_bool(text: str) -> Optional[bool]:
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


def ask_bool(color: np.ndarray, prompt: str) -> bool:
    try:
        _, raw = predict_multi_points_from_rgb(
            image=color,
            text_prompt="",
            all_prompt=prompt,
            base_url="http://172.28.102.11:22002/v1",
            model_name="Embodied-R1.5-SFT-0128",
            api_key="EMPTY",
            assume_bgr=False,
            temperature=0.0,
            max_tokens=128,
            return_raw=True,
        )
    except Exception as exc:
        raise RuntimeError(f"ER1.5 call failed: {exc}") from exc

    parsed = parse_bool(raw if raw is not None else "")
    if parsed is None:
        raise RuntimeError(f"ER1.5 bool parsing failed. Raw output: {raw!r}")
    return parsed


def target_prompt(object_desc: str) -> str:
    safe_desc = object_desc.replace('"', '\\"')
    return (
        f'Is there an object matching this description: "{safe_desc}"? '
        "Output only True or False."
    )


def on_target_found(object_desc: str) -> None:
    raise NotImplementedError(
        f"on_target_found is not implemented yet for target: {object_desc}"
    )


def _check_target(arx: ARXRobotEnv, prompt: str) -> bool:
    color = get_color_frame(arx)
    return ask_bool(color, prompt)


def _scan_direction(
    arx: ARXRobotEnv,
    object_prompt: str,
    vy: float,
    boundary_prompt: str,
    obj_prompt: str,
) -> bool:
    vx, vz = 0.0, 0.0
    move_duration = 1.0
    reached_end = False

    while not reached_end:
        arx.step_base(vx=vx, vy=vy, vz=vz, duration=move_duration)

        color = get_color_frame(arx)
        if ask_bool(color, obj_prompt):
            print("True")
            on_target_found(object_prompt)
            return True

        has_space = ask_bool(color, boundary_prompt)
        if not has_space:
            reached_end = True

    return False


def search_shelf(
    arx: ARXRobotEnv,
    object_prompt: str,
    v: float,
    drop_height: float,
    max_layer: int,
    reset_robot: bool = True,
    close_robot: bool = False,
) -> bool:
    if max_layer <= 0:
        raise ValueError("max_layer must be > 0")
    if drop_height <= 0:
        raise ValueError("drop_height must be > 0")
    start_height = 18.0

    try:
        if reset_robot:
            arx.reset()
        arx.step_lift(start_height)
        current_height = start_height
        layer_index = 1
        obj_prompt = target_prompt(object_prompt)

        if _check_target(arx, obj_prompt):
            print("True")
            on_target_found(object_prompt)
            return True

        while True:
            right_hit = _scan_direction(
                arx,
                object_prompt,
                vy=-v,
                boundary_prompt=RIGHT_EDGE_PROMPT,
                obj_prompt=obj_prompt,
            )
            if right_hit:
                return True

            left_hit = _scan_direction(
                arx,
                object_prompt,
                vy=v,
                boundary_prompt=LEFT_EDGE_PROMPT,
                obj_prompt=obj_prompt,
            )
            if left_hit:
                return True

            if layer_index >= max_layer:
                print("False")
                return False

            current_height -= drop_height
            arx.step_lift(current_height)
            layer_index += 1

            if _check_target(arx, obj_prompt):
                print("True")
                on_target_found(object_prompt)
                return True
    finally:
        if close_robot:
            arx.close()


def main() -> None:
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
    search_shelf(
        arx=arx,
        object_prompt="a cube cup on shelf",
        v=0.18,
        drop_height=2.0,
        max_layer=6,
        reset_robot=True,
        close_robot=True,
    )


if __name__ == "__main__":
    main()
