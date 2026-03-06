from __future__ import annotations

import re
import sys
import time
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

from arx_pointing import predict_multi_points_from_rgb
from single_arm_pick_place import single_arm_pick_place

ROOT_DIR = Path(__file__).resolve().parent.parent
ROS2_DIR = ROOT_DIR / "ARX_Realenv" / "ROS2"
if str(ROS2_DIR) not in sys.path:
    sys.path.append(str(ROS2_DIR))

from arx_ros2_env import ARXRobotEnv  # noqa: E402


FIXED_STEPS_PER_LAYER = 3
DEFAULT_CENTER_REGION_RATIO = 0.6


class UserAbortSearch(Exception):
    pass


def get_color_frame(arx: ARXRobotEnv) -> np.ndarray:
    target_size = (640, 480)
    camera_key = "camera_h_color"
    while True:
        frames = arx.node.get_camera(
            target_size=target_size, return_status=False)
        color = frames.get(camera_key)
        if color is not None:
            return color
        time.sleep(0.05)


def ask_point(color: np.ndarray, prompt: str) -> tuple[Optional[tuple[int, int]], Optional[str]]:
    try:
        points, raw = predict_multi_points_from_rgb(
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

    if not points:
        return None, raw

    u, v = points[0]
    return (int(round(u)), int(round(v))), raw


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


def _show_debug_result(
    color: np.ndarray,
    result: bool,
    bool_result: bool,
    point: Optional[tuple[int, int]],
    bounds: tuple[int, int, int, int],
    center_region_ratio: float,
) -> int:
    vis = color.copy()
    left, top, right, bottom = bounds
    box_color = (0, 255, 0) if result else (0, 0, 255)
    cv2.rectangle(vis, (left, top), (right, bottom), box_color, 2)
    if point is not None:
        cv2.circle(vis, point, 4, box_color, -1)

    point_text = "None" if point is None else f"({point[0]}, {point[1]})"
    lines = [
        f"Check: {bool_result}",
        f"Result: {result}",
        f"Point: {point_text}",
        f"Center box: {center_region_ratio:.0%}",
    ]
    y = 28
    for line in lines:
        cv2.putText(
            vis,
            line,
            (10, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (0, 255, 0) if result else (0, 0, 255),
            2,
        )
        y += 24
    win = "shelf_search_debug"
    cv2.namedWindow(win, cv2.WINDOW_NORMAL)
    cv2.imshow(win, vis)
    return cv2.waitKey(0) & 0xFF


def _center_region_bounds(
    color: np.ndarray, ratio: float = DEFAULT_CENTER_REGION_RATIO
) -> tuple[int, int, int, int]:
    h, w = color.shape[:2]
    half_w = int(round(w * ratio / 2.0))
    half_h = int(round(h * ratio / 2.0))
    cx, cy = w // 2, h // 2
    left = max(0, cx - half_w)
    top = max(0, cy - half_h)
    right = min(w - 1, cx + half_w)
    bottom = min(h - 1, cy + half_h)
    return left, top, right, bottom


def _point_in_bounds(
    point: Optional[tuple[int, int]], bounds: tuple[int, int, int, int]
) -> bool:
    if point is None:
        return False
    u, v = point
    left, top, right, bottom = bounds
    return left <= u <= right and top <= v <= bottom


def query_target_with_debug(
    arx: ARXRobotEnv,
    bool_prompt: str,
    point_prompt: str,
    debug_scan: bool,
    center_region_ratio: float,
) -> bool:
    while True:
        color = get_color_frame(arx)
        bool_result = ask_bool(color, bool_prompt)
        point = None
        bounds = _center_region_bounds(color, ratio=center_region_ratio)
        if bool_result:
            point, _ = ask_point(color, point_prompt)
        result = bool_result and _point_in_bounds(point, bounds)
        if not debug_scan:
            return result
        key = _show_debug_result(
            color=color,
            result=result,
            bool_result=bool_result,
            point=point,
            bounds=bounds,
            center_region_ratio=center_region_ratio,
        )
        if key == ord("r"):
            continue
        if key == ord("q"):
            raise UserAbortSearch("search aborted by user in debug window")
        return result


def target_bool_prompt(object_desc: str) -> str:
    safe_desc = object_desc.replace('"', '\\"')
    return (
        f'Is there an object matching this description: "{safe_desc}"? '
        "Output only True or False."
    )


def target_point_prompt(object_desc: str) -> str:
    safe_desc = object_desc.replace('"', '\\"')
    return (
        "Provide exactly one point coordinate of the object region matching this description: "
        f'"{safe_desc}". '
        'If no such object exists, return []. Format: [{"point_2d": [x, y]}]. '
        "Return only JSON."
    )


def on_target_found(arx: ARXRobotEnv, object_desc: str) -> bool:
    pick_ref, place_ref = single_arm_pick_place(
        arx=arx,
        pick_prompt=object_desc,
        place_prompt="",
        arm="left",
        item_type="cup",
        reset_robot=False,
        close_robot=False,
        debug=True,
        go_home=False,
        depth_median_n=10,
    )
    if pick_ref is None and place_ref is None:
        print("pick canceled, continue shelf search")
        return False
    return True


def _check_target(
    arx: ARXRobotEnv,
    bool_prompt: str,
    point_prompt: str,
    debug_scan: bool,
    center_region_ratio: float,
) -> bool:
    return query_target_with_debug(
        arx=arx,
        bool_prompt=bool_prompt,
        point_prompt=point_prompt,
        debug_scan=debug_scan,
        center_region_ratio=center_region_ratio,
    )


def _scan_direction(
    arx: ARXRobotEnv,
    object_prompt: str,
    vy: float,
    obj_bool_prompt: str,
    obj_point_prompt: str,
    debug_scan: bool,
    center_region_ratio: float,
) -> bool:
    vx, vz = 0.0, 0.0
    move_duration = 1.0

    for _ in range(FIXED_STEPS_PER_LAYER):
        arx.step_base(vx=vx, vy=vy, vz=vz, duration=move_duration)

        if query_target_with_debug(
            arx=arx,
            bool_prompt=obj_bool_prompt,
            point_prompt=obj_point_prompt,
            debug_scan=debug_scan,
            center_region_ratio=center_region_ratio,
        ):
            print("True")
            if on_target_found(arx, object_prompt):
                return True

    return False


def search_shelf(
    arx: ARXRobotEnv,
    object_prompt: str,
    v: float,
    drop_height: float,
    max_layer: int,
    center_region_ratio: float = DEFAULT_CENTER_REGION_RATIO,
    reset_robot: bool = True,
    close_robot: bool = False,
    debug_scan: bool = False,
) -> bool:
    if max_layer <= 0:
        raise ValueError("max_layer must be > 0")
    if drop_height <= 0:
        raise ValueError("drop_height must be > 0")
    if not 0.0 < center_region_ratio <= 1.0:
        raise ValueError("center_region_ratio must be in (0, 1]")
    start_height = 20.0

    try:
        if reset_robot:
            arx.reset()
        arx.step_lift(start_height)
        current_height = start_height
        layer_index = 1
        obj_bool_prompt = target_bool_prompt(object_prompt)
        obj_point_prompt = target_point_prompt(object_prompt)

        if _check_target(
            arx,
            obj_bool_prompt,
            obj_point_prompt,
            debug_scan=debug_scan,
            center_region_ratio=center_region_ratio,
        ):
            print("True")
            if on_target_found(arx, object_prompt):
                return True

        while True:
            # Snake scan: odd layer left->right (+v), even layer right->left (-v).
            curr_vy = v if (layer_index % 2 == 1) else -v
            layer_hit = _scan_direction(
                arx,
                object_prompt,
                vy=curr_vy,
                obj_bool_prompt=obj_bool_prompt,
                obj_point_prompt=obj_point_prompt,
                debug_scan=debug_scan,
                center_region_ratio=center_region_ratio,
            )
            if layer_hit:
                return True

            if layer_index >= max_layer:
                print("reached max_layer limit, stopping search")
                return False

            current_height -= drop_height
            arx.step_lift(current_height)
            layer_index += 1

            if _check_target(
                arx,
                obj_bool_prompt,
                obj_point_prompt,
                debug_scan=debug_scan,
                center_region_ratio=center_region_ratio,
            ):
                print("True")
                if on_target_found(arx, object_prompt):
                    return True
    except UserAbortSearch:
        print("search aborted by user")
        return False
    finally:
        cv2.destroyAllWindows()
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
        object_prompt="a red horse",
        v=1.0,
        drop_height=10.0,
        max_layer=3,
        center_region_ratio=0.4,
        reset_robot=True,
        close_robot=True,
        debug_scan=True,
    )


if __name__ == "__main__":
    main()

# TODO 扫到有就停着左右动
