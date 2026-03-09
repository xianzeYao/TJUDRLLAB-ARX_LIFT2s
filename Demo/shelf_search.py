from __future__ import annotations

from dataclasses import dataclass
import re
import sys
import time
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

from arx_pointing import predict_multi_points_from_rgb
from single_arm_pick_place import single_arm_pick_place
from demo_utils import step_base_duration
ROOT_DIR = Path(__file__).resolve().parent.parent
ROS2_DIR = ROOT_DIR / "ARX_Realenv" / "ROS2"
if str(ROS2_DIR) not in sys.path:
    sys.path.append(str(ROS2_DIR))

from arx_ros2_env import ARXRobotEnv  # noqa: E402


class UserAbortSearch(Exception):
    pass


DEFAULT_CHECK_INTERVAL = 0.2


@dataclass
class TargetQueryResult:
    color: np.ndarray
    bool_result: bool
    point: Optional[tuple[int, int]]
    bounds: tuple[int, int, int, int]
    result: bool


def get_color_frame(arx: ARXRobotEnv) -> np.ndarray:
    target_size = (640, 480)
    camera_key = "camera_h_color"
    while True:
        frames = arx.get_camera(
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
    query: TargetQueryResult,
    center_region_ratio: float,
    elapsed_move: Optional[float] = None,
    max_move_duration: Optional[float] = None,
) -> int:
    vis = query.color.copy()
    left, top, right, bottom = query.bounds
    box_color = (0, 255, 0) if query.result else (0, 0, 255)
    cv2.rectangle(vis, (left, top), (right, bottom), box_color, 2)
    if query.point is not None:
        cv2.circle(vis, query.point, 4, box_color, -1)

    point_text = "None" if query.point is None else f"({query.point[0]}, {query.point[1]})"
    lines = [
        f"Check: {query.bool_result}",
        f"Result: {query.result}",
        f"Point: {point_text}",
        f"Center box: {center_region_ratio:.0%}",
    ]
    if elapsed_move is not None and max_move_duration is not None:
        lines.append(f"Move time: {elapsed_move:.2f}/{max_move_duration:.2f}s")
    y = 28
    for line in lines:
        cv2.putText(
            vis,
            line,
            (10, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            box_color,
            2,
        )
        y += 24
    win = "shelf_search_debug"
    cv2.namedWindow(win, cv2.WINDOW_NORMAL)
    cv2.imshow(win, vis)
    return cv2.waitKey(0) & 0xFF


def _center_region_bounds(
    color: np.ndarray, ratio: float
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


def query_target(
    arx: ARXRobotEnv,
    bool_prompt: str,
    point_prompt: str,
    center_region_ratio: float,
) -> TargetQueryResult:
    start = time.time()
    color = get_color_frame(arx)
    bool_result = ask_bool(color, bool_prompt)
    point = None
    bounds = _center_region_bounds(color, ratio=center_region_ratio)
    if bool_result:
        point, _ = ask_point(color, point_prompt)
    result = bool_result and _point_in_bounds(point, bounds)
    elapsed = time.time() - start
    print(f"query_target elapsed: {elapsed:.3f}s")
    return TargetQueryResult(
        color=color,
        bool_result=bool_result,
        point=point,
        bounds=bounds,
        result=result,
    )


def target_bool_prompt(object_desc: str) -> str:
    safe_desc = object_desc.replace('"', '\\"')
    return (
        f'Is there an object matching this description: "{safe_desc}"? '
        "Output only True or False."
    )


def target_point_prompt(object_desc: str) -> str:
    safe_desc = object_desc.replace('"', '\\"')
    return (
        "Provide one or more points coordinate of objects region this sentence describes: "
            f"{safe_desc}. "
            'The answer should be presented in JSON format as follows: [{"point_2d": [x, y]}].'
    )


def on_target_found(
    arx: ARXRobotEnv,
    object_desc: str,
    debug_pick_place: bool,
    depth_median_n: int,
) -> bool:
    pick_ref, place_ref = single_arm_pick_place(
        arx=arx,
        pick_prompt=object_desc,
        place_prompt="",
        arm_side="fit",
        item_type="cup",
        debug=debug_pick_place,
        depth_median_n=depth_median_n,
        release_after_pick=True,
    )
    if pick_ref is None and place_ref is None:
        print("pick canceled, continue shelf search")
        return False
    return True


def _check_target(
    arx: ARXRobotEnv,
    object_prompt: str,
    bool_prompt: str,
    point_prompt: str,
    debug_raw: bool,
    debug_pick_place: bool,
    depth_median_n: int,
    center_region_ratio: float,
) -> bool:
    query = query_target(
        arx=arx,
        bool_prompt=bool_prompt,
        point_prompt=point_prompt,
        center_region_ratio=center_region_ratio,
    )
    if not query.result:
        return False
    return _handle_found_target(
        arx=arx,
        object_prompt=object_prompt,
        query=query,
        debug_raw=debug_raw,
        debug_pick_place=debug_pick_place,
        depth_median_n=depth_median_n,
        center_region_ratio=center_region_ratio,
    )


def _handle_found_target(
    arx: ARXRobotEnv,
    object_prompt: str,
    query: TargetQueryResult,
    debug_raw: bool,
    debug_pick_place: bool,
    depth_median_n: int,
    center_region_ratio: float,
    elapsed_move: Optional[float] = None,
    max_move_duration: Optional[float] = None,
) -> bool:
    arx.step_base(0.0, 0.0, 0.0)
    if debug_raw:
        key = _show_debug_result(
            query=query,
            center_region_ratio=center_region_ratio,
            elapsed_move=elapsed_move,
            max_move_duration=max_move_duration,
        )
        if key == ord("r"):
            return False
        if key == ord("q"):
            raise UserAbortSearch("search aborted by user in debug window")
    return on_target_found(
        arx,
        object_prompt,
        debug_pick_place=debug_pick_place,
        depth_median_n=depth_median_n,
    )


def _scan_direction(
    arx: ARXRobotEnv,
    object_prompt: str,
    vy: float,
    obj_bool_prompt: str,
    obj_point_prompt: str,
    debug_raw: bool,
    debug_pick_place: bool,
    depth_median_n: int,
    center_region_ratio: float,
    max_move_duration: float,
    check_interval: float,
) -> bool:
    vx, vz = 0.0, 0.0
    moved_duration = 0.0

    while moved_duration < max_move_duration:
        remaining_duration = max_move_duration - moved_duration
        phase_start = time.time()
        arx.step_base(vx=vx, vy=vy, vz=vz)

        while time.time() - phase_start < remaining_duration:
            check_start = time.time()
            query = query_target(
                arx=arx,
                bool_prompt=obj_bool_prompt,
                point_prompt=obj_point_prompt,
                center_region_ratio=center_region_ratio,
            )
            elapsed = min(time.time() - phase_start, remaining_duration)
            total_elapsed = moved_duration + elapsed
            if query.result:
                print("True")
                if _handle_found_target(
                    arx=arx,
                    object_prompt=object_prompt,
                    query=query,
                    debug_raw=debug_raw,
                    debug_pick_place=debug_pick_place,
                    depth_median_n=depth_median_n,
                    center_region_ratio=center_region_ratio,
                    elapsed_move=total_elapsed,
                    max_move_duration=max_move_duration,
                ):
                    return True
                moved_duration = total_elapsed
                break
            sleep_time = check_interval - (time.time() - check_start)
            if sleep_time > 0:
                time.sleep(sleep_time)
        else:
            moved_duration = max_move_duration

        arx.step_base(0.0, 0.0, 0.0)

    return False


def search_shelf(
    arx: ARXRobotEnv,
    object_prompt: str,
    v: float,
    drop_height: float,
    max_layer: int,
    center_region_ratio: float = 0.6,
    max_move_duration: float = 6.0,
    check_interval: float = DEFAULT_CHECK_INTERVAL,
    debug_raw: bool = False,
    debug_pick_place: bool = False,
    depth_median_n: int = 10,
) -> bool:
    if max_layer <= 0:
        raise ValueError("max_layer must be > 0")
    if drop_height <= 0:
        raise ValueError("drop_height must be > 0")
    if not 0.0 < center_region_ratio <= 1.0:
        raise ValueError("center_region_ratio must be in (0, 1]")
    if max_move_duration <= 0:
        raise ValueError("max_move_duration must be > 0")
    if check_interval < 0:
        raise ValueError("check_interval must be >= 0")
    start_height = 20.0

    try:
        arx.step_lift(start_height)
        current_height = start_height
        layer_index = 1
        obj_bool_prompt = target_bool_prompt(object_prompt)
        obj_point_prompt = target_point_prompt(object_prompt)

        if _check_target(
            arx,
            object_prompt,
            obj_bool_prompt,
            obj_point_prompt,
            debug_raw=debug_raw,
            debug_pick_place=debug_pick_place,
            depth_median_n=depth_median_n,
            center_region_ratio=center_region_ratio,
        ):
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
                debug_raw=debug_raw,
                debug_pick_place=debug_pick_place,
                depth_median_n=depth_median_n,
                center_region_ratio=center_region_ratio,
                max_move_duration=max_move_duration,
                check_interval=check_interval,
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
                object_prompt,
                obj_bool_prompt,
                obj_point_prompt,
                debug_raw=debug_raw,
                debug_pick_place=debug_pick_place,
                depth_median_n=depth_median_n,
                center_region_ratio=center_region_ratio,
            ):
                return True
    except UserAbortSearch:
        print("search aborted by user")
        return False
    finally:
        cv2.destroyAllWindows()


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
    try:
        arx.reset()
        search_shelf(
            arx=arx,
            object_prompt="a red horse",
            v=0.5,
            check_interval=0.2,
            max_move_duration=5,
            drop_height=10.0,
            max_layer=3,
            center_region_ratio=0.3,
            debug_raw=True,
            debug_pick_place=True,
            depth_median_n=10,
        )
        # step_base_duration(arx, 0.0, 0.0, -0.5, duration=20.6)
        # step_base_duration(arx, 0.8, 0.0, 0.0, duration=12)
        # step_base_duration(arx, 0.0, 0.0, 0.5, duration=10.3)
    finally:
        arx.close()


if __name__ == "__main__":
    main()
