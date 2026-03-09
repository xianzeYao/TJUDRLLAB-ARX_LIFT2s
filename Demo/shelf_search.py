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
    def __init__(self, remaining_move_duration: float):
        super().__init__("search aborted by user")
        self.remaining_move_duration = remaining_move_duration


DEFAULT_CHECK_INTERVAL = 0.2


@dataclass
class TargetQueryResult:
    color: np.ndarray
    bool_result: bool
    point: Optional[tuple[int, int]]
    bounds: tuple[int, int, int, int]
    result: bool


@dataclass
class ShelfSearchResult:
    success: bool
    remaining_move_duration: float
    layer_index: int
    arm_used: Optional[str] = None
    aborted: bool = False


@dataclass
class ScanDirectionResult:
    success: bool
    remaining_move_duration: float
    arm_used: Optional[str] = None


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
            assume_bgr=False,
            temperature=0.0,
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
        try:
            points, _ = predict_multi_points_from_rgb(
                image=color,
                text_prompt="",
                all_prompt=point_prompt,
                assume_bgr=False,
                temperature=0.0,
                return_raw=True,
            )
        except Exception as exc:
            raise RuntimeError(f"ER1.5 call failed: {exc}") from exc
        if points:
            u, v = points[0]
            point = (int(round(u)), int(round(v)))
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
) -> tuple[bool, Optional[str]]:
    pick_ref, place_ref, arm_used = single_arm_pick_place(
        arx=arx,
        pick_prompt=object_desc,
        place_prompt="",
        arm_side="fit",
        item_type="cup",
        debug=debug_pick_place,
        depth_median_n=depth_median_n,
    )
    if pick_ref is None and place_ref is None and arm_used is None:
        print("pick canceled, continue shelf search")
        return False, None
    return True, arm_used


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
) -> tuple[bool, Optional[str]]:
    arx.step_base(0.0, 0.0, 0.0)
    if debug_raw:
        key = _show_debug_result(
            query=query,
            center_region_ratio=center_region_ratio,
            elapsed_move=elapsed_move,
            max_move_duration=max_move_duration,
        )
        if key == ord("r"):
            return False, None
        if key == ord("q"):
            if elapsed_move is None or max_move_duration is None:
                remaining_move_duration = 0.0
            else:
                remaining_move_duration = max(
                    max_move_duration - elapsed_move, 0.0)
            raise UserAbortSearch(remaining_move_duration)
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
) -> ScanDirectionResult:
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
                found, arm_used = _handle_found_target(
                    arx=arx,
                    object_prompt=object_prompt,
                    query=query,
                    debug_raw=debug_raw,
                    debug_pick_place=debug_pick_place,
                    depth_median_n=depth_median_n,
                    center_region_ratio=center_region_ratio,
                    elapsed_move=total_elapsed,
                    max_move_duration=max_move_duration,
                )
                if found:
                    return ScanDirectionResult(
                        success=True,
                        remaining_move_duration=max(
                            max_move_duration - total_elapsed, 0.0),
                        arm_used=arm_used,
                    )
                moved_duration = total_elapsed
                break
            sleep_time = check_interval - (time.time() - check_start)
            if sleep_time > 0:
                time.sleep(sleep_time)
        else:
            moved_duration = max_move_duration

        arx.step_base(0.0, 0.0, 0.0)

    return ScanDirectionResult(
        success=False,
        remaining_move_duration=0.0,
        arm_used=None,
    )


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
) -> ShelfSearchResult:
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
    start_height = 13.0

    try:
        arx.step_lift(start_height)
        current_height = start_height
        layer_index = 1
        obj_bool_prompt = target_bool_prompt(object_prompt)
        obj_point_prompt = target_point_prompt(object_prompt)

        query = query_target(
            arx=arx,
            bool_prompt=obj_bool_prompt,
            point_prompt=obj_point_prompt,
            center_region_ratio=center_region_ratio,
        )
        if query.result:
            found, arm_used = _handle_found_target(
                arx=arx,
                object_prompt=object_prompt,
                query=query,
                debug_raw=debug_raw,
                debug_pick_place=debug_pick_place,
                depth_median_n=depth_median_n,
                center_region_ratio=center_region_ratio,
                elapsed_move=0.0,
                max_move_duration=max_move_duration,
            )
        else:
            found, arm_used = False, None
        if found:
            return ShelfSearchResult(
                success=True,
                remaining_move_duration=max_move_duration,
                layer_index=layer_index,
                arm_used=arm_used,
            )

        while True:
            # Snake scan: odd layer left->right (+v), even layer right->left (-v).
            curr_vy = v if (layer_index % 2 == 1) else -v
            scan_result = _scan_direction(
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
            if scan_result.success:
                return ShelfSearchResult(
                    success=True,
                    remaining_move_duration=scan_result.remaining_move_duration,
                    layer_index=layer_index,
                    arm_used=scan_result.arm_used,
                )

            if layer_index >= max_layer:
                print("reached max_layer limit, stopping search")
                return ShelfSearchResult(
                    success=False,
                    remaining_move_duration=0.0,
                    layer_index=layer_index,
                )

            current_height -= drop_height
            arx.step_lift(current_height)
            layer_index += 1

            query = query_target(
                arx=arx,
                bool_prompt=obj_bool_prompt,
                point_prompt=obj_point_prompt,
                center_region_ratio=center_region_ratio,
            )
            if query.result:
                found, arm_used = _handle_found_target(
                    arx=arx,
                    object_prompt=object_prompt,
                    query=query,
                    debug_raw=debug_raw,
                    debug_pick_place=debug_pick_place,
                    depth_median_n=depth_median_n,
                    center_region_ratio=center_region_ratio,
                    elapsed_move=0.0,
                    max_move_duration=max_move_duration,
                )
            else:
                found, arm_used = False, None
            if found:
                return ShelfSearchResult(
                    success=True,
                    remaining_move_duration=max_move_duration,
                    layer_index=layer_index,
                    arm_used=arm_used,
                )
    except UserAbortSearch as exc:
        print("search aborted by user")
        return ShelfSearchResult(
            success=False,
            remaining_move_duration=exc.remaining_move_duration,
            layer_index=layer_index,
            aborted=True,
        )
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
        v = 0.6
        max_move_duration = 8
        result = search_shelf(
            arx=arx,
            object_prompt="a yellow glue",
            v=v,
            check_interval=0.2,
            max_move_duration=max_move_duration,
            drop_height=13.0,
            max_layer=2,
            center_region_ratio=0.3,
            debug_raw=False,
            debug_pick_place=False,
            depth_median_n=5,
        )
        # 回归search原位
        continue_duration = max_move_duration - \
            result.remaining_move_duration if result.layer_index % 2 == 1 else result.remaining_move_duration
        step_base_duration(arx, 0.0, -v, 0.0,
                           duration=continue_duration)
        # 顺时针旋转180度，直走，再逆时针旋转90度

        step_base_duration(arx, 0.0, 0.0, -0.5, duration=20.6)
        step_base_duration(arx, 0.6, 0.0, 0.0, duration=20)
        arx.step_lift(14.0)
        step_base_duration(arx, 0.0, 0.0, 0.5, duration=10.3)
        # 抬高一点靠近桌子
        step_base_duration(arx, 0.6, 0.0, 0.0, duration=0.8)
        single_arm_pick_place(
            arx=arx,
            pick_prompt="",
            place_prompt="the center part of square plate",
            arm_side=result.arm_used,
            item_type="cup",
            debug=True,
            depth_median_n=5,
        )
    finally:
        arx.close()


if __name__ == "__main__":
    main()
