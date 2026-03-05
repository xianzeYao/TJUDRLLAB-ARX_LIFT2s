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


def _wrap_text(text: str, width: int = 56) -> list[str]:
    words = text.split()
    if not words:
        return [""]
    lines: list[str] = []
    cur = words[0]
    for w in words[1:]:
        if len(cur) + 1 + len(w) <= width:
            cur += " " + w
        else:
            lines.append(cur)
            cur = w
    lines.append(cur)
    return lines


def _show_debug_result(
    color: np.ndarray,
    title: str,
    prompt: str,
    result: bool,
) -> int:
    vis = color.copy()
    lines = [f"Type: {title}", f"Result: {result}",
             "Key: r=refresh, q=quit, others=accept", "Prompt:"]
    lines.extend(_wrap_text(prompt))
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


def query_bool_with_debug(
    arx: ARXRobotEnv,
    prompt: str,
    debug_title: str,
    debug_scan: bool,
) -> bool:
    while True:
        color = get_color_frame(arx)
        result = ask_bool(color, prompt)
        if not debug_scan:
            return result
        key = _show_debug_result(
            color=color,
            title=debug_title,
            prompt=prompt,
            result=result,
        )
        if key == ord("r"):
            continue
        if key == ord("q"):
            raise UserAbortSearch("search aborted by user in debug window")
        return result


def target_prompt(object_desc: str) -> str:
    safe_desc = object_desc.replace('"', '\\"')
    return (
        f'Is there an object matching this description: "{safe_desc}"? '
        "Output only True or False."
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
        go_home=True,
        depth_median_n=10,
    )
    if pick_ref is None and place_ref is None:
        print("pick canceled, continue shelf search")
        return False
    return True


def _check_target(arx: ARXRobotEnv, prompt: str, debug_scan: bool) -> bool:
    return query_bool_with_debug(
        arx=arx,
        prompt=prompt,
        debug_title="Target check (layer)",
        debug_scan=debug_scan,
    )


def _scan_direction(
    arx: ARXRobotEnv,
    object_prompt: str,
    vy: float,
    obj_prompt: str,
    debug_scan: bool,
) -> bool:
    vx, vz = 0.0, 0.0
    move_duration = 1.0

    for _ in range(FIXED_STEPS_PER_LAYER):
        arx.step_base(vx=vx, vy=vy, vz=vz, duration=move_duration)

        if query_bool_with_debug(
            arx=arx,
            prompt=obj_prompt,
            debug_title="Target check after move",
            debug_scan=debug_scan,
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
    reset_robot: bool = True,
    close_robot: bool = False,
    debug_scan: bool = False,
) -> bool:
    if max_layer <= 0:
        raise ValueError("max_layer must be > 0")
    if drop_height <= 0:
        raise ValueError("drop_height must be > 0")
    start_height = 20.0

    try:
        if reset_robot:
            arx.reset()
        arx.step_lift(start_height)
        current_height = start_height
        layer_index = 1
        obj_prompt = target_prompt(object_prompt)

        if _check_target(arx, obj_prompt, debug_scan=debug_scan):
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
                obj_prompt=obj_prompt,
                debug_scan=debug_scan,
            )
            if layer_hit:
                return True

            if layer_index >= max_layer:
                print("reached max_layer limit, stopping search")
                return False

            current_height -= drop_height
            arx.step_lift(current_height)
            layer_index += 1

            if _check_target(arx, obj_prompt, debug_scan=debug_scan):
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
        reset_robot=True,
        close_robot=True,
        debug_scan=True,
    )


if __name__ == "__main__":
    main()
# TODO 现在把check 有无和打点的逻辑合并一下，如果check有则进入一次打点，调用。然后对于现在的例子(640,480)的图，然后取一个长方形区域，这个长方形中心就在（320，240）然后加一个参数，比如60%，然后相当于得到四条边界。如果check有，且打的点在这个长方形里面，则显示绿色的，即原来的那个逻辑
# TODO 现在check就利用打点，如果没有点，就直接返回False，如果有点了，再看这个点在不在中心区域（然后对于现在的例子(640,480)的图，然后取一个长方形区域，这个长方形中心就在（320，240）然后加一个参数，比如60%，然后相当于得到四条边界），如果在中心区域才返回True，否则返回False，并且显示红色。

# TODO 扫到有就停着左右动