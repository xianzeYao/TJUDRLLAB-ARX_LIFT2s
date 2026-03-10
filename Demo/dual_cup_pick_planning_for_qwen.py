from __future__ import annotations

import sys
import textwrap
import time
from pathlib import Path
from typing import Dict, List, Literal, Optional, Tuple

import cv2
import numpy as np

from arx_pointing import predict_multi_points_from_rgb, predict_point_from_rgb
from demo_utils import draw_text_lines
from motion_pick_place_cup import (
    build_pick_cup_sequence,
    build_place_cup_sequence,
)
from point2pos_utils import (
    get_aligned_frames,
    pixel_to_ref_point_safe,
)

sys.path.append("../ARX_Realenv/ROS2")  # noqa
from arx_ros2_env import ARXRobotEnv  # noqa

QWEN_BASE_URL = "http://172.28.102.11:22014/v1"
QWEN_MODEL_NAME = "Qwen3-VL-8B-Instruct"

COASTER_PROMPTS = [
    "the center of coaster (the left one near cups most)",
    "the center of coaster (the right one near cups most)",
    "the center of coaster (the leftmost one)",
    "the center of coaster (the rightmost one)",
]


def _build_planning_pick_prompt(goal_cup: str) -> str:
    return f"""Given an RGB image, output the minimal sequence of cup pick actions required to finally pick the {goal_cup}.

Rules:
- A cup can only be picked if no other cup is placed on top of it.
- A cup that is partially or fully occluded by another cup is NOT pickable.
- If the goal cup is not immediately pickable, you must first pick the cups that block it.
- The order of pick actions is the actual execution order.

For each pick action, provide one valid 2D pick point on the middle center of the cup.

Output ONLY a JSON array in execution order:
[
  {{
    "point_2d": [x, y]
  }}
]"""


def _build_place_prompt(
    steps_n: int,
    no_last_place: bool,
) -> Tuple[str, List[str]]:
    place_count = steps_n - (1 if no_last_place else 0)
    lines = [COASTER_PROMPTS[i] for i in range(place_count)]
    prompt = (
        'Format: [{"point_2d": [x, y]}, ...]. Return only JSON.\n'
        + "\n".join(f"Point out the {line}" for line in lines)
    )
    return prompt, lines


def _decode_points(points: List[Tuple[float, float]]) -> List[Tuple[int, int]]:
    return [(int(round(u)), int(round(v))) for (u, v) in points]


def _predict_qwen_points(
    color: np.ndarray,
    prompt: str,
) -> Tuple[List[Tuple[int, int]], Optional[str]]:
    points, raw = predict_multi_points_from_rgb(
        color,
        text_prompt="",
        all_prompt=prompt,
        base_url=QWEN_BASE_URL,
        model_name=QWEN_MODEL_NAME,
        assume_bgr=True,
        return_raw=True,
        temperature=0.7,
    )
    return _decode_points(points), raw


def _predict_qwen_single_point(
    color: np.ndarray,
    prompt: str,
) -> Tuple[int, int]:
    u, v = predict_point_from_rgb(
        color,
        text_prompt=prompt,
        base_url=QWEN_BASE_URL,
        model_name=QWEN_MODEL_NAME,
        assume_bgr=True,
        temperature=0.7,
    )
    return int(round(u)), int(round(v))


def _arm_for_step(step_idx: int, first_side: Literal["left", "right"]) -> str:
    if first_side not in ("left", "right"):
        raise ValueError(
            f"first_side must be 'left' or 'right', got {first_side!r}"
        )
    if first_side == "left":
        return "left" if step_idx % 2 == 0 else "right"
    return "right" if step_idx % 2 == 0 else "left"


def _run_parallel_sequences(
    arx: ARXRobotEnv,
    left_seq: Optional[List[dict]],
    right_seq: Optional[List[dict]],
) -> None:
    left_seq = left_seq or []
    right_seq = right_seq or []
    max_len = max(len(left_seq), len(right_seq))
    for i in range(max_len):
        act = {}
        if i < len(left_seq):
            act.update(left_seq[i])
        if i < len(right_seq):
            act.update(right_seq[i])
        arx.step(act)


def _arm_home_action(arm: str, open_gripper: bool = True) -> Dict[str, np.ndarray]:
    gripper = -3.4 if open_gripper else 0.0
    active = np.array([0, 0, 0, 0, 0, 0, gripper], dtype=np.float32)
    return {"left": active} if arm == "left" else {"right": active}


def _build_points_only_vis(
    color: np.ndarray,
    pick_px: List[Tuple[int, int]],
    place_px: List[Optional[Tuple[int, int]]],
) -> np.ndarray:
    disp = color.copy()
    for i, p in enumerate(pick_px):
        cv2.circle(disp, p, 3, (0, 0, 255), -1)
        draw_text_lines(disp, [f"P{i+1}"], origin=(p[0] + 6, p[1] - 6))
    for i, p in enumerate(place_px):
        if p is None:
            continue
        cv2.circle(disp, p, 3, (255, 0, 0), -1)
        draw_text_lines(disp, [f"C{i+1}"], origin=(p[0] + 6, p[1] - 6))
    return disp


def _save_points_vis(vis_img: np.ndarray, save_path: str) -> None:
    out = Path(save_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(out), vis_img):
        raise RuntimeError(f"failed to save image: {out}")
    print(f"预测点位图已保存: {out}")


def dual_cup_pick_planning_for_qwen(
    arx: ARXRobotEnv,
    goal: str = "red cup",
    first_side: Literal["left", "right"] = "left",
    debug_raw: bool = True,
    depth_median_n: int = 10,
    no_last_place: bool = False,
    single_test: bool = False,
    dir: Optional[str] = None,
) -> None:
    try:
        arx.step_lift(13.0)
        time.sleep(1.0)

        win = "dual_cup_pick_planning_for_qwen"
        if debug_raw:
            cv2.namedWindow(win, cv2.WINDOW_NORMAL)

        planning_prompt = _build_planning_pick_prompt(goal)

        while True:
            color, depth = get_aligned_frames(
                arx, depth_median_n=depth_median_n)
            if color is None or depth is None:
                continue

            pick_px, pick_raw = _predict_qwen_points(color, planning_prompt)
            if not pick_px:
                print("Qwen 未返回有效 pick 点，自动重试。")
                continue

            if len(pick_px) > len(COASTER_PROMPTS):
                print(
                    f"pick 点过多({len(pick_px)}>{len(COASTER_PROMPTS)}), 只保留前 {len(COASTER_PROMPTS)} 个")
                pick_px = pick_px[: len(COASTER_PROMPTS)]

            steps_n = len(pick_px)
            place_prompt, place_prompt_lines = _build_place_prompt(
                steps_n, no_last_place
            )

            if single_test:
                place_only_px: List[Tuple[int, int]] = []
                for prompt in place_prompt_lines:
                    try:
                        place_only_px.append(
                            _predict_qwen_single_point(
                                color, f"Point out the {prompt}")
                        )
                    except RuntimeError as exc:
                        print(f"单点预测 place 失败，自动重试：{exc}")
                        place_only_px = []
                        break
                if len(place_only_px) != len(place_prompt_lines):
                    continue
                place_raw = None
            else:
                place_only_px, place_raw = _predict_qwen_points(
                    color, place_prompt
                )
                if len(place_only_px) < len(place_prompt_lines):
                    print(
                        f"place 点数不足({len(place_only_px)}/{len(place_prompt_lines)}), 自动重试")
                    continue
                place_only_px = place_only_px[: len(place_prompt_lines)]

            pick_px[0] = (pick_px[0][0], pick_px[0][1] - 15)

            place_px: List[Optional[Tuple[int, int]]] = []
            place_idx = 0
            for i in range(steps_n):
                if no_last_place and i == steps_n - 1:
                    place_px.append(None)
                    continue
                place_px.append(place_only_px[place_idx])
                place_idx += 1

            disp_save = _build_points_only_vis(color, pick_px, place_px)
            if dir:
                try:
                    _save_points_vis(disp_save, dir)
                except Exception as exc:
                    print(f"保存预测点位图失败: {exc}")

            if debug_raw:
                disp_show = disp_save.copy()
                prompt_lines = textwrap.wrap(planning_prompt, width=60)
                info_lines = [
                    f"Steps: {steps_n} | no_last_place={no_last_place}",
                    "Press 'r' to re-predict, 'e' to execute, 'q' to quit",
                ]
                draw_text_lines(
                    disp_show,
                    ["Planning Prompt:"] + prompt_lines + info_lines,
                    origin=(10, 25),
                    line_height=20,
                    color=(0, 0, 255),
                    scale=0.5,
                    thickness=2,
                )
                cv2.imshow(win, disp_show)
                print("Pick 原始输出:")
                print(pick_raw if pick_raw else "<empty>")
                if place_raw is not None:
                    print("Place 原始输出:")
                    print(place_raw if place_raw else "<empty>")
                print("按 'r' 重预测，按 'e' 执行，按 'q' 退出")
                key = cv2.waitKey(0)
                if key == ord("r"):
                    continue
                if key == ord("q"):
                    break
                if key != ord("e"):
                    continue

            pick_refs: List[np.ndarray] = []
            place_refs: List[Optional[np.ndarray]] = []

            for i, p in enumerate(pick_px):
                arm = _arm_for_step(i, first_side=first_side)
                pick_ref = pixel_to_ref_point_safe(
                    p, depth, robot_part=arm
                )
                if pick_ref is None:
                    print(f"预测像素 {p} 深度无效或像素越界，自动刷新")
                    pick_refs = []
                    break
                pick_refs.append(pick_ref)
            if len(pick_refs) != len(pick_px):
                continue

            for i, p in enumerate(place_px):
                if p is None:
                    place_refs.append(None)
                    continue
                arm = _arm_for_step(i, first_side=first_side)
                place_ref = pixel_to_ref_point_safe(
                    p, depth, robot_part=arm
                )
                if place_ref is None:
                    print(f"预测像素 {p} 深度无效或像素越界，自动刷新")
                    place_refs = []
                    break
                place_refs.append(place_ref)
            if len(place_refs) != len(place_px):
                continue

            first_arm = _arm_for_step(0, first_side=first_side)
            first_pick = build_pick_cup_sequence(
                pick_refs[0], arm=first_arm)
            if first_arm == "left":
                _run_parallel_sequences(arx, first_pick, None)
            else:
                _run_parallel_sequences(arx, None, first_pick)
            time.sleep(1.0)

            last_place_idx = steps_n - 2 if no_last_place else steps_n - 1
            home_place_idx = last_place_idx if no_last_place else last_place_idx - 1
            for i in range(steps_n - 1):
                cur_arm = _arm_for_step(i, first_side=first_side)
                next_arm = _arm_for_step(i + 1, first_side=first_side)
                cur_place = (
                    build_place_cup_sequence(place_refs[i], arm=cur_arm)
                    if place_refs[i] is not None
                    else []
                )
                if i == home_place_idx:
                    cur_place = list(cur_place) + [
                        _arm_home_action(cur_arm, open_gripper=False)
                    ]
                next_pick = build_pick_cup_sequence(
                    pick_refs[i + 1], arm=next_arm)

                left_seq = cur_place if cur_arm == "left" else (
                    next_pick if next_arm == "left" else []
                )
                right_seq = cur_place if cur_arm == "right" else (
                    next_pick if next_arm == "right" else []
                )
                _run_parallel_sequences(arx, left_seq, right_seq)
                time.sleep(1.0)

            if not no_last_place:
                last_arm = _arm_for_step(steps_n - 1, first_side=first_side)
                last_place = build_place_cup_sequence(
                    place_refs[steps_n - 1], arm=last_arm
                )
                last_place = list(last_place) + [
                    _arm_home_action(last_arm, open_gripper=False)
                ]
                if last_arm == "left":
                    _run_parallel_sequences(arx, last_place, None)
                else:
                    _run_parallel_sequences(arx, None, last_place)
                time.sleep(1.0)

            break

    finally:
        cv2.destroyAllWindows()


def main() -> None:
    arx = ARXRobotEnv(
        duration_per_step=1.0 / 40.0,
        min_steps=30,
        max_v_xyz=0.15,
        max_a_xyz=0.1,
        max_v_rpy=0.5,
        max_a_rpy=0.6,
        camera_type="all",
        camera_view=("camera_h",),
        img_size=(640, 480),
    )
    try:
        arx.reset()
        dual_cup_pick_planning_for_qwen(
            arx,
            goal="red cup",
            no_last_place=False,
            single_test=False,
            depth_median_n=5,
        )
    finally:
        arx.close()


if __name__ == "__main__":
    main()
