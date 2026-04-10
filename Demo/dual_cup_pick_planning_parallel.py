from __future__ import annotations

import sys
import time
import textwrap
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

from utils import (
    build_pick_cup_sequence,
    build_place_cup_sequence,
    do_replan,
    draw_text_lines,
    get_aligned_frames,
    pixel_to_ref_point_safe,
    predict_multi_points_from_rgb,
    predict_point_from_rgb,
)

CURRENT_DIR = Path(__file__).resolve().parent
ROOT_DIR = CURRENT_DIR.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.append(str(ROOT_DIR))
from ARX_Realenv.ROS2.arx_ros2_env import ARXRobotEnv  # noqa

COASTER_PROMPTS = [
    "the center of coaster (the left one near cups most)",
    "the center of coaster (the right one near cups most)",
    "the center of coaster (the leftmost one)",
    "the center of coaster (the rightmost one)",
]


def _arm_for_step(step_idx: int, first_side: Literal["left", "right"]) -> str:
    if first_side not in ("left", "right"):
        raise ValueError(
            f"first_side must be 'left' or 'right', got {first_side!r}"
        )
    if first_side == "left":
        return "left" if step_idx % 2 == 0 else "right"
    return "right" if step_idx % 2 == 0 else "left"


def _build_multi_prompt(
    plan_steps: List[str], no_last_place: bool
) -> Tuple[str, List[str]]:
    lines: List[str] = []
    for i, step in enumerate(plan_steps):
        lines.append(f"Point out the {step}")
        if not (no_last_place and i == len(plan_steps) - 1):
            lines.append(f"Point out the {COASTER_PROMPTS[i]}")
    prompt = (
        'Format: [{"point_2d": [x, y]}, ...]. Return only JSON.\n'
        + "\n".join(lines)
    )
    print(lines)
    return prompt, lines


def _decode_points(points: List[Tuple[float, float]]) -> List[Tuple[int, int]]:
    return [(int(round(u)), int(round(v))) for (u, v) in points]


def _extract_unique_cup_steps(
    plan_steps: List[str],
    goal: str,
    max_steps: int = 4,
) -> List[str]:
    unique_cups: List[str] = []
    seen = set()
    text = " ".join([*plan_steps, goal])
    for match in re.finditer(r"\b([A-Za-z]+)\s+cup\b", text, flags=re.IGNORECASE):
        cup = match.group(0).strip()
        key = cup.lower()
        if key in seen:
            continue
        seen.add(key)
        unique_cups.append(cup)

    return unique_cups[:max_steps]


def _build_single_prompts(
    plan_steps: List[str], no_last_place: bool
) -> List[str]:
    lines: List[str] = []
    for i, step in enumerate(plan_steps):
        lines.append(step)
        if not (no_last_place and i == len(plan_steps) - 1):
            lines.append(COASTER_PROMPTS[i])
    return lines


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
        arx.step_smooth_eef(act)


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


def _build_planning_prompt(goal_cup: str, prompt_idx: int = 0) -> str:
    prompts = [
        (
            f"Current Goal is: pick the {goal_cup}. "
            "I need to pick up the cups from top to the goal cup."
            "What is the picking plan steps to finish the goal?"
        ),
        (
            f"Current Goal is: pick the {goal_cup}. "
            f"Start from the toppest cup and trace downward to identify the blocking cups up to the {goal_cup}."
            "What cups are the blocking cups in the plan?"
        ),
    ]
    # prompts = [(f"""Given an RGB image, output the minimal sequence of cup pick actions required to finally pick the {goal_cup}
    #                     Rules:
    #                     - A cup can only be picked if no other cup is placed on top of it.
    #                     - A cup that is partially or fully occluded by another cup is NOT pickable.
    #                     - If the goal cup is not immediately pickable, you must first pick the cups that block it.
    #                     - The order of pick actions is the actual execution order.
    #                     For each pick action, provide the color of the cup like :red cup."""),
    #            (f"""Given an RGB image, output the minimal sequence of cup pick actions required to finally pick the {goal_cup}
    #                     Rules:
    #                     - A cup can only be picked if no other cup is placed on top of it.
    #                     - A cup that is partially or fully occluded by another cup is NOT pickable.
    #                     - If the goal cup is not immediately pickable, you must first pick the cups that block it.
    #                     - The order of pick actions is the actual execution order.
    #                     For each pick action, provide the color of cup.like :red cup.""")]
    return prompts[prompt_idx % len(prompts)]


def dual_arm_pick_planning_parallel(
    arx: ARXRobotEnv,
    goal: str = "red cup",
    first_side: Literal["left", "right"] = "left",
    debug_raw: bool = True,
    depth_median_n: int = 10,
    no_last_place: bool = False,
    single_test: bool = False,
    dir: Optional[str] = None,
):
    try:
        arx.step_lift(17.0)
        time.sleep(1.0)

        planned = True
        plan_steps: List[str] = []
        plan_cups: List[str] = []

        goal_cup = goal
        plan_attempt_idx = 0
        planning_prompt = _build_planning_prompt(goal_cup, plan_attempt_idx)
        confirm_win = "Planning Step"
        if debug_raw:
            cv2.namedWindow(confirm_win, cv2.WINDOW_NORMAL)

        while True:
            planning_prompt = _build_planning_prompt(
                goal_cup, plan_attempt_idx)
            frames = arx.get_camera(
                target_size=(640, 480), return_status=False)
            color = frames.get("camera_h_color")
            if color is None:
                continue
            current_plan, current_cups, current_raw_plan = do_replan(
                color, planning_prompt
            )
            if not debug_raw:
                if not current_plan:
                    continue
                print("规划原始输出:")
                print(current_raw_plan if current_raw_plan else "<empty>")
                plan_steps = _extract_unique_cup_steps(
                    current_plan, goal=goal_cup)
                plan_cups = plan_steps[:]
                if not plan_steps:
                    print("未提取到有效 cup 步骤，重新规划。")
                    plan_attempt_idx += 1
                    continue
                print("规划已自动确认，进入执行模式。")
                break

            vis_img = color.copy()
            prompt_lines = textwrap.wrap(planning_prompt, width=60)
            draw_text_lines(
                vis_img,
                ["Planning Prompt:"] + prompt_lines,
                origin=(10, 30),
                line_height=22,
                color=(0, 0, 255),
                scale=0.5,
                thickness=2,
            )
            print("规划原始输出:")
            print(current_raw_plan if current_raw_plan else "<empty>")
            print(f"生成的规划结果 ({len(current_plan)} 步):")
            if not current_plan:
                print("未生成有效步骤！")
            else:
                for i, step in enumerate(current_plan):
                    print(f"  {i+1}. {step}")
            cv2.imshow(confirm_win, vis_img)
            print("按'y' 确认, 'r' 重试, 'p' 更新目标, 'q' 退出")
            key = cv2.waitKey(0)
            if key == ord("y") and current_plan:
                plan_steps = _extract_unique_cup_steps(
                    current_plan, goal=goal_cup)
                plan_cups = plan_steps[:]
                if not plan_steps:
                    print("未提取到有效 cup 步骤，重新规划。")
                    plan_attempt_idx += 1
                    continue
                print("规划已确认，进入执行模式。")
                break
            if key == ord("r"):
                print("重新尝试规划...")
                plan_attempt_idx += 1
                continue
            if key == ord("p"):
                new_goal = input("输入新的需求 (留空保持当前): ").strip()
                if new_goal:
                    goal_cup = new_goal
                    plan_attempt_idx = 0
                    planning_prompt = _build_planning_prompt(
                        goal_cup, plan_attempt_idx)
                    print(f"新的 pick prompt 已设置为: {planning_prompt!r}")
                continue
            if key == ord("q"):
                print("退出程序。")
                planned = False
                break

        if planned:
            if debug_raw:
                cv2.destroyWindow(confirm_win)
                win = "dual_cup_pick_planning"
                cv2.namedWindow(win, cv2.WINDOW_NORMAL)

        while planned and plan_steps:
            arx.step_lift(13.0)
            time.sleep(1.0)
            color, depth = get_aligned_frames(
                arx, depth_median_n=depth_median_n)
            if color is None or depth is None:
                continue

            target_steps = plan_steps
            if plan_cups:
                if len(plan_cups) >= len(plan_steps):
                    target_steps = [plan_cups[i]
                                    for i in range(len(plan_steps))]
                else:
                    print(
                        f"plan_cups 数量不足({len(plan_cups)}/{len(plan_steps)}), 使用 plan_steps")

            needed_points = len(plan_steps) * 2 - (1 if no_last_place else 0)
            if single_test:
                prompt_lines = _build_single_prompts(
                    target_steps, no_last_place)
                if len(prompt_lines) != needed_points:
                    print(
                        f"prompt 数量异常({len(prompt_lines)}/{needed_points}), 按 r 重试")
                    continue
                pts: List[Tuple[int, int]] = []
                for prompt in prompt_lines:
                    try:
                        u, v = predict_point_from_rgb(
                            color,
                            text_prompt=prompt,
                            temperature=0.7,
                            assume_bgr=False,
                        )
                    except RuntimeError as exc:
                        print(f"单点预测失败，按 r 重试：{exc}")
                        pts = []
                        break
                    pts.append((int(round(u)), int(round(v))))
            else:
                prompt, prompt_lines = _build_multi_prompt(
                    target_steps, no_last_place)
                points = predict_multi_points_from_rgb(
                    color,
                    text_prompt="",
                    all_prompt=prompt,
                    temperature=0.7,
                    assume_bgr=False,
                )
                pts = _decode_points(points)

            if len(pts) < needed_points:
                print(f"点数不足({len(pts)}/{needed_points}), 按 r 重试")
                continue

            pts[0] = (pts[0][0], pts[0][1]-15)  # 微调第一个 pick 点位置

            # 按顺序映射：奇数行为 pick，偶数行为 place
            pick_px: List[Tuple[int, int]] = []
            place_px: List[Optional[Tuple[int, int]]] = []
            p_idx = 0
            for i in range(len(plan_steps)):
                if p_idx >= len(pts):
                    print(f"点数不足({len(pts)}/{needed_points}), 按 r 重试")
                    p_idx = -1
                    break
                pick_px.append(pts[p_idx])
                p_idx += 1
                if not (no_last_place and i == len(plan_steps) - 1):
                    if p_idx >= len(pts):
                        print(f"点数不足({len(pts)}/{needed_points}), 按 r 重试")
                        p_idx = -1
                        break
                    place_px.append(pts[p_idx])
                    p_idx += 1
                else:
                    place_px.append(None)

            if p_idx == -1:
                continue

            disp_save = _build_points_only_vis(color, pick_px, place_px)
            if dir:
                try:
                    _save_points_vis(disp_save, dir)
                except Exception as exc:
                    print(f"保存预测点位图失败: {exc}")

            # 展示图保留 prompt 与说明；保存图不叠左上角 prompt。
            if debug_raw:
                disp_show = disp_save.copy()
                prompt_lines_show = [f"{i+1}. {x}" for i,
                                     x in enumerate(prompt_lines)]
                info_lines = [
                    f"Steps: {len(plan_steps)} | no_last_place={no_last_place}",
                    "Press 'r' to re-predict, 'e' to execute, 'q' to quit",
                ]
                draw_text_lines(
                    disp_show,
                    ["Point Prompt:"] + prompt_lines_show + info_lines,
                    origin=(10, 25),
                    line_height=20,
                    color=(0, 0, 255),
                    scale=0.55,
                    thickness=2,
                )
                cv2.imshow(win, disp_show)
                print("按 'r' 重预测，按 'e' 执行，按 'q' 退出")

                key = cv2.waitKey(0)
                if key == ord("r"):
                    continue
                if key == ord("q"):
                    break
                if key != ord("e"):
                    continue

            # 执行动作：先左后右，place 时交替另一侧 pick
            pick_refs: List[np.ndarray] = []
            place_refs: List[Optional[np.ndarray]] = []
            for i, p in enumerate(pick_px):
                arm = _arm_for_step(i, first_side=first_side)
                pick_ref = pixel_to_ref_point_safe(
                    p,
                    depth,
                    robot_part=arm,
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
                    p,
                    depth,
                    robot_part=arm,
                )
                if place_ref is None:
                    print(f"预测像素 {p} 深度无效或像素越界，自动刷新")
                    place_refs = []
                    break
                place_refs.append(place_ref)

            if len(place_refs) != len(place_px):
                continue

            steps_n = len(plan_steps)
            if steps_n == 0:
                continue

            # 先执行第 0 步的 pick（单臂）
            first_arm = _arm_for_step(0, first_side=first_side)
            first_pick = build_pick_cup_sequence(
                pick_refs[0], arm=first_arm)
            if first_arm == "left":
                _run_parallel_sequences(arx, first_pick, None)
            else:
                _run_parallel_sequences(arx, None, first_pick)
            time.sleep(1.0)

            # 交替并行：当前步 place 与下一步 pick 同时执行
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
                    cur_place = list(cur_place) + \
                        [_arm_home_action(cur_arm, open_gripper=False)]
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

            # 最后一步是否放置
            if not no_last_place:
                last_arm = _arm_for_step(steps_n - 1, first_side=first_side)
                last_place = build_place_cup_sequence(
                    place_refs[steps_n - 1], arm=last_arm
                )
                last_place = list(last_place) + \
                    [_arm_home_action(last_arm, open_gripper=False)]
                if last_arm == "left":
                    _run_parallel_sequences(arx, last_place, None)
                else:
                    _run_parallel_sequences(arx, None, last_place)
                time.sleep(1.0)

            break

    finally:
        cv2.destroyAllWindows()


def main():
    arx = ARXRobotEnv(
        duration_per_step=1.0 / 40.0,
        min_steps=30,
        max_v_xyz=0.15, max_a_xyz=0.1,
        max_v_rpy=0.5, max_a_rpy=0.6,
        camera_type="all",
        camera_view=("camera_h",),
        img_size=(640, 480),
        # video=True,
        # video_fps=30.0,
        # dir="../Video4demo",
        # video_name="dual_cup_pick_planning_parallel_red",
    )
    try:
        arx.reset()
        dual_arm_pick_planning_parallel(
            arx,
            goal="purple cup",
            no_last_place=False,
            single_test=True,
            # dir="../Video4demo/dual_cup_pick_planning_parallel_red.png",
            depth_median_n=5,
        )
    finally:
        arx.close()


if __name__ == "__main__":
    main()
