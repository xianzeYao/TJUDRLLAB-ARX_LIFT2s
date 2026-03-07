from __future__ import annotations

import sys
import time
import textwrap
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

from arx_pointing import predict_multi_points_from_rgb, predict_point_from_rgb
from demo_utils import do_replan, draw_text_lines
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

COASTER_PROMPTS = [
    "the center of coaster (the left one near cups most)",
    "the center of coaster (the right one near cups most)",
    "the center of coaster (the leftmost one)",
    "the coaster (the rightmost one)",
]


def _arm_for_step(step_idx: int) -> str:
    return "left" if step_idx % 2 == 0 else "right"


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


def dual_arm_pick_planning_parallel(
    arx: ARXRobotEnv,
    goal: str = "red cup",
    reset_robot: bool = True,
    close_robot: bool = True,
    debug_raw: bool = True,
    depth_median_n: int = 10,
    no_last_place: bool = False,
    single_test: bool = False,
    dir: Optional[str] = None,
):
    try:
        if reset_robot:
            arx.reset()
        arx.step_lift(17.0)
        time.sleep(1.0)

        planned = True
        plan_steps: List[str] = []
        plan_cups: List[str] = []

        goal_cup = goal
        planning_prompt = (
            f"Current Goal is: pick the {goal_cup}. "
            "I need to pick up the cups from top to the goal cup."
            "What is the picking plan steps to finish the goal?"
        )
        confirm_win = "Planning Step"
        if debug_raw:
            cv2.namedWindow(confirm_win, cv2.WINDOW_NORMAL)

        while True:
            frames = arx.get_camera(
                target_size=(640, 480), return_status=False)
            color = frames.get("camera_h_color")
            if color is None:
                continue
            current_plan, current_cups = do_replan(color, planning_prompt)
            if not debug_raw:
                if not current_plan:
                    continue
                plan_steps = current_plan[:4]
                plan_cups = current_cups[:4] if current_cups else []
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
                plan_steps = current_plan[:4]
                plan_cups = current_cups[:4] if current_cups else []
                print("规划已确认，进入执行模式。")
                break
            if key == ord("r"):
                print("重新尝试规划...")
                continue
            if key == ord("p"):
                new_goal = input("输入新的需求 (留空保持当前): ").strip()
                if new_goal:
                    goal_cup = new_goal
                    planning_prompt = (
                        f"Current Goal is: pick the {goal_cup}. "
                        "I need to pick up the cups from top to the goal cup."
                        "What is the picking plan steps to finish the goal?"
                    )
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
            color, depth = get_aligned_frames(arx, depth_median_n=depth_median_n)
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
                            temperature=0.0,
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
                    temperature=0.0,
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
                arm = _arm_for_step(i)
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
                arm = _arm_for_step(i)
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
            first_arm = _arm_for_step(0)
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
                cur_arm = _arm_for_step(i)
                next_arm = _arm_for_step(i + 1)
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
                last_arm = _arm_for_step(steps_n - 1)
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
        if close_robot:
            arx.close()


def main():
    arx = ARXRobotEnv(
        duration_per_step=1.0 / 20.0,
        min_steps=20,
        max_v_xyz=0.15, max_a_xyz=0.15,
        max_v_rpy=0.3, max_a_rpy=1.00,
        camera_type="all",
        camera_view=("camera_h",),
        img_size=(640, 480),
        # video=True,
        # video_fps=30.0,
        # dir="../Video4demo",
        # video_name="dual_cup_pick_planning_parallel_red",
    )
    dual_arm_pick_planning_parallel(
        arx,
        close_robot=False,
        goal="red cup",
        no_last_place=True,
        single_test=True,
        # dir="../Video4demo/dual_cup_pick_planning_parallel_red.png",
        depth_median_n=15,
    )
    time.sleep(10.0)
    arx.close()
if __name__ == "__main__":
    main()
