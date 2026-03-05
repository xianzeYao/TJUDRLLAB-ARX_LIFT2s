from __future__ import annotations
import sys
import time
from demo_utils import (
    do_replan,
    draw_text_lines,
    execute_pick_place_cup_sequence,
)
from arx_pointing import predict_multi_points_from_rgb
from point2pos_utils import load_cam2ref, load_intrinsics, pixel_to_ref_point

import textwrap
from typing import List

import cv2
import numpy as np


sys.path.append("../ARX_Realenv/ROS2")  # noqa
from arx_ros2_env import ARXRobotEnv  # noqa


def _predict_pick_place_once(
    color: np.ndarray, base_prompt: str
) -> tuple[tuple[int, int], tuple[int, int]]:
    full_prompt = (
        "Provide exactly two points coordinate of the pick object and the place coaster this sentence describes: "
        f"{base_prompt} "
        'The answer should be presented in JSON format as follows: [{"point_2d": [x, y]}]. '
        "Return only JSON. First point is the object, second point is the coaster."
    )
    points = predict_multi_points_from_rgb(
        color,
        text_prompt="",
        all_prompt=full_prompt,
        assume_bgr=False,
        temperature=0.0,
    )
    if len(points) < 2:
        raise RuntimeError(f"未获取到足够点({len(points)}): {points}")
    pick = (int(round(points[0][0])), int(round(points[0][1])))
    place = (int(round(points[1][0])), int(round(points[1][1])))
    return pick, place


def _predict_pick_only(
    color: np.ndarray, base_prompt: str
) -> tuple[int, int]:
    full_prompt = (
        "Provide exactly one point coordinate of objects region this sentence describes: "
        f"{base_prompt} "
        'The answer should be presented in JSON format as follows: [{"point_2d": [x, y]}]. '
        "Return only JSON."
    )
    points = predict_multi_points_from_rgb(
        color,
        text_prompt="",
        all_prompt=full_prompt,
        assume_bgr=False,
        temperature=0.0,
    )
    if not points:
        raise RuntimeError(f"未获取到点: {points}")
    pick = (int(round(points[0][0])), int(round(points[0][1])))
    return pick


def _arm_for_step(step_idx: int) -> str:
    return "left" if step_idx % 2 == 0 else "right"


def _coaster_side_for_arm(arm: str) -> str:
    return "left coaster" if arm == "left" else "right coaster"


def _build_dual_prompt(target_text: str, arm: str) -> str:
    coaster_side = _coaster_side_for_arm(arm)
    return f"Point out the {target_text} and the nearest {coaster_side} of it and has no cup on it."


def _predict_step(
    color: np.ndarray,
    step_idx: int,
    step_text: str,
    target_text: str,
    arm: str,
    skip_place: bool,
) -> dict:
    if skip_place:
        dual_prompt = f"Point out the {target_text}."
    else:
        dual_prompt = _build_dual_prompt(target_text, arm)

    result = {
        "step_idx": step_idx,
        "arm": arm,
        "skip_place": skip_place,
        "dual_prompt": dual_prompt,
        "pick_px": None,
        "place_px": None,
        "ok": False,
    }

    try:
        if skip_place:
            result["pick_px"] = _predict_pick_only(color, dual_prompt)
        else:
            pick_px, place_px = _predict_pick_place_once(color, dual_prompt)
            result["pick_px"] = pick_px
            result["place_px"] = place_px
        result["ok"] = True
    except RuntimeError as exc:
        print(f"预测失败，按 r 重试：{exc}")

    return result


def _execute_steps(
    arx: ARXRobotEnv,
    steps: list[dict],
    depth: np.ndarray,
    K: np.ndarray,
    T_left: np.ndarray,
    T_right: np.ndarray,
) -> bool:
    try:
        for step in steps:
            T_cam2ref = T_left if step["arm"] == "left" else T_right
            pick_ref = pixel_to_ref_point(step["pick_px"], depth, K, T_cam2ref)
            place_ref = None
            if not step["skip_place"]:
                if step["place_px"] is None:
                    raise ValueError("place 点缺失")
                place_ref = pixel_to_ref_point(
                    step["place_px"], depth, K, T_cam2ref)
            execute_pick_place_cup_sequence(
                arx=arx,
                pick_ref=pick_ref,
                place_ref=place_ref,
                arm=step["arm"],
                do_pick=True,
                do_place=not step["skip_place"],
                go_home=step.get("go_home", True),
            )
        return True
    except ValueError as exc:
        print(f"像素/深度异常，重新预测：{exc}")
        return False


def dual_arm_pick_planning(
    arx: ARXRobotEnv,
    reset_robot: bool = True,
    close_robot: bool = True,
    no_last_place: bool = False,
    goal: str = "red cup",
):
    try:
        if reset_robot:
            arx.reset()
        arx.step_lift(16.0)
        time.sleep(1.0)
        K = load_intrinsics()
        T_left, T_right = load_cam2ref()

        planned = True
        step_idx = 0
        plan_steps: List[str] = []

        goal_cup = goal
        planning_prompt = (
            f"Current Goal is: pick the {goal_cup}. "
            "I need to pick up the cups from top to the goal cup."
            "What is the picking plan steps to finish the goal?"
        )
        confirm_win = "Planning Step"
        cv2.namedWindow(confirm_win, cv2.WINDOW_NORMAL)

        while True:
            frames = arx.node.get_camera(
                target_size=(640, 480), return_status=False)
            color = frames.get("camera_h_color")
            if color is None:
                cv2.waitKey(1)
                continue
            # 调用 VLM 生成步骤
            current_plan, plan_cups = do_replan(color, planning_prompt)

            # --- 可视化：在图像上打印出规划 prompt 供确认 ---
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
            print("按'y' 确认, 'r' 重试, 'p' 更新目标, 'n' 退出")
            key = cv2.waitKey(0)
            if key == ord("y") and current_plan:
                plan_steps = current_plan
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
            if key == ord("n"):
                print("退出程序。")
                planned = False
                break

        if planned:
            cv2.destroyWindow(confirm_win)
            win = "dual_cup_pick_planning"
            cv2.namedWindow(win, cv2.WINDOW_NORMAL)

        while planned and step_idx < len(plan_steps):
            if step_idx != 0:
                arx.step_lift(13.5)
                time.sleep(1.0)
            frames = arx.node.get_camera(
                target_size=(640, 480), return_status=False)
            color = frames.get("camera_h_color")
            depth = frames.get("camera_h_aligned_depth_to_color")
            if color is None or depth is None:
                cv2.waitKey(1)
                continue

            color = color.copy()
            arm = _arm_for_step(step_idx)
            pick_text = plan_steps[step_idx]
            target_text = pick_text
            if plan_cups and step_idx < len(plan_cups):
                target_text = plan_cups[step_idx]
            is_last = step_idx == len(plan_steps) - 1
            skip_place = no_last_place and is_last
            current = _predict_step(
                color, step_idx, pick_text, target_text, arm, skip_place
            )

            pick_px = current["pick_px"]
            place_px = current["place_px"]
            dual_prompt = current["dual_prompt"]
            arm = current["arm"]
            skip_place = current["skip_place"]
            display_prompt = (
                f"Point out the {pick_text}."
                if skip_place
                else _build_dual_prompt(pick_text, arm)
            )

            if pick_px is not None:
                cv2.circle(color, pick_px, 3,  (0, 0, 255), -1)
            if place_px is not None:
                cv2.circle(color, place_px, 3,  (255, 0, 0), -1)

            prompt_lines = textwrap.wrap(display_prompt, width=60)
            if skip_place:
                prompt_lines = ["(no_last_place)"] + prompt_lines
            draw_text_lines(
                color,
                [f"Step {step_idx + 1}/{len(plan_steps)} ({arm}):"] +
                prompt_lines,
                origin=(10, 30),
                line_height=20,
                color=(0, 0, 255),
                scale=0.55,
                thickness=2,
            )
            cv2.imshow(win, color)

            key = cv2.waitKey(0)
            if key == ord("r"):
                continue
            if key == ord("n"):
                break
            if key == ord("e"):
                if pick_px is None:
                    print("当前未预测到足够点，按 r 重新预测。")
                    continue
                ok = _execute_steps(
                    arx=arx,
                    steps=[
                        {
                            "pick_px": pick_px,
                            "place_px": place_px,
                            "arm": arm,
                            "skip_place": skip_place,
                            "go_home": not (no_last_place and is_last),
                        }
                    ],
                    depth=depth,
                    K=K,
                    T_left=T_left,
                    T_right=T_right,
                )
                if not ok:
                    continue

                step_idx += 1

        if planned and step_idx >= len(plan_steps):
            print("全部步骤已完成。")

    finally:
        cv2.destroyAllWindows()
        if close_robot:
            arx.close()


def main():
    arx = ARXRobotEnv(
        duration_per_step=1.0 / 20.0,
        min_steps=20,
        max_v_xyz=0.25, max_a_xyz=0.20,
        max_v_rpy=0.3, max_a_rpy=1.00,
        camera_type="all",
        camera_view=("camera_h",),
        img_size=(640, 480),
    )
    dual_arm_pick_planning(arx, close_robot=False,
                           no_last_place=True, goal="red cup")


if __name__ == "__main__":
    main()
