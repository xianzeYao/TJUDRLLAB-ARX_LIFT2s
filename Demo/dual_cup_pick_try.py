from __future__ import annotations

import json
import re
import sys
import textwrap
import time
from typing import Literal, Optional, Tuple

import cv2

from arx_pointing import predict_multi_points_from_rgb
from demo_utils import draw_text_lines
from single_arm_pick_place import single_arm_pick_place

sys.path.append("../ARX_Realenv/ROS2")  # noqa
from arx_ros2_env import ARXRobotEnv  # noqa

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


def _build_next_step_prompt(goal_cup: str) -> str:
    return textwrap.dedent(
        f"""
        Current goal: pick the {goal_cup}.
        I need to pick up the cups from top to the goal cup.
        If there is no cup directly upon the {goal_cup}, return the goal cup.
        What is the exact next picking plan step to finish the goal? Return only the cup description.
        """
    ).strip()


def _unwrap_answer_block(raw: str) -> str:
    match = re.search(r"<answer>(.*?)</answer>", raw,
                      flags=re.DOTALL | re.IGNORECASE)
    if match:
        return match.group(1).strip()
    return raw


def _extract_json_object(raw: str) -> str:
    cleaned = _unwrap_answer_block(raw.strip())
    code_match = re.search(
        r"```(?:json)?\s*(\{.*?\})\s*```", cleaned, flags=re.DOTALL)
    if code_match:
        return code_match.group(1)
    obj_match = re.search(r"\{.*\}", cleaned, flags=re.DOTALL)
    if not obj_match:
        raise ValueError(f"未找到 JSON 对象: {raw!r}")
    return obj_match.group(0)


def _normalize_cup_desc(cup_desc: Optional[str], goal_cup: str) -> str:
    desc = (cup_desc or "").strip()
    desc = desc.strip(" \t\r\n\"'.,;:")
    if not desc:
        desc = goal_cup
    if "cup" not in desc.lower():
        desc = f"{desc} cup"
    return desc


def _extract_cup_phrase(desc: str) -> Optional[str]:
    match = re.search(
        r"\b([A-Za-z]+(?:\s+[A-Za-z]+)*)\s+cup\b",
        desc,
        flags=re.IGNORECASE,
    )
    if not match:
        return None
    return f"{match.group(1).strip()} cup"


def _parse_next_step(raw_text: Optional[str], goal_cup: str) -> Tuple[str, bool]:
    if not raw_text:
        raise ValueError("空规划结果")
    cup_desc_raw: Optional[str] = None
    try:
        payload = json.loads(_extract_json_object(raw_text))
        if isinstance(payload, list):
            if not payload:
                raise ValueError("空 JSON 列表")
            payload = payload[0]
        if isinstance(payload, dict):
            cup_desc_raw = payload.get("cup")
        elif isinstance(payload, str):
            cup_desc_raw = payload
        else:
            raise ValueError(f"规划结果不是可解析的 cup 描述: {payload!r}")
    except (json.JSONDecodeError, ValueError, TypeError):
        cup_desc_raw = _unwrap_answer_block(raw_text).strip()

    cup_desc = _normalize_cup_desc(
        _extract_cup_phrase(cup_desc_raw or "") or cup_desc_raw,
        goal_cup=goal_cup,
    )
    goal_desc = _normalize_cup_desc(
        _extract_cup_phrase(goal_cup) or goal_cup,
        goal_cup=goal_cup,
    )
    is_goal = cup_desc.lower() == goal_desc.lower()
    return cup_desc, is_goal


def _predict_next_step(
    color,
    goal_cup: str,
    max_retries: int = 5,
) -> Tuple[str, bool, str]:
    prompt = _build_next_step_prompt(goal_cup)
    last_text: Optional[str] = None
    last_error: Optional[Exception] = None
    for _ in range(max_retries):
        raw_result = predict_multi_points_from_rgb(
            color,
            text_prompt="",
            all_prompt=prompt,
            assume_bgr=False,
            return_raw=True,
            temperature=0.0,
        )
        if isinstance(raw_result, tuple):
            _, raw_text = raw_result
        else:
            raw_text = None
        last_text = raw_text
        try:
            cup_desc, is_goal = _parse_next_step(raw_text, goal_cup=goal_cup)
            return cup_desc, is_goal, prompt
        except (json.JSONDecodeError, ValueError, TypeError) as exc:
            last_error = exc
            continue

    raise RuntimeError(
        f"failed to parse next-step result after {max_retries} retries: "
        f"raw={last_text!r}, err={last_error!r}"
    )


def dual_cup_pick_try(
    arx: ARXRobotEnv,
    goal: str = "red cup",
    first_side: Literal["left", "right"] = "left",
    debug_raw: bool = True,
    depth_median_n: int = 5,
    single_arm_debug: bool = True,
    no_last_place: bool = False,
) -> None:
    planning_win = "dual_cup_pick_try_planning"
    step_idx = 0
    place_idx = 0
    goal_cup = goal

    try:
        arx.step_lift(17.0)
        time.sleep(1.0)
        if debug_raw:
            cv2.namedWindow(planning_win, cv2.WINDOW_NORMAL)

        while True:
            frames = arx.get_camera(target_size=(
                640, 480), return_status=False)
            color = frames.get("camera_h_color")
            if color is None:
                continue

            try:
                cup_desc, is_goal, planning_prompt = _predict_next_step(
                    color,
                    goal_cup=goal_cup,
                )
            except RuntimeError as exc:
                print(f"一步规划失败，自动重试：{exc}")
                continue

            do_place = not (is_goal and no_last_place)
            if do_place and place_idx >= len(COASTER_PROMPTS):
                print(
                    f"已无可用 coaster 槽位(place_idx={place_idx})，"
                    "但目标仍未完成，结束本次尝试。"
                )
                break

            arm = _arm_for_step(step_idx, first_side=first_side)
            place_prompt = COASTER_PROMPTS[place_idx] if do_place else ""

            if debug_raw:
                vis = color.copy()
                prompt_lines = textwrap.wrap(planning_prompt, width=62)
                info_lines = [
                    f"Step: {step_idx + 1}",
                    f"Arm: {arm}",
                    f"Goal: {goal_cup}",
                    f"Next cup: {cup_desc}",
                    f"is_goal={is_goal} | no_last_place={no_last_place}",
                    f"Place: {place_prompt or '[skip place]'}",
                    "Press 'y' to confirm, 'r' to retry, 'p' to update goal, 'q' to quit",
                ]
                draw_text_lines(
                    vis,
                    ["Planning Prompt:"] + prompt_lines + [""] + info_lines,
                    origin=(10, 25),
                    line_height=20,
                    color=(0, 0, 255),
                    scale=0.5,
                    thickness=2,
                )
                cv2.imshow(planning_win, vis)
                print(
                    f"第 {step_idx + 1} 步规划: cup={cup_desc!r}, is_goal={is_goal}, arm={arm}")
                print("按 'y' 确认，按 'r' 重试，按 'p' 更新目标，按 'q' 退出")
                key = cv2.waitKey(0)
                if key == ord("r"):
                    continue
                if key == ord("p"):
                    new_goal = input("输入新的目标 (留空保持当前): ").strip()
                    if new_goal:
                        goal_cup = new_goal
                        print(f"新的 goal 已设置为: {goal_cup!r}")
                    continue
                if key == ord("q"):
                    print("退出程序。")
                    break
                if key != ord("y"):
                    continue

            arx.step_lift(13.0)
            time.sleep(1.0)
            pick_ref, _, used_arm = single_arm_pick_place(
                arx,
                pick_prompt=cup_desc,
                place_prompt="",
                arm_side=arm,
                item_type="cup",
                debug=single_arm_debug,
                depth_median_n=depth_median_n,
            )
            if pick_ref is None or used_arm is None:
                print("pick 阶段被取消，结束程序。")
                break

            place_ref = None
            if do_place:
                _, place_ref, place_arm = single_arm_pick_place(
                    arx,
                    pick_prompt="",
                    place_prompt=place_prompt,
                    arm_side=arm,
                    item_type="cup",
                    debug=single_arm_debug,
                    depth_median_n=depth_median_n,
                )
                if place_ref is None or place_arm is None:
                    print("place 阶段被取消，结束程序。")
                    break
                place_idx += 1
            step_idx += 1

            if is_goal:
                print(
                    f"目标 {goal_cup!r} 已完成，"
                    f"{'最后一步未放置。' if not do_place else '最后一步已完成 pick+place。'}"
                )
                break

            if place_idx >= len(COASTER_PROMPTS):
                print(
                    f"已用完 {len(COASTER_PROMPTS)} 个 coaster 槽位，"
                    f"但目标 {goal_cup!r} 仍未完成，结束本次尝试。"
                )
                break
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
        dual_cup_pick_try(
            arx,
            goal="red cup",
            first_side="left",
            debug_raw=True,
            depth_median_n=5,
            single_arm_debug=False,
            no_last_place=False,
        )
        time.sleep(3.0)
    finally:
        arx.close()


if __name__ == "__main__":
    main()
