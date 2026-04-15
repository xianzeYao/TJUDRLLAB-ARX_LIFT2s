from __future__ import annotations

"""Demo 里的通用文本解析、底盘控制和动作序列执行辅助函数。"""

import re
import sys
import time
from pathlib import Path
from typing import Callable, List, Optional, Tuple

import cv2
import numpy as np

from arx_pointing import predict_multi_points_from_rgb
from motion_pick_place_cup import (
    GRIPPER_CUP,
    build_pick_cup_sequence,
    build_place_cup_sequence,
)
from motion_pick_place_deepbox import (
    GRIPPER_DEEPBOX,
    build_pick_deepbox_sequence,
    build_place_deepbox_sequence,
)
from motion_pick_place_normal_object import (
    GRIPPER_NORMAL_OBJECT,
    build_pick_normal_object_sequence,
    build_place_normal_object_sequence,
)
from motion_pick_place_straw import (
    CLOSE as GRIPPER_STRAW,
    build_pick_straw_sequence,
    build_place_straw_sequence,
)
from motion_pick_place_hug import (
    build_pick_hug_sequence,
    build_place_hug_sequence,
)
from motion_move_away import build_move_away_sequence

ROOT_DIR = Path(__file__).resolve().parent.parent
DEPLOYMENT_DIR = ROOT_DIR / "Deployment"
if str(DEPLOYMENT_DIR) not in sys.path:
    sys.path.append(str(DEPLOYMENT_DIR))

NORMAL_OBJECT_PICK_CLOSE_THRESHOLD = -0.1

LIFT_SAMPLES = np.array([0.0, 5.0, 10.0, 15.0, 20.0], dtype=np.float32)
REF_HEIGHT_SAMPLES_M = np.array(
    [0.55, 0.65, 0.80, 0.92, 1.02], dtype=np.float32)


def _extract_cup_phrases(text: str) -> List[str]:
    phrases: List[str] = []
    seen = set()
    for m in re.finditer(r"\b([A-Za-z]+)\s+cup\b", text):
        phrase = m.group(0).strip()
        key = phrase.lower()
        if key in seen:
            continue
        seen.add(key)
        phrases.append(phrase)
    return phrases


def _unwrap_answer_block(raw: str) -> str:
    match = re.search(r"<answer>(.*?)</answer>", raw,
                      flags=re.DOTALL | re.IGNORECASE)
    if not match:
        return raw
    return match.group(1).strip()


def extract_numbered_sentences(raw: Optional[str]) -> Tuple[List[str], List[str]]:
    """提取形如 '1. xxx' / '2) xxx' / '3- xxx' 的编号句子，并提取 'xx cup'。"""
    if not raw:
        return [], []
    # 去掉代码块包裹
    raw_clean = re.sub(
        r"```(?:json|python)?\n?(.*?)\n?```", r"\1", raw, flags=re.DOTALL
    )
    raw_clean = _unwrap_answer_block(raw_clean)
    steps: List[str] = []

    # 行内 / 行间匹配：1. xxx 2) yyy 3- zzz
    inline_matches = re.finditer(
        r"(\d+)[\.\)\-]\s*(.+?)(?=(?:\d+[\.\)\-])|$)",
        raw_clean,
        flags=re.DOTALL,
    )
    for m in inline_matches:
        steps.append(m.group(2).strip())

    # 去重保持顺序
    seen = set()
    uniq_steps = []
    for s in steps:
        if s not in seen:
            seen.add(s)
            uniq_steps.append(s)
    if not uniq_steps:
        # 兼容模型返回非编号的逐行答案。
        fallback_lines = [
            line.strip(" -*\t")
            for line in raw_clean.splitlines()
            if line.strip()
        ]
        uniq_steps = fallback_lines

    cup_text = " ".join(uniq_steps) if uniq_steps else raw_clean
    cups = _extract_cup_phrases(cup_text)
    return uniq_steps, cups


def do_replan(
    color_img: np.ndarray,
    planning_prompt: str,
    max_retries: int = 5,
) -> Tuple[List[str], List[str], Optional[str]]:
    """重复调用规划模型，直到解析出有效步骤列表。"""
    last_text: Optional[str] = None
    for _ in range(max_retries):
        raw_result = predict_multi_points_from_rgb(
            color_img,
            text_prompt="",
            all_prompt=planning_prompt,
            assume_bgr=False,
            return_raw=True,
            temperature=0.7,
        )
        if isinstance(raw_result, tuple):
            _, pick_answer_text = raw_result
        else:
            pick_answer_text = None

        last_text = pick_answer_text
        pick_plan, cups = extract_numbered_sentences(pick_answer_text)
        if pick_plan:
            return pick_plan, cups, pick_answer_text

    raise RuntimeError(
        f"failed to parse planning result after {max_retries} retries: {last_text!r}")


def draw_text_lines(
    img: np.ndarray,
    lines: List[str],
    origin: tuple[int, int] = (10, 30),
    line_height: int = 22,
    color: tuple[int, int, int] = (0, 0, 255),
    scale: float = 0.5,
    thickness: int = 2,
) -> None:
    """按固定行高在图像上绘制多行文字。"""
    x, y = origin
    for i, line in enumerate(lines):
        cv2.putText(
            img,
            line,
            (x, y + i * line_height),
            cv2.FONT_HERSHEY_SIMPLEX,
            scale,
            color,
            thickness,
        )


def draw_point_label(
    img: np.ndarray,
    label: str,
    pos: tuple[int, int],
    color: tuple[int, int, int] = (255, 255, 255),
) -> None:
    """在指定像素点附近绘制一个简短标签。"""
    draw_text_lines(
        img,
        [label],
        origin=(pos[0] + 6, pos[1] - 6),
        line_height=18,
        color=color,
        scale=0.55,
        thickness=2,
    )


def step_base_duration(
    arx,
    vx: float,
    vy: float,
    vz: float,
    duration: float,
    should_stop: Optional[Callable[[], bool]] = None,
    poll_interval_s: float = 0.05,
) -> bool:
    """让底盘按给定速度运动一段时间，可选在运动中轮询中断条件。"""
    if duration <= 0:
        print("base move duration must be positive")
        return True
    arx.step_base(vx, vy, vz)
    completed = True
    try:
        if should_stop is None:
            time.sleep(duration)
        else:
            end_time = time.monotonic() + duration
            while True:
                remaining = end_time - time.monotonic()
                if remaining <= 0:
                    break
                if should_stop():
                    completed = False
                    break
                time.sleep(min(max(0.0, float(poll_interval_s)), remaining))
    finally:
        arx.step_base(0.0, 0.0, 0.0)
    return completed


def run_push_away(arx) -> None:
    """调用 ACT 模型执行 push-away，再把右臂回到 home。"""
    from testACT_parallel import run_act

    run_act(
        arx=arx,
        model_path="/home/arx/Arx_Lift2s/Deployment/models/push_away_act",
        arm_side="right",
        hz=20.0,
        chunk_size=30,
        max_steps=250,
        temporal_ensemble_coeff=0.01,
    )
    success, error_message = arx.set_special_mode(1, side="right")
    if not success:
        raise RuntimeError(
            f"failed to reset right arm after push_away: {error_message}"
        )


def estimate_lift_from_goal_z(
    goal_z: float,
    current_lift: float,
    target_goal_z: float = 0.0,
    meters_per_lift_unit: float = float(
        (REF_HEIGHT_SAMPLES_M[-1] - REF_HEIGHT_SAMPLES_M[0]) /
        (LIFT_SAMPLES[-1] - LIFT_SAMPLES[0])
    ),
    min_lift: float = 0.0,
    max_lift: float = 20.0,
) -> float:
    """根据目标点高度估算新的升降目标，并限制在有效范围内。"""
    if abs(meters_per_lift_unit) < 1e-8:
        raise ValueError("meters_per_lift_unit must be non-zero")
    target_lift = float(current_lift) + (
        float(goal_z) - float(target_goal_z)
    ) / float(meters_per_lift_unit)
    return float(np.clip(target_lift, min_lift, max_lift))


def get_pick_close_target(item_type: str) -> float:
    """返回指定物体类型 pick 完成后的目标夹爪值。"""
    if item_type == "cup":
        return float(GRIPPER_CUP)
    if item_type == "straw":
        return float(GRIPPER_STRAW)
    if item_type == "deepbox":
        return float(GRIPPER_DEEPBOX)
    if item_type == "normal object":
        return float(NORMAL_OBJECT_PICK_CLOSE_THRESHOLD)
    raise ValueError(f"unknown item_type: {item_type!r}")


def execute_pick_place_cup_sequence(
    arx,
    pick_ref: Optional[np.ndarray],
    place_ref: Optional[np.ndarray],
    arm: str,
    do_pick: bool = True,
    do_place: bool = True,
) -> None:
    """执行杯子的抓取/放置动作序列。"""
    if do_pick:
        if pick_ref is None:
            raise ValueError("pick_ref 为空")
        pick_seq = build_pick_cup_sequence(pick_ref, arm=arm)
        for act in pick_seq:
            arx.step_smooth_eef(act)
    if do_place:
        if place_ref is None:
            raise ValueError("place_ref 为空")
        place_seq = build_place_cup_sequence(place_ref, arm=arm)
        for act in place_seq:
            arx.step_smooth_eef(act)


def execute_pick_place_straw_sequence(
    arx,
    pick_ref: Optional[np.ndarray],
    place_ref: Optional[np.ndarray],
    arm: str,
    do_pick: bool = True,
    do_place: bool = True,
) -> None:
    """执行吸管的抓取/放置动作序列。"""
    if do_pick:
        if pick_ref is None:
            raise ValueError("pick_ref 为空")
        pick_seq = build_pick_straw_sequence(pick_ref, arm=arm)
        for act in pick_seq:
            arx.step_smooth_eef(act)
    if do_place:
        if place_ref is None:
            raise ValueError("place_ref 为空")
        place_seq = build_place_straw_sequence(place_ref, arm=arm)
        for act in place_seq:
            arx.step_smooth_eef(act)


def execute_pick_place_deepbox_sequence(
    arx,
    pick_ref: Optional[np.ndarray],
    place_ref: Optional[np.ndarray],
    arm: str,
    do_pick: bool = True,
    do_place: bool = True,
) -> None:
    """执行深盒类目标的抓取/放置动作序列。"""
    if do_pick:
        if pick_ref is None:
            raise ValueError("pick_ref 为空")
        pick_seq = build_pick_deepbox_sequence(pick_ref, arm=arm)
        for act in pick_seq:
            arx.step_smooth_eef(act)
    if do_place:
        if place_ref is None:
            raise ValueError("place_ref 为空")
        place_seq = build_place_deepbox_sequence(place_ref, arm=arm)
        for act in place_seq:
            arx.step_smooth_eef(act)


def execute_pick_place_normal_object_sequence(
    arx,
    pick_ref: Optional[np.ndarray],
    place_ref: Optional[np.ndarray],
    arm: str,
    do_pick: bool = True,
    do_place: bool = True,
) -> None:
    """执行普通物体的抓取/放置动作序列。"""
    if do_pick:
        if pick_ref is None:
            raise ValueError("pick_ref 为空")
        pick_seq = build_pick_normal_object_sequence(pick_ref, arm=arm)
        for act in pick_seq:
            arx.step_smooth_eef(act)
    if do_place:
        if place_ref is None:
            raise ValueError("place_ref 为空")
        place_seq = build_place_normal_object_sequence(place_ref, arm=arm)
        for act in place_seq:
            arx.step_smooth_eef(act)


def execute_pick_place_hug_sequence(
    arx,
    left_ref: Optional[np.ndarray],
    right_ref: Optional[np.ndarray],
    do_pick: bool = True,
    do_place: bool = False,
    left_pose: Optional[np.ndarray] = None,
    right_pose: Optional[np.ndarray] = None,
) -> None:
    """执行 hug 抓取/放置动作序列。
    注意pick 和 place 不能同时执行。
    """
    if do_pick:
        if left_ref is None:
            raise ValueError("left_ref 为空")
        if right_ref is None:
            raise ValueError("right_ref 为空")
        pick_seq = build_pick_hug_sequence(left_ref=left_ref, right_ref=right_ref)
        for act in pick_seq:
            arx.step_smooth_eef(act)
    if do_place:
        if left_ref is None:
            raise ValueError("left_ref 为空")
        if right_ref is None:
            raise ValueError("right_ref 为空")
        if left_pose is None:
            raise ValueError("left_pose 为空，place 需要当前左臂末端姿态")
        if right_pose is None:
            raise ValueError("right_pose 为空，place 需要当前右臂末端姿态")
        place_seq = build_place_hug_sequence(
            left_ref=left_ref,
            right_ref=right_ref,
            left_pose=left_pose,
            right_pose=right_pose,
        )
        for act in place_seq:
            arx.step_smooth_eef(act)


def _build_place_sequence_for_item(
    item_type: str,
    place_ref: np.ndarray,
    arm: str,
):
    if item_type == "cup":
        return build_place_cup_sequence(place_ref, arm=arm)
    if item_type == "straw":
        return build_place_straw_sequence(place_ref, arm=arm)
    if item_type == "deepbox":
        return build_place_deepbox_sequence(place_ref, arm=arm)
    if item_type == "normal object":
        return build_place_normal_object_sequence(place_ref, arm=arm)
    raise ValueError(f"unknown item_type: {item_type!r}")


def _raise_lowest_place_z_only(
    place_seq,
    *,
    arm: str,
    z_raise: float,
):
    if abs(float(z_raise)) < 1e-8:
        return place_seq

    z_values: List[float] = []
    for act in place_seq:
        active = act.get(arm)
        if isinstance(active, np.ndarray) and active.shape[0] >= 3:
            z_values.append(float(active[2]))
    if not z_values:
        return place_seq

    min_z = min(z_values)
    adjusted_seq = []
    for act in place_seq:
        adjusted_act = {}
        for key, value in act.items():
            if isinstance(value, np.ndarray):
                adjusted_act[key] = np.asarray(value, dtype=np.float32).copy()
            else:
                adjusted_act[key] = value
        active = adjusted_act.get(arm)
        if (
            isinstance(active, np.ndarray)
            and active.shape[0] >= 3
            and np.isclose(float(active[2]), min_z)
        ):
            active[2] += float(z_raise)
        adjusted_seq.append(adjusted_act)
    return adjusted_seq


def _adjust_last_place_x_only(
    place_seq,
    *,
    arm: str,
    x_shift: float,
):
    if abs(float(x_shift)) < 1e-8 or not place_seq:
        return place_seq

    adjusted_seq = []
    last_idx = len(place_seq) - 1
    for idx, act in enumerate(place_seq):
        adjusted_act = {}
        for key, value in act.items():
            if isinstance(value, np.ndarray):
                adjusted_act[key] = np.asarray(value, dtype=np.float32).copy()
            else:
                adjusted_act[key] = value
        if idx == last_idx:
            active = adjusted_act.get(arm)
            if isinstance(active, np.ndarray) and active.shape[0] >= 1:
                active[0] += float(x_shift)
        adjusted_seq.append(adjusted_act)
    return adjusted_seq


def execute_return_to_source_sequence(
    arx,
    pick_ref: Optional[np.ndarray],
    arm: str,
    item_type: str,
    ref_x_shift: float = 0.07,
    ref_z_shift: float = -0.15,
    bottom_z_raise: float = 0.15,
    retreat_x_shift: float = -0.05,
) -> None:
    """把当前抓着的物体按对应放置模板送回 pick 源位附近释放。"""
    if pick_ref is None:
        raise ValueError("pick_ref 为空")
    adjusted_pick_ref = np.asarray(pick_ref, dtype=np.float32).copy()
    adjusted_pick_ref[0] += float(ref_x_shift)
    adjusted_pick_ref[2] += float(ref_z_shift)
    place_seq = _build_place_sequence_for_item(
        item_type=item_type,
        place_ref=adjusted_pick_ref,
        arm=arm,
    )
    place_seq = _raise_lowest_place_z_only(
        place_seq,
        arm=arm,
        z_raise=bottom_z_raise,
    )
    place_seq = _adjust_last_place_x_only(
        place_seq,
        arm=arm,
        x_shift=retreat_x_shift,
    )
    for act in place_seq:
        arx.step_smooth_eef(act)


def execute_move_away(
    arx,
    blocker_ref: Optional[np.ndarray],
    arm: str,
) -> None:
    """执行拨开障碍物的动作序列。"""
    if blocker_ref is None:
        raise ValueError("blocker_ref 为空")
    move_away_seq = build_move_away_sequence(blocker_ref, arm=arm)
    for act in move_away_seq:
        arx.step_smooth_eef(act)
