from __future__ import annotations

import re
import sys
import time
from pathlib import Path
from typing import List, Optional, Tuple

import cv2
import numpy as np

from arx_pointing import predict_multi_points_from_rgb
from motion_pick_place_cup import build_pick_cup_sequence, build_place_cup_sequence
from motion_pick_place_deepbox import (
    build_pick_deepbox_sequence,
    build_place_deepbox_sequence,
)
from motion_pick_place_normal_object import (
    build_pick_normal_object_sequence,
    build_place_normal_object_sequence,
)
from motion_pick_place_straw import (
    build_pick_straw_sequence,
    build_place_straw_sequence,
)

ROOT_DIR = Path(__file__).resolve().parent.parent
DEPLOYMENT_DIR = ROOT_DIR / "Deployment"
if str(DEPLOYMENT_DIR) not in sys.path:
    sys.path.append(str(DEPLOYMENT_DIR))

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
) -> None:
    if duration <= 0:
        print("base move duration must be positive")
        return
    arx.step_base(vx, vy, vz)
    time.sleep(duration)
    arx.step_base(0.0, 0.0, 0.0)


def run_push_away(arx) -> None:
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
    if abs(meters_per_lift_unit) < 1e-8:
        raise ValueError("meters_per_lift_unit must be non-zero")
    target_lift = float(current_lift) + (
        float(goal_z) - float(target_goal_z)
    ) / float(meters_per_lift_unit)
    return float(np.clip(target_lift, min_lift, max_lift))


def execute_pick_place_cup_sequence(
    arx,
    pick_ref: Optional[np.ndarray],
    place_ref: Optional[np.ndarray],
    arm: str,
    do_pick: bool = True,
    do_place: bool = True,
) -> None:
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
