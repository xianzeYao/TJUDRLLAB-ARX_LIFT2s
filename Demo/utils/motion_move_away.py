"""拨开遮挡物的简化动作模板。"""

import numpy as np
from typing import Dict, Optional

CALIBRATE_OFFSET_LEFT = 0.015
CALIBRATE_OFFSET_RIGHT = 0.01
GRIPPER_OFFSET = 0.14
APPROACH_Z_OFFSET = 0.02
PUSH_DISTANCE_Y = 0.05


def _make_arm_action(arm: str, active: np.ndarray) -> Dict[str, np.ndarray]:
    """把单臂动作包装成环境接口要求的字典。"""
    if arm == "left":
        return {"left": active}
    if arm == "right":
        return {"right": active}
    raise ValueError(f"arm must be 'left' or 'right', got: {arm!r}")


def _get_calibrate_offset(arm: str) -> float:
    """返回左右臂各自的 Y 方向补偿。"""
    if arm == "left":
        return CALIBRATE_OFFSET_LEFT
    if arm == "right":
        return CALIBRATE_OFFSET_RIGHT
    raise ValueError(f"arm must be 'left' or 'right', got: {arm!r}")


def _get_push_sign(arm: str) -> float:
    """根据左右臂确定推开的方向符号。"""
    if arm == "left":
        return 1.0
    if arm == "right":
        return -1.0
    raise ValueError(f"arm must be 'left' or 'right', got: {arm!r}")


def make_move_away_approach_action(
    pt_ref: Optional[np.ndarray],
    arm: str,
) -> Dict[str, np.ndarray]:
    """生成接近障碍物、准备推开的动作。"""
    base = np.zeros(3, dtype=np.float32) if pt_ref is None else pt_ref
    calibrate_offset = _get_calibrate_offset(arm)
    push_sign = _get_push_sign(arm)
    active = np.array(
        [
            base[0] - GRIPPER_OFFSET+0.0525,
            base[1] + calibrate_offset+push_sign*PUSH_DISTANCE_Y,
            base[2] + APPROACH_Z_OFFSET,
            0,
            0,
            0,
            0,
        ],
        dtype=np.float32,
    )
    return _make_arm_action(arm, active)



def make_move_away_push_action(
    pt_ref: Optional[np.ndarray],
    arm: str,
) -> Dict[str, np.ndarray]:
    """生成侧向推开障碍物的动作。"""
    base = np.zeros(3, dtype=np.float32) if pt_ref is None else pt_ref
    calibrate_offset = _get_calibrate_offset(arm)
    push_sign = _get_push_sign(arm)
    active = np.array(
        [
            base[0] - GRIPPER_OFFSET + 0.0525,
            base[1] + calibrate_offset + push_sign *
            PUSH_DISTANCE_Y-push_sign * PUSH_DISTANCE_Y*2.5,
            base[2] + APPROACH_Z_OFFSET,
            0,
            0,
            0,
            0,
        ],
        dtype=np.float32,
    )
    return _make_arm_action(arm, active)


def build_move_away_sequence(
    pt_ref: Optional[np.ndarray],
    arm: str,
):
    """返回“接近 + 推开”两步组成的动作序列。"""
    return [
        make_move_away_approach_action(pt_ref, arm),
        make_move_away_push_action(pt_ref, arm),
    ]
