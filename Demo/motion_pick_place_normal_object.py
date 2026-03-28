import numpy as np
from typing import Dict, Optional

CLOSE = 0.0
OPEN = -3.3
GRIPPER_OFFSET = 0.14
GRIPPER_NORMAL_OBJECT = -0.5
Z_NORMAL_OBJECT = 0.075
PITCH_NORMAL_OBJECT = 0.35
CALIBRATE_OFFSET_LEFT = 0.015
CALIBRATE_OFFSET_RIGHT = 0.01


def _make_arm_action(arm: str, active: np.ndarray) -> Dict[str, np.ndarray]:
    if arm == "left":
        return {"left": active}
    if arm == "right":
        return {"right": active}
    raise ValueError(f"arm must be 'left' or 'right', got: {arm!r}")


def _get_calibrate_offset(arm: str) -> float:
    if arm == "left":
        return CALIBRATE_OFFSET_LEFT
    if arm == "right":
        return CALIBRATE_OFFSET_RIGHT
    raise ValueError(f"arm must be 'left' or 'right', got: {arm!r}")


def make_pick_move_action(pt_ref: Optional[np.ndarray], arm: str) -> Dict[str, np.ndarray]:
    """张开夹爪偏移到目标附近，保持不动。"""
    base = np.zeros(3, dtype=np.float32) if pt_ref is None else pt_ref
    calibrate_offset = _get_calibrate_offset(arm)
    active = np.array(
        [
            base[0] - GRIPPER_OFFSET - 0.03,
            base[1] + calibrate_offset,
            base[2] + 0.06,
            0,
            PITCH_NORMAL_OBJECT,
            0,
            OPEN,
        ],
        dtype=np.float32,
    )
    return _make_arm_action(arm, active)


def make_pick_robust_action(pt_ref: Optional[np.ndarray], arm: str) -> Dict[str, np.ndarray]:
    """执行向前移动，准备鲁棒夹取位置，保持不动。"""
    base = np.zeros(3, dtype=np.float32) if pt_ref is None else pt_ref
    calibrate_offset = _get_calibrate_offset(arm)
    active = np.array(
        [
            base[0] - GRIPPER_OFFSET + 0.03,
            base[1] + calibrate_offset,
            base[2] + 0.05,
            0,
            PITCH_NORMAL_OBJECT,
            0,
            OPEN,
        ],
        dtype=np.float32,
    )
    return _make_arm_action(arm, active)


def make_close_action(pt_ref: Optional[np.ndarray], arm: str) -> Dict[str, np.ndarray]:
    """执行夹紧动作，保持不动。"""
    base = np.zeros(3, dtype=np.float32) if pt_ref is None else pt_ref
    calibrate_offset = _get_calibrate_offset(arm)
    active = np.array(
        [
            base[0] - GRIPPER_OFFSET + 0.03,
            base[1] + calibrate_offset,
            base[2] + 0.05,
            0,
            PITCH_NORMAL_OBJECT,
            0,
            GRIPPER_NORMAL_OBJECT,
        ],
        dtype=np.float32,
    )
    return _make_arm_action(arm, active)


def make_pick_stop_action(pt_ref: Optional[np.ndarray], arm: str) -> Dict[str, np.ndarray]:
    """回撤一点抓回位置，抬高保证一个重力对抗，保持不动。"""
    base = np.zeros(3, dtype=np.float32) if pt_ref is None else pt_ref
    calibrate_offset = _get_calibrate_offset(arm)
    active = np.array(
        [
            base[0] - GRIPPER_OFFSET - 0.02,
            base[1] + calibrate_offset,
            base[2] + 0.07,
            0,
            PITCH_NORMAL_OBJECT,
            0,
            GRIPPER_NORMAL_OBJECT,
        ],
        dtype=np.float32,
    )
    return _make_arm_action(arm, active)


def make_pick_back_action(pt_ref: Optional[np.ndarray], arm: str) -> Dict[str, np.ndarray]:
    """夹住回到初始位置，保持不动。"""
    base = np.zeros(3, dtype=np.float32) if pt_ref is None else pt_ref
    active = np.array(
        [
            (base[0] - GRIPPER_OFFSET) / 4,
            0,
            (base[2] + 0.07) / 2 + 0.065,
            0,
            PITCH_NORMAL_OBJECT,
            0,
            GRIPPER_NORMAL_OBJECT,
        ],
        dtype=np.float32,
    )
    return _make_arm_action(arm, active)


def make_place_move_action(pt_ref: Optional[np.ndarray], arm: str) -> Dict[str, np.ndarray]:
    """保持抓取偏移到放置目标附近，保持不动。"""
    base = np.zeros(3, dtype=np.float32) if pt_ref is None else pt_ref
    calibrate_offset = _get_calibrate_offset(arm)
    active = np.array(
        [
            base[0] - GRIPPER_OFFSET - 0.04,
            base[1] + calibrate_offset,
            base[2] + Z_NORMAL_OBJECT + 0.1,
            0,
            PITCH_NORMAL_OBJECT,
            0,
            GRIPPER_NORMAL_OBJECT,
        ],
        dtype=np.float32,
    )
    return _make_arm_action(arm, active)


def make_place_robust_action(pt_ref: Optional[np.ndarray], arm: str) -> Dict[str, np.ndarray]:
    """执行向前移动，准备鲁棒放置位置。"""
    base = np.zeros(3, dtype=np.float32) if pt_ref is None else pt_ref
    calibrate_offset = _get_calibrate_offset(arm)
    active = np.array(
        [
            base[0] - GRIPPER_OFFSET,
            base[1] + calibrate_offset,
            base[2] + Z_NORMAL_OBJECT + 0.1,
            0,
            PITCH_NORMAL_OBJECT,
            0,
            GRIPPER_NORMAL_OBJECT,
        ],
        dtype=np.float32,
    )
    return _make_arm_action(arm, active)


def make_down_action(pt_ref: Optional[np.ndarray], arm: str) -> Dict[str, np.ndarray]:
    """下降到放置位置，保持不动。"""
    base = np.zeros(3, dtype=np.float32) if pt_ref is None else pt_ref
    calibrate_offset = _get_calibrate_offset(arm)
    active = np.array(
        [
            base[0] - GRIPPER_OFFSET,
            base[1] + calibrate_offset,
            base[2] + Z_NORMAL_OBJECT + 0.02,
            0,
            PITCH_NORMAL_OBJECT,
            0,
            GRIPPER_NORMAL_OBJECT,
        ],
        dtype=np.float32,
    )
    return _make_arm_action(arm, active)


def make_open_action(pt_ref: Optional[np.ndarray], arm: str) -> Dict[str, np.ndarray]:
    """夹爪张开放置"""
    base = np.zeros(3, dtype=np.float32) if pt_ref is None else pt_ref
    calibrate_offset = _get_calibrate_offset(arm)
    active = np.array(
        [
            base[0] - GRIPPER_OFFSET,
            base[1] + calibrate_offset,
            base[2] + Z_NORMAL_OBJECT + 0.02,
            0,
            PITCH_NORMAL_OBJECT,
            0,
            OPEN,
        ],
        dtype=np.float32,
    )
    return _make_arm_action(arm, active)


def make_place_stop_action(pt_ref: Optional[np.ndarray], arm: str) -> Dict[str, np.ndarray]:
    """回撤一点放置位置，保持不动。"""
    base = np.zeros(3, dtype=np.float32) if pt_ref is None else pt_ref
    calibrate_offset = _get_calibrate_offset(arm)
    active = np.array(
        [
            base[0] - GRIPPER_OFFSET - 0.05,
            base[1] + calibrate_offset,
            base[2] + Z_NORMAL_OBJECT + 0.07,
            0,
            PITCH_NORMAL_OBJECT,
            0,
            OPEN,
        ],
        dtype=np.float32,
    )
    return _make_arm_action(arm, active)


def make_release_action(pt_ref: Optional[np.ndarray], arm: str) -> Dict[str, np.ndarray]:
    """夹爪home位置张开放置"""
    active = np.array(
        [0, 0, 0.03, 0, 0, 0, OPEN],
        dtype=np.float32,
    )
    return _make_arm_action(arm, active)


def build_pick_normal_object_sequence(pt_ref: Optional[np.ndarray], arm: str):
    """返回抓取动作序列，不执行。"""
    return [
        make_pick_move_action(pt_ref, arm),
        make_pick_robust_action(pt_ref, arm),
        make_close_action(pt_ref, arm),
        make_pick_stop_action(pt_ref, arm),
        make_pick_back_action(pt_ref, arm),
    ]


def build_place_normal_object_sequence(pt_ref: Optional[np.ndarray], arm: str):
    """返回放置动作序列，不执行。"""
    return [
        make_place_move_action(pt_ref, arm),
        make_place_robust_action(pt_ref, arm),
        make_down_action(pt_ref, arm),
        make_open_action(pt_ref, arm),
        make_place_stop_action(pt_ref, arm),
    ]
