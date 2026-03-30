"""杯子抓取/放置动作模板，所有函数都只生成动作，不直接执行。"""

import numpy as np
from typing import Dict, Optional

CLOSE = 0.0
OPEN = -3.3
GRIPPER_OFFSET = 0.15
GRIPPER_CUP = -2.2
# GRIPPER_CUP = -2.2 or -0.5 # for pick
Z_CUP = 0.045
CALIBRATE_OFFSET_LEFT = 0.015
CALIBRATE_OFFSET_RIGHT = 0.01


def _make_arm_action(arm: str, active: np.ndarray) -> Dict[str, np.ndarray]:
    """把单臂 7 维动作包装成 `ARXRobotEnv` 需要的 action 字典。"""
    if arm == "left":
        return {"left": active}
    if arm == "right":
        return {"right": active}
    raise ValueError(f"arm must be 'left' or 'right', got: {arm!r}")


def _get_calibrate_offset(arm: str) -> float:
    """返回左右臂各自的 Y 方向标定补偿。"""
    if arm == "left":
        return CALIBRATE_OFFSET_LEFT
    if arm == "right":
        return CALIBRATE_OFFSET_RIGHT
    raise ValueError(f"arm must be 'left' or 'right', got: {arm!r}")


def make_pick_move_action(pt_ref: Optional[np.ndarray], arm: str) -> Dict[str, np.ndarray]:
    """生成抓取前的接近动作。`pt_ref` 预期为参考系下的 3D 点。"""
    base = np.zeros(3, dtype=np.float32) if pt_ref is None else pt_ref
    calibrate_offset = _get_calibrate_offset(arm)
    active = np.array(
        [base[0] - GRIPPER_OFFSET-0.03,
            base[1] + calibrate_offset, base[2]+0.01, 0, 0, 0, OPEN],
        dtype=np.float32,
    )
    return _make_arm_action(arm, active)


def make_pick_robust_action(pt_ref: Optional[np.ndarray], arm: str) -> Dict[str, np.ndarray]:
    """生成向前探入的鲁棒抓取动作。"""
    base = np.zeros(3, dtype=np.float32) if pt_ref is None else pt_ref
    calibrate_offset = _get_calibrate_offset(arm)
    active = np.array(
        [base[0] - GRIPPER_OFFSET + 0.03,
            base[1] + calibrate_offset, base[2]+0.01, 0, 0, 0, OPEN],
        dtype=np.float32,
    )
    return _make_arm_action(arm, active)


def make_close_action(pt_ref: Optional[np.ndarray], arm: str) -> Dict[str, np.ndarray]:
    """生成在当前抓取位姿下的夹爪闭合动作。"""
    base = np.zeros(3, dtype=np.float32) if pt_ref is None else pt_ref
    calibrate_offset = _get_calibrate_offset(arm)
    active = np.array(
        [base[0] - GRIPPER_OFFSET + 0.03, base[1] + calibrate_offset,
            base[2]+0.01, 0, 0, 0, GRIPPER_CUP],
        dtype=np.float32,
    )
    return _make_arm_action(arm, active)


def make_pick_stop_action(pt_ref: Optional[np.ndarray], arm: str) -> Dict[str, np.ndarray]:
    """生成抓取后的抬升回撤动作。"""
    base = np.zeros(3, dtype=np.float32) if pt_ref is None else pt_ref
    calibrate_offset = _get_calibrate_offset(arm)
    active = np.array(
        [base[0] - GRIPPER_OFFSET-0.02, base[1] + calibrate_offset,
         base[2] + Z_CUP, 0, 0, 0, GRIPPER_CUP],
        dtype=np.float32,
    )
    return _make_arm_action(arm, active)


def make_pick_back_action(pt_ref: Optional[np.ndarray], arm: str) -> Dict[str, np.ndarray]:
    """生成夹住杯子后回到更安全中间位的动作。"""
    base = np.zeros(3, dtype=np.float32) if pt_ref is None else pt_ref
    calibrate_offset = _get_calibrate_offset(arm)
    active = np.array(
        [(base[0] - GRIPPER_OFFSET)/4,
         0, (base[2] + Z_CUP)/2+0.05, 0, 0, 0, GRIPPER_CUP],
        dtype=np.float32,
    )
    return _make_arm_action(arm, active)


def make_place_move_action(pt_ref: Optional[np.ndarray], arm: str) -> Dict[str, np.ndarray]:
    """生成带着杯子接近放置点的动作。"""
    base = np.zeros(3, dtype=np.float32) if pt_ref is None else pt_ref
    calibrate_offset = _get_calibrate_offset(arm)
    active = np.array(
        [base[0] - GRIPPER_OFFSET-0.04, base[1] + calibrate_offset,
         base[2] + Z_CUP+0.1, 0, 0, 0, GRIPPER_CUP],
        dtype=np.float32,
    )
    return _make_arm_action(arm, active)


def make_place_robust_action(pt_ref: Optional[np.ndarray], arm: str) -> Dict[str, np.ndarray]:
    """生成向前探入的鲁棒放置动作。"""
    base = np.zeros(3, dtype=np.float32) if pt_ref is None else pt_ref
    calibrate_offset = _get_calibrate_offset(arm)
    active = np.array(
        [base[0] - GRIPPER_OFFSET, base[1] + calibrate_offset,
         base[2] + Z_CUP+0.1, 0, 0, 0, GRIPPER_CUP],
        dtype=np.float32,
    )
    return _make_arm_action(arm, active)


def make_down_action(pt_ref: Optional[np.ndarray], arm: str) -> Dict[str, np.ndarray]:
    """生成下降到最终放置高度的动作。"""
    base = np.zeros(3, dtype=np.float32) if pt_ref is None else pt_ref
    calibrate_offset = _get_calibrate_offset(arm)
    active = np.array(
        [base[0] - GRIPPER_OFFSET, base[1] + calibrate_offset,
         base[2] + Z_CUP+0.02, 0, 0, 0, GRIPPER_CUP],
        dtype=np.float32,
    )
    return _make_arm_action(arm, active)


def make_open_action(pt_ref: Optional[np.ndarray], arm: str) -> Dict[str, np.ndarray]:
    """生成放置点位的夹爪张开动作。"""
    base = np.zeros(3, dtype=np.float32) if pt_ref is None else pt_ref
    calibrate_offset = _get_calibrate_offset(arm)
    active = np.array(
        [base[0] - GRIPPER_OFFSET, base[1] + calibrate_offset,
            base[2] + Z_CUP+0.02, 0, 0, 0, OPEN],
        dtype=np.float32,
    )
    return _make_arm_action(arm, active)


def make_place_stop_action(pt_ref: Optional[np.ndarray], arm: str) -> Dict[str, np.ndarray]:
    """生成放置完成后的回撤动作。"""
    base = np.zeros(3, dtype=np.float32) if pt_ref is None else pt_ref
    calibrate_offset = _get_calibrate_offset(arm)
    active = np.array(
        [base[0] - GRIPPER_OFFSET - 0.05, base[1] + calibrate_offset,
         base[2] + Z_CUP+0.07, 0, 0, 0, OPEN],
        dtype=np.float32,
    )
    return _make_arm_action(arm, active)


def make_release_action(pt_ref: Optional[np.ndarray], arm: str) -> Dict[str, np.ndarray]:
    """生成 home 附近的夹爪张开动作，多用于收尾。"""
    active = np.array([0, 0, 0, 0, 0, 0, OPEN], dtype=np.float32)
    return _make_arm_action(arm, active)


def build_pick_cup_sequence(pt_ref: Optional[np.ndarray], arm: str):
    """返回完整抓杯动作序列，供上层逐步喂给 `step_smooth_eef()`。"""
    return [
        make_pick_move_action(pt_ref, arm),
        make_pick_robust_action(pt_ref, arm),
        make_close_action(pt_ref, arm),
        make_pick_stop_action(pt_ref, arm),
        make_pick_back_action(pt_ref, arm),
    ]


def build_place_cup_sequence(pt_ref: Optional[np.ndarray], arm: str):
    """返回完整放杯动作序列，供上层逐步喂给 `step_smooth_eef()`。"""
    return [
        make_place_move_action(pt_ref, arm),
        make_place_robust_action(pt_ref, arm),
        make_down_action(pt_ref, arm),
        make_open_action(pt_ref, arm),
        make_place_stop_action(pt_ref, arm),
    ]
