import numpy as np
from typing import Dict, Optional

SWAP_OFFSET = 0.05
GRIPPER_OFFSET = 0.15


def make_swap_move_action(pt_ref: Optional[np.ndarray],) -> Dict[str, np.ndarray]:
    """左右臂靠近垃圾"""
    base = np.zeros(3, dtype=np.float32) if pt_ref is None else pt_ref
    action = {"left":  np.array(
        [base[0]-GRIPPER_OFFSET, base[1]+SWAP_OFFSET+0.04, 0.03, 0, 0, 0, 0.0],
        dtype=np.float32),
        "right": np.array(
            [base[0]-GRIPPER_OFFSET, base[1]-SWAP_OFFSET, 0.05, -0.2, 0, 0, 0.0], dtype=np.float32
    )}
    return action


def make_swap_left_action(pt_ref: Optional[np.ndarray],) -> Dict[str, np.ndarray]:
    """左臂不动，右臂左移"""
    base = np.zeros(3, dtype=np.float32) if pt_ref is None else pt_ref
    action = {"right":  np.array(
        [base[0]-GRIPPER_OFFSET+0.02, base[1] +
            SWAP_OFFSET+0.3, 0.0025, 0.4, 0, 0, 0.0],
        dtype=np.float32)}
    return action


def make_swap_lift_action(pt_ref: Optional[np.ndarray],) -> Dict[str, np.ndarray]:
    """左臂不动，右臂抬起"""
    base = np.zeros(3, dtype=np.float32) if pt_ref is None else pt_ref
    action = {"right":  np.array(
        [base[0]-GRIPPER_OFFSET+0.02, base[1] +
            SWAP_OFFSET+0.27, 0.15, 0, 0, 0, 0.0],
        dtype=np.float32)}
    return action


def make_swap_right_action(pt_ref: Optional[np.ndarray],) -> Dict[str, np.ndarray]:
    """左臂不动，右臂右平移"""
    base = np.zeros(3, dtype=np.float32) if pt_ref is None else pt_ref
    action = {"right":  np.array(
        [base[0]-GRIPPER_OFFSET+0.02, base[1]-SWAP_OFFSET, 0.15, 0.0, 0, 0, 0.0], dtype=np.float32)}
    return action


def make_swap_down_action(pt_ref: Optional[np.ndarray],) -> Dict[str, np.ndarray]:
    """左臂不动，右臂下移"""
    base = np.zeros(3, dtype=np.float32) if pt_ref is None else pt_ref
    action = {"right":  np.array(
        [base[0]-GRIPPER_OFFSET+0.02, base[1]-SWAP_OFFSET, 0.05, -0.2, 0, 0, 0.0],
        dtype=np.float32)}
    return action


def build_swap_sequence(pt_ref: Optional[np.ndarray]):
    """返回动作序列，不执行。"""
    result = []
    result.append(make_swap_move_action(pt_ref))
    for _ in range(3):
        result.append(make_swap_left_action(pt_ref))
        result.append(make_swap_lift_action(pt_ref))
        result.append(make_swap_right_action(pt_ref))
        result.append(make_swap_down_action(pt_ref))
    return result
