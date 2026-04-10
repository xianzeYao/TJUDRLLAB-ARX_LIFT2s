from __future__ import annotations

from typing import Dict

import numpy as np


CLOSE = 0.0
OPEN = -3.3
GRIPPER_X_FRONT_OFFSET = 0.15
GRIPPER_Y_OUT_OFFSET = 0.13
GRIPPER_Y_CLOSE_OFFSET = 0.14
GRIPPER_Z_LIFT = 0.1

CALIBRATE_OFFSET_LEFT = 0.015
CALIBRATE_OFFSET_RIGHT = 0.01


def _make_dual_arm_action(
    left_active: np.ndarray,
    right_active: np.ndarray,
) -> Dict[str, np.ndarray]:
    return {
        "left": left_active.astype(np.float32, copy=False),
        "right": right_active.astype(np.float32, copy=False),
    }


def make_pick_move_action(
    left_ref: np.ndarray,
    right_ref: np.ndarray,
) -> Dict[str, np.ndarray]:
    """第1阶段：双臂张开，移动到抱取前接近位。"""
    left = np.array(
        [
            left_ref[0] - GRIPPER_X_FRONT_OFFSET,
            left_ref[1] + GRIPPER_Y_OUT_OFFSET,
            left_ref[2],
            0.0,
            0.0,
            0.0,
            CLOSE,
        ],
        dtype=np.float32,
    )
    right = np.array(
        [
            right_ref[0] - GRIPPER_X_FRONT_OFFSET,
            right_ref[1] - GRIPPER_Y_OUT_OFFSET,
            right_ref[2],
            0.0,
            0.0,
            0.0,
            CLOSE,
        ],
        dtype=np.float32,
    )
    return _make_dual_arm_action(left, right)


def make_pick_robust_action(
    left_ref: np.ndarray,
    right_ref: np.ndarray,
) -> Dict[str, np.ndarray]:
    """第2阶段：双臂鲁棒接近（预留微调位）。"""
    left = np.array(
        [
            left_ref[0] + 0.05,
            left_ref[1] + GRIPPER_Y_OUT_OFFSET,
            left_ref[2],
            0.0,
            0.0,
            0.0,
            CLOSE,
        ],
        dtype=np.float32,
    )
    right = np.array(
        [
            right_ref[0] + 0.05,
            right_ref[1] - GRIPPER_Y_OUT_OFFSET,
            right_ref[2],
            0.0,
            0.0,
            0.0,
            CLOSE,
        ],
        dtype=np.float32,
    )
    return _make_dual_arm_action(left, right)


def make_close_action(
    left_ref: np.ndarray,
    right_ref: np.ndarray,
) -> Dict[str, np.ndarray]:
    """第3阶段：双臂同步闭合，执行抱取。"""
    left = np.array(
        [
            left_ref[0] + 0.03,
            left_ref[1] + GRIPPER_Y_OUT_OFFSET- GRIPPER_Y_CLOSE_OFFSET,
            left_ref[2],
            0.0,
            0.0,
            0.0,
            CLOSE,
        ],
        dtype=np.float32,
    )
    right = np.array(
        [
            right_ref[0] + 0.03,
            right_ref[1] - GRIPPER_Y_OUT_OFFSET + GRIPPER_Y_CLOSE_OFFSET,
            right_ref[2],
            0.0,
            0.0,
            0.0,
            CLOSE,
        ],
        dtype=np.float32,
    )
    return _make_dual_arm_action(left, right)


def make_pick_stop_action(
    left_ref: np.ndarray,
    right_ref: np.ndarray,
) -> Dict[str, np.ndarray]:
    """第4阶段：抱取后稳定停留（预留抬升/减振）。"""
    left = np.array(
        [
            left_ref[0] + 0.03,
            left_ref[1] + GRIPPER_Y_OUT_OFFSET- GRIPPER_Y_CLOSE_OFFSET,
            left_ref[2] + GRIPPER_Z_LIFT,
            0.0,
            0.0,
            0.0,
            CLOSE,
        ],
        dtype=np.float32,
    )
    right = np.array(
        [
            right_ref[0] + 0.03,
            right_ref[1] - GRIPPER_Y_OUT_OFFSET + GRIPPER_Y_CLOSE_OFFSET,
            right_ref[2] + GRIPPER_Z_LIFT,
            0.0,
            0.0,
            0.0,
            CLOSE,
        ],
        dtype=np.float32,
    )
    return _make_dual_arm_action(left, right)


def make_pick_back_action(
    left_ref: np.ndarray,
    right_ref: np.ndarray,
) -> Dict[str, np.ndarray]:
    """第5阶段：抱取后回撤到安全位。"""
    left = np.array(
        [
            (left_ref[0] + 0.03) * 0.75,
            left_ref[1] + GRIPPER_Y_OUT_OFFSET- GRIPPER_Y_CLOSE_OFFSET,
            left_ref[2] + GRIPPER_Z_LIFT,
            0.0,
            0.0,
            0.0,
            CLOSE,
        ],
        dtype=np.float32,
    )
    right = np.array(
        [
            (right_ref[0] + 0.03) * 0.75,
            right_ref[1] - GRIPPER_Y_OUT_OFFSET + GRIPPER_Y_CLOSE_OFFSET,
            right_ref[2] + GRIPPER_Z_LIFT,
            0.0,
            0.0,
            0.0,
            CLOSE,
        ],
        dtype=np.float32,
    )
    return _make_dual_arm_action(left, right)


def build_pick_hug_sequence(
    left_ref: np.ndarray,
    right_ref: np.ndarray,
) -> list[Dict[str, np.ndarray]]:
    """构建双臂 hug 抓取动作序列（五阶段）。"""
    return [
        make_pick_move_action(left_ref, right_ref),
        make_pick_robust_action(left_ref, right_ref),
        make_close_action(left_ref, right_ref),
        make_pick_stop_action(left_ref, right_ref),
        make_pick_back_action(left_ref, right_ref),
    ]


def build_place_hug_sequence(
    left_ref: np.ndarray,
    right_ref: np.ndarray,
) -> list[Dict[str, np.ndarray]]:
    """构建双臂 hug 放置动作序列。"""
    _ = left_ref, right_ref
    raise NotImplementedError("build_place_hug_sequence is not implemented yet.")
