from __future__ import annotations

from typing import Dict

import numpy as np


CLOSE = 0.0
OPEN = -3.3
GRIPPER_X_FRONT_OFFSET = 0.15
GRIPPER_X_CLOSE_OFFSET = 0.07
GRIPPER_Y_OUT_OFFSET = 0.13
GRIPPER_Y_CLOSE_OFFSET = 0.14
GRIPPER_Z_LIFT_4 = 0.1
GRIPPER_Z_LIFT = 0.26



# 夹爪绕Z轴转
GRIPPER_Z_rot = np.deg2rad(15)

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
            left_ref[0] + GRIPPER_X_CLOSE_OFFSET,
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
            right_ref[0] + GRIPPER_X_CLOSE_OFFSET,
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
            left_ref[0] + GRIPPER_X_CLOSE_OFFSET,
            left_ref[1] + GRIPPER_Y_OUT_OFFSET- GRIPPER_Y_CLOSE_OFFSET,
            left_ref[2],
            0.0,
            0.0,
            0.0 - GRIPPER_Z_rot,
            CLOSE,
        ],
        dtype=np.float32,
    )
    right = np.array(
        [
            right_ref[0] + GRIPPER_X_CLOSE_OFFSET,
            right_ref[1] - GRIPPER_Y_OUT_OFFSET + GRIPPER_Y_CLOSE_OFFSET,
            right_ref[2],
            0.0,
            0.0,
            0.0 + GRIPPER_Z_rot,
            CLOSE,
        ],
        dtype=np.float32,
    )
    return _make_dual_arm_action(left, right)


def make_pick_stop_action(
    left_ref: np.ndarray,
    right_ref: np.ndarray,
) -> Dict[str, np.ndarray]:
    """第4阶段：抱取后回撤到安全位（预留抬升/减振）。"""
    left = np.array(
        [
            (left_ref[0] + GRIPPER_X_CLOSE_OFFSET) * 0.5,
            left_ref[1] + GRIPPER_Y_OUT_OFFSET- GRIPPER_Y_CLOSE_OFFSET,
            left_ref[2] + GRIPPER_Z_LIFT_4,
            0.0,
            0.0,
            0.0 - GRIPPER_Z_rot,
            CLOSE,
        ],
        dtype=np.float32,
    )
    right = np.array(
        [
            (right_ref[0] + GRIPPER_X_CLOSE_OFFSET) * 0.5,
            right_ref[1] - GRIPPER_Y_OUT_OFFSET + GRIPPER_Y_CLOSE_OFFSET,
            right_ref[2] + GRIPPER_Z_LIFT_4,
            0.0,
            0.0,
            0.0 + GRIPPER_Z_rot,
            CLOSE,
        ],
        dtype=np.float32,
    )
    return _make_dual_arm_action(left, right)


def make_pick_back_action(
    left_ref: np.ndarray,
    right_ref: np.ndarray,
) -> Dict[str, np.ndarray]:
    """第5阶段：抬高以免挡到摄像头。"""
    left = np.array(
        [
            (left_ref[0] + GRIPPER_X_CLOSE_OFFSET) * 0.5,
            left_ref[1] + GRIPPER_Y_OUT_OFFSET- GRIPPER_Y_CLOSE_OFFSET,
            left_ref[2] + GRIPPER_Z_LIFT,
            0.0,
            0.0,
            0.0 - GRIPPER_Z_rot,
            CLOSE,
        ],
        dtype=np.float32,
    )
    right = np.array(
        [
            (right_ref[0] + GRIPPER_X_CLOSE_OFFSET) * 0.5,
            right_ref[1] - GRIPPER_Y_OUT_OFFSET + GRIPPER_Y_CLOSE_OFFSET,
            right_ref[2] + GRIPPER_Z_LIFT,
            0.0,
            0.0,
            0.0 + GRIPPER_Z_rot,
            CLOSE,
        ],
        dtype=np.float32,
    )
    return _make_dual_arm_action(left, right)


def make_place_move_action(
    left_ref: np.ndarray,
    right_ref: np.ndarray,
    left_pose: np.ndarray,
    right_pose: np.ndarray,
) -> Dict[str, np.ndarray]:
    """第1阶段：物体移动到放置点前方。"""
    left = np.array(
        [
            left_pose[0],
            left_pose[1],
            left_ref[2] + 0.2,
            0.0,
            0.0,
            left_pose[5],
            CLOSE,
        ],
        dtype=np.float32,
    )
    right = np.array(
        [
            right_pose[0],
            right_pose[1],
            right_ref[2] + 0.2,
            0.0,
            0.0,
            right_pose[5],
            CLOSE,
        ],
        dtype=np.float32,
    )
    return _make_dual_arm_action(left, right)


def make_place_robust_action(
    left_ref: np.ndarray,
    right_ref: np.ndarray,
    left_pose: np.ndarray,
    right_pose: np.ndarray,
) -> Dict[str, np.ndarray]:
    """第2阶段：物体向前推入放置点。"""
    left = np.array(
        [
            left_ref[0] - 0.07,
            left_pose[1],
            left_ref[2] + 0.2,
            0.0,
            0.0,
            left_pose[5],
            CLOSE,
        ],
        dtype=np.float32,
    )
    right = np.array(
        [
            right_ref[0] - 0.07,
            right_pose[1],
            right_ref[2] + 0.2,
            0.0,
            0.0,
            right_pose[5],
            CLOSE,
        ],
        dtype=np.float32,
    )
    return _make_dual_arm_action(left, right)


def make_down_action(
    left_ref: np.ndarray,
    right_ref: np.ndarray,
    left_pose: np.ndarray,
    right_pose: np.ndarray,
) -> Dict[str, np.ndarray]:
    """第3阶段：物体放下。"""
    left = np.array(
        [
            left_ref[0] - 0.08,
            left_pose[1],
            left_ref[2] + 0.1,
            0.0,
            0.0,
            left_pose[5],
            CLOSE,
        ],
        dtype=np.float32,
    )
    right = np.array(
        [
            right_ref[0] - 0.08,
            right_pose[1],
            right_ref[2] + 0.1,
            0.0,
            0.0,
            right_pose[5],
            CLOSE,
        ],
        dtype=np.float32,
    )
    return _make_dual_arm_action(left, right)


def make_out_action(
    left_ref: np.ndarray,
    right_ref: np.ndarray,
    left_pose: np.ndarray,
    right_pose: np.ndarray,
) -> Dict[str, np.ndarray]:
    """第4阶段：张开双臂释放物体。"""
    left = np.array(
        [
            left_ref[0] - 0.08,
            left_pose[1] + GRIPPER_Y_OUT_OFFSET,
            left_ref[2] + 0.1,
            0.0,
            0.0,
            0.0,
            CLOSE,
        ],
        dtype=np.float32,
    )
    right = np.array(
        [
            right_ref[0] - 0.08,
            right_pose[1]- GRIPPER_Y_OUT_OFFSET,
            right_ref[2] + 0.1,
            0.0,
            0.0,
            0.0,
            CLOSE,
        ],
        dtype=np.float32,
    )
    return _make_dual_arm_action(left, right)


def make_place_stop_action(
    left_ref: np.ndarray,
    right_ref: np.ndarray,
    left_pose: np.ndarray,
    right_pose: np.ndarray,
) -> Dict[str, np.ndarray]:
    """第5阶段：双臂收回。"""
    left = np.array(
        [
            0.1,
            left_pose[1] + GRIPPER_Y_OUT_OFFSET,
            left_ref[2] + 0.2,
            0.0,
            0.0,
            0.0,
            CLOSE,
        ],
        dtype=np.float32,
    )
    right = np.array(
        [
            0.1,
            right_pose[1]- GRIPPER_Y_OUT_OFFSET,
            right_ref[2] + 0.2,
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
    left_pose: np.ndarray,
    right_pose: np.ndarray,
) -> list[Dict[str, np.ndarray]]:
    """构建双臂 hug 放置动作序列（五阶段）。"""
    left_pose[1] -= 0.01
    right_pose[1] +=0.01
    return [
        make_place_move_action(left_ref, right_ref, left_pose, right_pose),
        make_place_robust_action(left_ref, right_ref, left_pose, right_pose),
        make_down_action(left_ref, right_ref, left_pose, right_pose),
        make_out_action(left_ref, right_ref, left_pose, right_pose),
        make_place_stop_action(left_ref, right_ref, left_pose, right_pose),
    ]
