import numpy as np
from typing import Dict, Optional

CALIBRATE_OFFSET_LEFT = 0.015
CALIBRATE_OFFSET_RIGHT = 0.01
GRIPPER_OFFSET = 0.14
APPROACH_Z_OFFSET = 0.02
PUSH_DISTANCE_Y = 0.05


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


def _get_push_sign(arm: str) -> float:
    if arm == "left":
        return 1.0
    if arm == "right":
        return -1.0
    raise ValueError(f"arm must be 'left' or 'right', got: {arm!r}")


def make_move_away_approach_action(
    pt_ref: Optional[np.ndarray],
    arm: str,
) -> Dict[str, np.ndarray]:
    base = np.zeros(3, dtype=np.float32) if pt_ref is None else pt_ref
    calibrate_offset = _get_calibrate_offset(arm)
    push_sign = _get_push_sign(arm)
    active = np.array(
        [
            base[0] - GRIPPER_OFFSET+0.055,
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
    base = np.zeros(3, dtype=np.float32) if pt_ref is None else pt_ref
    calibrate_offset = _get_calibrate_offset(arm)
    push_sign = _get_push_sign(arm)
    active = np.array(
        [
            base[0] - GRIPPER_OFFSET + 0.055,
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
    return [
        make_move_away_approach_action(pt_ref, arm),
        make_move_away_push_action(pt_ref, arm),
    ]
