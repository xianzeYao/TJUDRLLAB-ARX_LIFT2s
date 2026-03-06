# from nav_utils import depth_to_meters, get_key, extract_actions, merge_forward_actions, path_to_actions, refine_trajectory_strict
# from qwen3_vl_8b_tool import predict_point_from_rgb
from arx_pointing import predict_multi_points_from_rgb

import numpy as np
import threading
import time
import math
from pathlib import Path
import cv2

from arm_control.msg._pos_cmd import PosCmd

import json
import sys
import select
import termios
sys.path.append("../ARX_Realenv/ROS2")  # noqa

from arx_ros2_env import ARXRobotEnv

from typing import Literal, Tuple, List

WORKSPACE = Path(__file__).resolve().parent.parent
DEFAULT_INTRINSICS = WORKSPACE / "ARX_Realenv/Tools/instrinsics_camerah.json"
DEFAULT_LEFT_EXTRINSICS = WORKSPACE / \
    "ARX_Realenv/Tools/extrinsics_cam_h_left.json"
DEFAULT_RIGHT_EXTRINSICS = WORKSPACE / \
    "ARX_Realenv/Tools/extrinsics_cam_h_right.json"

BIAS_REF2CAM = np.array([0.0, 0.24, 0.0, 0.0])

# sweep distance
SWEEP_DISTANCE = 0.55


def load_intrinsics(path: Path | str | None = None) -> np.ndarray:
    """读取 3x3 相机内参矩阵。"""
    intr_path = Path(path) if path else DEFAULT_INTRINSICS
    data = json.loads(intr_path.read_text())
    K = np.asarray(data["camera_matrix"], dtype=np.float64)
    if K.shape != (3, 3):
        raise ValueError(f"内参矩阵形状异常: {K.shape}")
    return K


def _load_cam2ref_matrix(ext_path: Path) -> np.ndarray:
    payload = json.loads(ext_path.read_text())
    T = np.asarray(payload.get("T_cam2ref"), dtype=np.float64)
    if T.shape != (4, 4):
        raise ValueError(f"外参矩阵形状异常: {T.shape}")
    return T


def load_cam2ref(
    path: Path | str | None = None,
    side: Literal["left", "right"] | None = None,
) -> np.ndarray | tuple[np.ndarray, np.ndarray]:
    """
    读取 T_cam2ref。

    规则:
    - 传 `path`：返回该文件的单个 4x4 矩阵；
    - 不传 `path` 且传 `side`：返回对应 side 的默认 4x4 矩阵；
    - 默认（path/side 都不传）：返回 (T_left, T_right) 两个 4x4 矩阵。
    """
    if path is not None:
        return _load_cam2ref_matrix(Path(path))

    if side == "left":
        return _load_cam2ref_matrix(DEFAULT_LEFT_EXTRINSICS)
    if side == "right":
        return _load_cam2ref_matrix(DEFAULT_RIGHT_EXTRINSICS)
    if side is not None:
        raise ValueError(f"side must be 'left' or 'right', got: {side!r}")

    return (
        _load_cam2ref_matrix(DEFAULT_LEFT_EXTRINSICS),
        _load_cam2ref_matrix(DEFAULT_RIGHT_EXTRINSICS),
    )


def get_arx_color_depth(arx: ARXRobotEnv):
    frames = arx.get_camera(target_size=(640, 480), return_status=False)
    color = frames.get("camera_h_color")
    depth = frames.get("camera_h_aligned_depth_to_color")
    return color, depth


def depth_to_meters(raw_depth: float) -> float:
    """兼容毫米与米的深度值。"""
    if not np.isfinite(raw_depth) or raw_depth < 0:
        raise ValueError(f"无效深度值: {raw_depth}")
    if raw_depth > 10.0:
        return float(raw_depth) / 1000.0
    return float(raw_depth)


def pixel_to_pw(pixel, depth, robot_part="center"):
    assert robot_part in ["center", "left", "right"]
    u, v = pixel
    z = depth_to_meters(float(depth[int(v), int(u)]))
    while z <= 0:
        print("Invalide depth!")
        depth = get_arx_color_depth()[1]
        z = depth_to_meters(float(depth[int(v), int(u)]))
    # 像素 → 相机坐标
    K = load_intrinsics()
    fx, fy = K[0, 0], K[1, 1]
    cx, cy = K[0, 2], K[1, 2]
    x = (u - cx) * z / fx
    y = (v - cy) * z / fy
    Pc = np.array([x, y, z, 1.0], dtype=np.float64)

    # 相机坐标 → 世界坐标
    T_CAM2L = load_cam2ref(side="left")
    T_CAM2R = load_cam2ref(side="right")
    if robot_part == "center":
        Pw = T_CAM2L @ Pc
        Pw = Pw + BIAS_REF2CAM
    if robot_part == "left":
        Pw = T_CAM2L @ Pc
    if robot_part == "right":
        Pw = T_CAM2R @ Pc

    return Pw[:3]


def path_to_actions(
    path: List[Tuple[int, int]],
    init_yaw: float = 0.0,
):
    actions = []
    cur_yaw = init_yaw

    for i in range(1, len(path)):
        x0, y0 = path[i - 1]
        x1, y1 = path[i]

        dx = x1 - x0
        dy = y1 - y0

        target_yaw = math.atan2(dy, dx)
        d_yaw = target_yaw - cur_yaw

        # 归一化到 [-pi, pi]
        while d_yaw > math.pi:
            d_yaw -= 2 * math.pi
        while d_yaw < -math.pi:
            d_yaw += 2 * math.pi

        dist = math.hypot(dx, dy)

        if abs(d_yaw) > 1e-3:
            actions.append(("rotate", -d_yaw))
            cur_yaw = target_yaw

        if dist > 1e-3:
            actions.append(("forward", dist))

    return actions


def auto_point_nav(arx: ARXRobotEnv, goal: str):
    # get color and depth
    color, depth = get_arx_color_depth(arx)

    # get goal points
    points, generated_content = predict_multi_points_from_rgb(
        color,
        text_prompt=goal,
        all_prompt=None,
        base_url="http://172.28.102.11:22002/v1",
        model_name="Embodied-R1.5-SFT-0128",
        api_key="EMPTY",
        assume_bgr=False,
        return_raw=True
    )

    # visualize goal points
    for point in points:
        cv2.circle(
            color,
            center=(int(point[0]), int(point[1])),
            radius=5,
            color=(0, 0, 255),
            thickness=-1  # -1 表示实心圆
        )

    cv2.imwrite("Testdata4Nav/goal_detect.png", color)

    # transform from pixel point to world point
    all_goals = []
    for point in points:
        goal_pw = pixel_to_pw(point, depth)
        all_goals.append(goal_pw)

    # sort according to the distance
    sorted_goals = sorted(
        all_goals,
        key=lambda p: math.hypot(p[0], p[1])
    )

    path = [(p[0], -p[1]) for p in sorted_goals]
    path.insert(0, (0.0, 0.0))

    # path to action
    print(path)
    actions = path_to_actions(path)
    print(actions)

    # move to goal
    for action, action_content in actions:
        if action == "forward":
            if action_content < SWEEP_DISTANCE:
                continue
            arx.step_base(vx=0.5, vy=0.0, vz=0.0, duration=(
                action_content-SWEEP_DISTANCE)/0.06)
        elif action == "rotate":
            if action_content <= 0:
                arx.step_base(vx=0.0, vy=0.0, vz=-0.5, duration=float(
                    (max(-action_content-0.1, 0.0)/(math.pi / 20.6))))
            else:
                arx.step_base(vx=0.0, vy=0.0, vz=0.5, duration=float(
                    (action_content/(math.pi / 20.6))))

# ToDO: check if there is need to add


def reliable_model_predict(
    predict_time,
    color,
    text_prompt,
    all_prompt,
    base_url="http://172.28.102.11:22002/v1",
    model_name="Embodied-R1.5-SFT-0128",
    api_key="EMPTY",
    assume_bgr=False
):
    all_prediction = []
    for _ in range(predict_time):
        points, generated_content = predict_multi_points_from_rgb(
            color,
            text_prompt=text_prompt,
            all_prompt=all_prompt,
            base_url=base_url,
            model_name=model_name,
            api_key=api_key,
            assume_bgr=assume_bgr,
            return_raw=True
        )
        all_prediction.append(generated_content)
    print(all_prediction)

    true_count = all_prediction.count("True")
    false_count = all_prediction.count("False")
    if true_count > false_count:
        return True
    else:
        return False


def wait_until_see_goal(arx: ARXRobotEnv, goal: str = "paper ball", max_waiting_time: int = 60):
    prompt = (
        f'Is there an object matching this description: "{goal}"? '
        "Output only True or False."
    )
    # prompt = f"""Is there {goal} in the picture? If there is no {goal}, output 'False'; if there is {goal}, output 'True'."""
    end_signal = False
    start_time = time.time()
    while not end_signal and time.time() - start_time < max_waiting_time:
        color, depth = get_arx_color_depth(arx)
        end_signal = reliable_model_predict(
            predict_time=5,
            color=color,
            text_prompt=goal,
            all_prompt=prompt
        )
        points, generated_content = predict_multi_points_from_rgb(
            color,
            text_prompt=goal,
            all_prompt=prompt,
            base_url="http://172.28.102.11:22002/v1",
            model_name="Embodied-R1.5-SFT-0128",
            api_key="EMPTY",
            assume_bgr=False,
            return_raw=True
        )
        if generated_content == "True":
            end_signal = True
        time.sleep(0.5)

    return end_signal


def get_key_nonblock():
    """非阻塞读取键盘，如果没有按键返回None"""
    dr, dw, de = select.select([sys.stdin], [], [], 0)
    if dr:
        return sys.stdin.read(1)
    return None


def init_keyboard():
    fd = sys.stdin.fileno()
    old_settings = termios.tcgetattr(fd)
    new_settings = termios.tcgetattr(fd)
    new_settings[3] = new_settings[3] & ~termios.ICANON & ~termios.ECHO
    termios.tcsetattr(fd, termios.TCSADRAIN, new_settings)
    return old_settings


def restore_keyboard(old_settings):
    termios.tcsetattr(sys.stdin, termios.TCSADRAIN, old_settings)


def nav_sweep_floor(arx: ARXRobotEnv, goal: str = "trash objects (e.g., white paper balls)"):
    old_settings = init_keyboard()
    start_height = 12.0
    arx.step_lift(start_height)

    print("Start continuous detection... press 'q' to stop.")

    while True:

        # 检查是否按下q
        key = get_key_nonblock()
        if key == 'q':
            print("Stop signal received.")
            break

        # 检测垃圾
        detect_goal = wait_until_see_goal(
            arx,
            goal=goal,
            max_waiting_time=3
        )

        if detect_goal:
            print("Rubbish detected, navigating...")
            auto_point_nav(
                arx,
                goal=goal
            )
        else:
            print("No rubbish, continue scanning...")

        time.sleep(0.5)

    restore_keyboard(old_settings)


if __name__ == "__main__":

    try:
        arx = ARXRobotEnv(
            duration_per_step=1.0 / 20.0,
            min_steps=20,
            max_v_xyz=0.15, max_a_xyz=0.20,
            max_v_rpy=0.5, max_a_rpy=1.00,
            camera_type="all",
            camera_view=("camera_h",),
            img_size=(640, 480),
        )

        arx.reset()
        nav_sweep_floor(arx)

    finally:
        arx.close()
