import os
import time
import threading
import queue
from typing import Dict, Iterable, Optional, Literal

import numpy as np
import rclpy
from rclpy.executors import SingleThreadedExecutor
from rclpy.node import Node
from sensor_msgs.msg import Image

from arm_control.msg._pos_cmd import PosCmd
from arx5_arm_msg.msg._robot_cmd import RobotCmd
from arx5_arm_msg.msg._robot_status import RobotStatus

import cv2
from cv_bridge import CvBridge
from message_filters import Subscriber, ApproximateTimeSynchronizer


class RobotIO(Node):
    def __init__(self, camera_type: Literal["color", "depth", "all"] = "all",
                 camera_view: Iterable[str] = ("camera_l", "camera_h"),
                 save_video: bool = False, video_fps: float = 20.0,
                 save_dir: Optional[str] = None,
                 target_size: Optional[tuple[int, int]] = None,
                 video_name: Optional[str] = None):
        super().__init__('robot_io')
        self.bridge = CvBridge() if CvBridge is not None else None

        self.cmd_pub_l = self.create_publisher(RobotCmd, 'arm_cmd_l', 5)
        self.cmd_pub_r = self.create_publisher(RobotCmd, 'arm_cmd_r', 5)
        self.cmd_pub_base = self.create_publisher(PosCmd, 'ARX_VR_L', 5)

        self.latest_height = 0.0
        self.latest_status: Dict[str, Optional[RobotStatus]] = {
            "left": None, "right": None}
        self.latest_base: Optional[PosCmd] = None

        self.status_snapshot: Optional[Dict[str, Optional[RobotStatus]]] = None
        self.status_lock = threading.Lock()

        self.create_subscription(
            RobotStatus, 'arm_status_l', lambda msg: self._on_status('left', msg), 5)
        self.create_subscription(
            RobotStatus, 'arm_status_r', lambda msg: self._on_status('right', msg), 5)
        self.sta_cmd_sub = self.create_subscription(
            PosCmd, 'body_information', self._on_base_status, 1)

        self.camera_type = camera_type  # color/depth/all
        self.camera_view = list(camera_view) if camera_view else []
        self.default_save_video = bool(save_video)
        self.video_fps = float(video_fps)
        self.default_save_dir = os.fspath(save_dir) if save_dir else None
        self.default_target_size = tuple(target_size) if target_size else None
        self.video_name = str(video_name).strip() if video_name else None
        self.continuous_video = bool(
            self.default_save_video and self.default_save_dir)
        self.video_writers: Dict[tuple[str, str], cv2.VideoWriter] = {}
        self.video_shapes: Dict[tuple[str, str], tuple[int, int]] = {}
        self.cam_lock = threading.Lock()
        self.latest_images: Dict[str, Image] = {}
        self.subscribed_topics = []
        self.save_queue: "queue.Queue[Optional[tuple[str, str, float, np.ndarray, bool]]]" = queue.Queue(
        )
        self.saver_thread = threading.Thread(
            target=self._save_worker, daemon=True)
        self.saver_thread.start()

        subs: list[Subscriber] = []
        labels: list[str] = []

        types = [
            "color", "aligned_depth_to_color"] if camera_type == "all" else [camera_type]
        for cam in self.camera_view:
            for typ in types:
                if "aligned" in typ:
                    topic = f"/{cam}_namespace/{cam}/aligned_depth_to_color/image_raw"
                else:
                    topic = f"/{cam}_namespace/{cam}/{typ}/image_rect_raw"
                subs.append(Subscriber(self, Image, topic, qos_profile=5))
                labels.append(f"{cam}_{typ}")
                self.subscribed_topics.append(topic)
        self.labels = labels
        if subs:
            self.sync = ApproximateTimeSynchronizer(
                subs, queue_size=5, slop=0.02)
            self.sync.registerCallback(self._on_images_status)
            self.get_logger().info(
                f"Subscribed camera topics: {self.subscribed_topics}")
        else:
            self.get_logger().warn("No camera subscriptions configured.")
        self.get_logger().info(
            f"Init camera_view={self.camera_view}, types={types}")

    def _on_status(self, side: str, msg: RobotStatus):
        with self.status_lock:
            self.latest_status[side] = msg

    def _on_base_status(self, msg: PosCmd):
        with self.status_lock:
            self.latest_base = msg
            self.latest_height = float(msg.height)

    def _on_images_status(self, *msgs):

        with self.status_lock:
            snap = dict(self.latest_status)
            snap["base"] = self.latest_base
            self.status_snapshot = snap
        with self.cam_lock:
            for label, msg in zip(self.labels, msgs):
                self.latest_images[label] = msg
        if self.continuous_video and self.bridge is not None:
            for label, msg in zip(self.labels, msgs):
                img = self._decode_image_msg(
                    label, msg, self.default_target_size)
                if img is None:
                    continue
                stamp = getattr(msg, "header", None)
                if stamp:
                    ts = stamp.stamp.sec + stamp.stamp.nanosec * 1e-9
                else:
                    ts = time.time()
                self.save_queue.put(
                    (self.default_save_dir, label, ts, img, True))

    def send_base_msg(self, cmd: PosCmd):
        """Send base command."""
        if not rclpy.ok():
            try:
                self.get_logger().warn("ROS not ready, base command not sent")
            except Exception:
                pass
            return False
        sub_count = self.cmd_pub_base.get_subscription_count()
        if sub_count == 0:
            try:
                self.get_logger().warn(f"No base subscribers, base command not sent")
            except Exception:
                pass
            return False
        # track latest height even when publishing commands
        try:
            self.latest_height = float(cmd.height)
        except Exception:
            pass
        self.cmd_pub_base.publish(cmd)
        return True

    def send_control_msg(self, side: str, cmd: RobotCmd):
        pub = self.cmd_pub_l if side == "left" else self.cmd_pub_r
        if not rclpy.ok():
            try:
                self.get_logger().warn("ROS not ready, arm command not sent")
            except Exception:
                pass
            return False
        sub_count = pub.get_subscription_count()
        if sub_count == 0:
            try:
                self.get_logger().warn(
                    f"{side} no subscribers, arm command not sent")
            except Exception:
                pass
            return False
        pub.publish(cmd)
        return True

    def get_robot_status(self):
        # TODO:考虑删除此方法，直接使用 get_camera(return_status=True) 获取快照
        with self.status_lock:
            status = dict(self.latest_status)
            status["base"] = self.latest_base
            return status

    def get_camera(self, save_dir: Optional[str] = None, target_size: Optional[tuple[int, int]] = None,
                   save_video: Optional[bool] = None,
                   return_status: bool = False):
        """Return latest approx-synced camera frames, optional status snapshot."""
        if self.bridge is None:
            print("CvBridge not initialized, cannot decode images.")
            return (dict(), self.status_snapshot) if return_status else dict()
        save_as_video = self.default_save_video if save_video is None else bool(
            save_video)
        frames = dict()
        with self.cam_lock:
            items = list(self.latest_images.items())
        for key, msg in items:
            img = self._decode_image_msg(key, msg, target_size)
            if img is None:
                continue
            frames[key] = img
            if save_dir:
                stamp = getattr(msg, "header", None)
                if stamp:
                    ts = stamp.stamp.sec + stamp.stamp.nanosec * 1e-9
                else:
                    ts = time.time()
                if not self._should_skip_on_demand_save(save_dir, save_as_video):
                    self.save_queue.put(
                        (save_dir, key, ts, img, save_as_video))
        if return_status:
            with self.status_lock:
                # prefer snapshot if available; otherwise use latest
                if self.status_snapshot is not None:
                    snap = dict(self.status_snapshot)
                else:
                    snap = dict(self.latest_status)
                # ensure base present
                if "base" not in snap:
                    snap["base"] = self.latest_base
            return frames, snap
        return frames

    def _decode_image_msg(self, key: str, msg: Image, target_size: Optional[tuple[int, int]]) -> Optional[np.ndarray]:
        try:
            img = self.bridge.imgmsg_to_cv2(
                msg, desired_encoding='passthrough')
            if "depth" not in key:
                img = img[:, :, ::-1]
            if target_size:
                img = cv2.resize(img, target_size)
            return img
        except Exception as exc:  # pragma: no cover
            self.get_logger().warn(f"{key} decode failed: {exc}")
            return None

    def _should_skip_on_demand_save(self, save_dir: str, save_as_video: bool) -> bool:
        # Continuous video mode already writes every callback frame.
        if not save_as_video or not self.continuous_video:
            return False
        try:
            return os.path.abspath(os.fspath(save_dir)) == os.path.abspath(self.default_save_dir)
        except Exception:
            return False

    def get_camera_keys(self):
        with self.cam_lock:
            return list(self.latest_images.keys())

    def _save_worker(self):
        while True:
            task = self.save_queue.get()
            if task is None:
                self.save_queue.task_done()
                break
            save_dir, key, ts, img, save_video = task
            try:
                os.makedirs(save_dir, exist_ok=True)
                if save_video:
                    self._save_video_frame(save_dir, key, img)
                else:
                    base = os.path.join(save_dir, f"{key}_{ts}")
                    if "depth" in key:
                        np.save(base + ".npy", img)
                        vis = self._depth_to_vis(img)
                        cv2.imwrite(base + "_vis.png", vis,
                                    [cv2.IMWRITE_PNG_COMPRESSION, 0])
                    else:
                        cv2.imwrite(base + ".png", img,
                                    [cv2.IMWRITE_PNG_COMPRESSION, 0])
            except Exception as exc:  # pragma: no cover
                try:
                    self.get_logger().warn(f"save {key} failed: {exc}")
                except Exception:
                    pass
            finally:
                self.save_queue.task_done()
        self._release_video_writers()

    @staticmethod
    def _depth_to_vis(img: np.ndarray) -> np.ndarray:
        depth_float = img.astype(np.float32, copy=False)
        finite = depth_float[np.isfinite(depth_float)]
        if finite.size == 0:
            return np.zeros(depth_float.shape[:2], dtype=np.uint8)
        vmax = float(np.percentile(finite, 99))
        if vmax <= 1e-6:
            return np.zeros(depth_float.shape[:2], dtype=np.uint8)
        return cv2.convertScaleAbs(depth_float, alpha=255.0 / vmax)

    def _prepare_video_frame(self, key: str, img: np.ndarray) -> np.ndarray:
        if "depth" in key:
            frame = self._depth_to_vis(img)
        else:
            frame = img
        if frame.dtype != np.uint8:
            frame = cv2.convertScaleAbs(frame)
        if frame.ndim == 2:
            frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
        elif frame.ndim == 3 and frame.shape[2] == 1:
            frame = cv2.cvtColor(frame[:, :, 0], cv2.COLOR_GRAY2BGR)
        return frame

    def _get_video_writer(self, save_dir: str, key: str, frame: np.ndarray) -> cv2.VideoWriter:
        writer_key = (save_dir, key)
        h, w = frame.shape[:2]
        writer = self.video_writers.get(writer_key)
        shape = self.video_shapes.get(writer_key)
        if writer is not None and shape != (w, h):
            writer.release()
            writer = None
        if writer is None:
            filename = f"{self.video_name}_{key}.mp4" if self.video_name else f"{key}.mp4"
            out_path = os.path.join(save_dir, filename)
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            writer = cv2.VideoWriter(
                out_path, fourcc, self.video_fps, (w, h), True)
            if not writer.isOpened():
                raise RuntimeError(f"open video writer failed: {out_path}")
            self.video_writers[writer_key] = writer
            self.video_shapes[writer_key] = (w, h)
        return writer

    def _save_video_frame(self, save_dir: str, key: str, img: np.ndarray):
        frame = self._prepare_video_frame(key, img)
        writer = self._get_video_writer(save_dir, key, frame)
        writer.write(frame)

    def _release_video_writers(self):
        for writer in self.video_writers.values():
            try:
                writer.release()
            except Exception:
                pass
        self.video_writers.clear()
        self.video_shapes.clear()

    def stop_saver(self):
        self.save_queue.put(None)
        if getattr(self, "saver_thread", None) is not None:
            self.saver_thread.join(timeout=2.0)


def start_robot_io(camera_type: Literal["color", "depth", "all"] = "all",
                   camera_view: Iterable[str] = ("camera_l", "camera_h"),
                   save_video: bool = False,
                   video_fps: float = 20.0,
                   save_dir: Optional[str] = None,
                   target_size: Optional[tuple[int, int]] = None,
                   video_name: Optional[str] = None):
    node = RobotIO(camera_type=camera_type,
                   camera_view=camera_view,
                   save_video=save_video,
                   video_fps=video_fps,
                   save_dir=save_dir,
                   target_size=target_size,
                   video_name=video_name)
    executor = SingleThreadedExecutor()
    executor.add_node(node)
    t = threading.Thread(target=executor.spin, daemon=True)
    t.start()
    return node, executor, t


def build_observation(
    camera_all: Dict[str, Image] | Dict,
    status_all: Dict[str, object] | None,
    include_arm: bool = True,
    include_camera: bool = True,
    include_base: bool = True,
) -> Dict[str, np.ndarray]:
    """
    Pack status和相机到扁平观测字典。

    include_arm / include_camera / include_base 控制是否写入对应部分，默认全开。
    """
    obs: Dict[str, np.ndarray] = {}

    if include_arm and isinstance(status_all, dict):
        lstatus = status_all.get("left")
        rstatus = status_all.get("right")
        if lstatus is not None:
            obs["left_end_pos"] = np.array(lstatus.end_pos, dtype=np.float32)
            obs["left_joint_pos"] = np.array(
                lstatus.joint_pos, dtype=np.float32)
            obs["left_joint_cur"] = np.array(
                lstatus.joint_cur, dtype=np.float32)
            obs["left_joint_vel"] = np.array(
                lstatus.joint_vel, dtype=np.float32)
        if rstatus is not None:
            obs["right_end_pos"] = np.array(rstatus.end_pos, dtype=np.float32)
            obs["right_joint_pos"] = np.array(
                rstatus.joint_pos, dtype=np.float32)
            obs["right_joint_cur"] = np.array(
                rstatus.joint_cur, dtype=np.float32)
            obs["right_joint_vel"] = np.array(
                rstatus.joint_vel, dtype=np.float32)

    if include_base and isinstance(status_all, dict):
        base_status = status_all.get("base")
        if base_status is not None:
            obs["base_height"] = np.array(
                [base_status.height], dtype=np.float32)
            obs["base_wheel1"] = np.array(
                [base_status.temp_float_data[1]], dtype=np.float32)  # 后轮
            obs["base_wheel2"] = np.array(
                [base_status.temp_float_data[2]], dtype=np.float32)  # 右前
            obs["base_wheel3"] = np.array(
                [base_status.temp_float_data[3]], dtype=np.float32)  # 左前
    if include_camera:
        # Attach camera frames as numpy arrays
        for key, msg in (camera_all or {}).items():
            if msg is None:
                continue
            try:
                img = msg
                if isinstance(msg, np.ndarray):
                    img_np = msg
                else:
                    img_np = np.asarray(img)
                if "color" in key:
                    img_np = np.asarray(img_np, dtype=np.uint8)
                obs[key] = img_np
            except Exception:
                # Skip broken frame
                continue
    return obs


def _quat_normalize(q: np.ndarray) -> np.ndarray:
    n = float(np.linalg.norm(q))
    if n <= 1e-8:
        return np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float32)
    return (q / n).astype(np.float32)


def _quat_from_rpy(rpy: np.ndarray) -> np.ndarray:
    """将 roll/pitch/yaw 转成归一化四元数，便于做姿态插值。"""
    roll, pitch, yaw = [float(x) for x in rpy]
    cr = np.cos(roll * 0.5)
    sr = np.sin(roll * 0.5)
    cp = np.cos(pitch * 0.5)
    sp = np.sin(pitch * 0.5)
    cy = np.cos(yaw * 0.5)
    sy = np.sin(yaw * 0.5)
    qx = sr * cp * cy - cr * sp * sy
    qy = cr * sp * cy + sr * cp * sy
    qz = cr * cp * sy - sr * sp * cy
    qw = cr * cp * cy + sr * sp * sy
    return _quat_normalize(np.array([qx, qy, qz, qw], dtype=np.float32))


def _rpy_from_quat(q: np.ndarray) -> np.ndarray:
    """将插值后的四元数再转回 roll/pitch/yaw。"""
    q = _quat_normalize(q)
    x, y, z, w = [float(v) for v in q]
    sinr_cosp = 2.0 * (w * x + y * z)
    cosr_cosp = 1.0 - 2.0 * (x * x + y * y)
    roll = np.arctan2(sinr_cosp, cosr_cosp)

    sinp = 2.0 * (w * y - z * x)
    if abs(sinp) >= 1.0:
        pitch = np.sign(sinp) * (np.pi / 2.0)
    else:
        pitch = np.arcsin(sinp)

    siny_cosp = 2.0 * (w * z + x * y)
    cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
    yaw = np.arctan2(siny_cosp, cosy_cosp)
    return np.array([roll, pitch, yaw], dtype=np.float32)


def _quat_slerp(q0: np.ndarray, q1: np.ndarray, t: float) -> np.ndarray:
    """球面线性插值，避免直接线性插值 RPY 带来的姿态跳变。"""
    q0 = _quat_normalize(q0)
    q1 = _quat_normalize(q1)
    dot = float(np.dot(q0, q1))
    if dot < 0.0:
        q1 = -q1
        dot = -dot
    dot = float(np.clip(dot, -1.0, 1.0))
    if dot > 0.9995:
        return _quat_normalize(q0 + t * (q1 - q0))
    theta0 = float(np.arccos(dot))
    sin_theta0 = float(np.sin(theta0))
    theta = theta0 * t
    s0 = np.sin(theta0 - theta) / sin_theta0
    s1 = np.sin(theta) / sin_theta0
    return _quat_normalize(s0 * q0 + s1 * q1)


def _quat_angle(q0: np.ndarray, q1: np.ndarray) -> float:
    """计算两个姿态之间的最短角距离，单位是弧度。"""
    q0 = _quat_normalize(q0)
    q1 = _quat_normalize(q1)
    dot = float(np.dot(q0, q1))
    dot = float(np.clip(abs(dot), -1.0, 1.0))
    return 2.0 * float(np.arccos(dot))


def _trapezoid_params(d: float, v_max: float, a_max: float) -> tuple[float, float, float, float]:
    """根据位移、速度上限、加速度上限，计算梯形/三角速度曲线各阶段时长。"""
    if d <= 0.0:
        return 0.0, 0.0, 0.0, 0.0
    v_max = max(float(v_max), 1e-6)
    a_max = max(float(a_max), 1e-6)
    t_accel = v_max / a_max
    d_accel = 0.5 * a_max * t_accel * t_accel
    if d <= 2.0 * d_accel:
        # 位移太短时达不到 v_max，会退化成三角速度曲线。
        t_accel = np.sqrt(d / a_max)
        v_peak = a_max * t_accel
        total = 2.0 * t_accel
        return total, t_accel, 0.0, v_peak
    t_flat = (d - 2.0 * d_accel) / v_max
    total = 2.0 * t_accel + t_flat
    return total, t_accel, t_flat, v_max


def _trapezoid_position(t: float, d: float, v_max: float, a_max: float) -> float:
    """给定时刻 t，返回该梯形速度曲线下已经走过的距离。"""
    if d <= 0.0:
        return 0.0
    total, t_accel, t_flat, v_peak = _trapezoid_params(d, v_max, a_max)
    if t <= 0.0:
        return 0.0
    if t >= total:
        return d
    if t_flat <= 1e-9:
        # triangular
        if t <= t_accel:
            return 0.5 * a_max * t * t
        t_remain = total - t
        return d - 0.5 * a_max * t_remain * t_remain
    d_accel = 0.5 * a_max * t_accel * t_accel
    if t <= t_accel:
        return 0.5 * a_max * t * t
    if t <= t_accel + t_flat:
        return d_accel + v_peak * (t - t_accel)
    t_dec = t - t_accel - t_flat
    return d_accel + v_peak * t_flat + v_peak * t_dec - 0.5 * a_max * t_dec * t_dec


def _trapezoid_fraction(t: float, d: float, v_max: float, a_max: float) -> float:
    """将时刻 t 映射成 0..1 的插值进度。"""
    if d <= 0.0:
        return 1.0
    pos = _trapezoid_position(t, d, v_max, a_max)
    return float(np.clip(pos / d, 0.0, 1.0))


def plan_action_sequences(curr_obs: Dict[str, np.ndarray],
                          action: Dict[str, np.ndarray],
                          duration_per_step: float,
                          limits: Dict[str, Dict[str, Optional[float]]],
                          min_steps: int,
                          ) -> Dict[str, list[np.ndarray]]:
    """在 xyz/rpy 的速度、加速度约束下，为双臂规划逐步执行的末端目标序列。"""
    v_xyz = limits["xyz"]["v"]
    a_xyz = limits["xyz"]["a"]
    v_rpy = limits["rpy"]["v"]
    a_rpy = limits["rpy"]["a"]

    if v_xyz is None or a_xyz is None or v_rpy is None or a_rpy is None:
        raise ValueError("Missing xyz/rpy limits for constrained planning.")

    dt = float(duration_per_step) if duration_per_step > 0 else 0.02
    eps_xyz = 5e-4
    eps_rpy = 5e-4
    eps_grip = 5e-4
    grip_finish_ratio = 0.5
    min_steps_i = max(int(min_steps), 0)
    min_steps_grip = min_steps_i // 2
    if min_steps_i > 0 and min_steps_grip == 0:
        min_steps_grip = 1
    results: Dict[str, list[np.ndarray]] = {}

    for side in ("left", "right"):
        target = action.get(side)
        if target is None:
            continue
        if not isinstance(target, np.ndarray):
            target = np.array(target, dtype=np.float32)
        curr_end = curr_obs.get(f"{side}_end_pos")
        curr_joint = curr_obs.get(f"{side}_joint_pos")
        if curr_end is None or curr_joint is None:
            continue

        curr_gripper = float(curr_joint[6])
        start_pose = np.array(curr_end, dtype=np.float32)
        start_grip = curr_gripper
        delta_xyz = target[:3] - start_pose[:3]
        # 用笛卡尔位移中变化最大的那个轴，作为平移梯形曲线的标量距离。
        d_xyz = float(np.max(np.abs(delta_xyz)))

        q0 = _quat_from_rpy(start_pose[3:6])
        q1 = _quat_from_rpy(target[3:6])
        # 姿态距离在四元数空间中计算，插值完成后再转回 RPY。
        d_rpy = float(_quat_angle(q0, q1))

        delta_g = float(target[6]) - start_grip
        d_g = abs(delta_g)

        t_xyz = _trapezoid_params(d_xyz, v_xyz, a_xyz)[
            0] if d_xyz > eps_xyz else 0.0
        t_rpy = _trapezoid_params(d_rpy, v_rpy, a_rpy)[
            0] if d_rpy > eps_rpy else 0.0
        # 平移和旋转谁更慢，就由谁决定整段动作需要多少控制周期。
        steps_xyz = int(np.ceil(t_xyz / dt)) if t_xyz > 0 else 0
        steps_rpy = int(np.ceil(t_rpy / dt)) if t_rpy > 0 else 0
        pose_steps = max(steps_xyz, steps_rpy)
        if pose_steps > 0 and min_steps_i > 0:
            pose_steps = max(pose_steps, min_steps_i)

        if d_g > eps_grip:
            if pose_steps > 0:
                # 默认让夹爪比整段位姿动作更早完成。
                grip_steps = max(
                    1, int(np.ceil(pose_steps * grip_finish_ratio)))
            else:
                grip_steps = max(
                    1, min_steps_grip) if min_steps_grip > 0 else 1
        else:
            grip_steps = 0

        max_steps = max(pose_steps, grip_steps)
        if max_steps <= 0:
            continue

        seq: list[np.ndarray] = []
        for i in range(max_steps):
            t = (i + 1) * dt
            # 分别把当前时刻换算成平移和旋转的插值进度。
            if d_xyz > eps_xyz:
                s_xyz = _trapezoid_fraction(t, d_xyz, v_xyz, a_xyz)
            else:
                s_xyz = 1.0
            if d_rpy > eps_rpy:
                s_rpy = _trapezoid_fraction(t, d_rpy, v_rpy, a_rpy)
            else:
                s_rpy = 1.0

            pos_xyz = start_pose[:3] + delta_xyz * s_xyz
            if d_rpy > eps_rpy:
                q = _quat_slerp(q0, q1, s_rpy)
                pos_rpy = _rpy_from_quat(q)
            else:
                pos_rpy = start_pose[3:6]

            if grip_steps > 0:
                s_grip = min((i + 1) / grip_steps, 1.0)
            else:
                s_grip = 1.0

            grip_val = start_grip + delta_g * s_grip
            # 每一步的格式为 [x, y, z, roll, pitch, yaw, gripper]。
            seq.append(np.concatenate([pos_xyz, pos_rpy, [grip_val]]))
        results[side] = seq

    return results
