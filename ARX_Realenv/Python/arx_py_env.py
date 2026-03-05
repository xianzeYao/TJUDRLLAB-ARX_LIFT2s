"""
Python-only env using ARX_X5 Python SDK (arms), ARX_LIFT Python SDK (base/head),
and pyrealsense2 for camera. No ROS2 dependency.
"""
from __future__ import annotations
import time
from typing import Dict, Optional, Tuple

try:
    import pyrealsense2 as rs
except ImportError:  # pragma: no cover
    rs = None

try:
    # SingleArm wraps arx_x5_python bindings
    from ARX_X5.py.arx_x5_python.bimanual.script.single_arm import SingleArm
except ImportError:  # pragma: no cover
    SingleArm = None

try:
    import arx_lift_python as arx_lift
except ImportError:  # pragma: no cover
    arx_lift = None


class ARXPythonEnv:
    """
    Python SDK env:
    - left/right SingleArm (optional)
    - base/head via arx_lift_python (optional)
    - RealSense color+depth via pyrealsense2
    """

    def __init__(
        self,
        left_config: Optional[dict] = None,
        right_config: Optional[dict] = None,
        base_bus: str = "can5",
        base_robot_type: str = "X7S",
        realsense_serial: Optional[str] = None,
        depth_align_to_color: bool = True,
        frame_size: Tuple[int, int] = (640, 480),
        fps: int = 30,
    ):
        if SingleArm is None:
            raise ImportError("SingleArm import failed; ensure arx_x5_python is built and on PYTHONPATH.")
        if rs is None:
            raise ImportError("pyrealsense2 not available; install librealsense-python.")
        self.left = SingleArm(left_config) if left_config else None
        self.right = SingleArm(right_config) if right_config else None

        if arx_lift is None:
            self.base = None
        else:
            rt = arx_lift.LiftHeadControlLoop.RobotType[base_robot_type]
            self.base = arx_lift.LiftHeadControlLoop(base_bus, rt)

        self.frame_size = frame_size
        self.fps = fps
        self.depth_align_to_color = depth_align_to_color

        self._pipeline = rs.pipeline()
        self._align = rs.align(rs.stream.color) if depth_align_to_color else None
        cfg = rs.config()
        if realsense_serial:
            cfg.enable_device(realsense_serial)
        cfg.enable_stream(rs.stream.color, frame_size[0], frame_size[1], rs.format.bgr8, fps)
        cfg.enable_stream(rs.stream.depth, frame_size[0], frame_size[1], rs.format.z16, fps)
        self._pipeline.start(cfg)

    def reset(self):
        if self.left:
            self.left.go_home()
        if self.right:
            self.right.go_home()
        return self.get_observation()

    def _wait_frames(self):
        frames = self._pipeline.wait_for_frames()
        if self.depth_align_to_color and self._align:
            frames = self._align.process(frames)
        color = frames.get_color_frame()
        depth = frames.get_depth_frame()
        color_np = None
        depth_np = None
        if color:
            import numpy as np
            color_np = np.asanyarray(color.get_data())
        if depth:
            import numpy as np
            depth_np = np.asanyarray(depth.get_data())
        return color_np, depth_np

    def get_observation(self) -> Dict[str, object]:
        obs: Dict[str, object] = {}
        color_np, depth_np = self._wait_frames()
        if color_np is not None:
            obs["color"] = color_np
        if depth_np is not None:
            obs["depth"] = depth_np
        if self.left:
            obs["left_joint_pos"] = self.left.get_joint_positions()
            obs["left_joint_vel"] = self.left.get_joint_velocities()
            obs["left_joint_cur"] = self.left.get_joint_currents()
            obs["left_end_pos"] = self.left.get_ee_pose()
        if self.right:
            obs["right_joint_pos"] = self.right.get_joint_positions()
            obs["right_joint_vel"] = self.right.get_joint_velocities()
            obs["right_joint_cur"] = self.right.get_joint_currents()
            obs["right_end_pos"] = self.right.get_ee_pose()
        if self.base:
            obs["base_height"] = self.base.get_height()
            obs["base_waist"] = self.base.get_waist_pos()
            obs["base_head_yaw"] = self.base.get_head_yaw()
            obs["base_head_pitch"] = self.base.get_head_pitch()
        return obs

    def step(self, action: Dict[str, object]):
        """
        action:
          - left/right: np.array([x,y,z,roll,pitch,yaw,grip])
          - base: dict with optional keys height, waist, head_yaw, head_pitch, chassis (vx,vy,wz,mode)
        """
        if self.left and "left" in action:
            target = action["left"]
            self.left.set_ee_pose_xyzrpy(target[:6])
            self.left.set_catch_pos(float(target[6]))
        if self.right and "right" in action:
            target = action["right"]
            self.right.set_ee_pose_xyzrpy(target[:6])
            self.right.set_catch_pos(float(target[6]))
        if self.base and "base" in action:
            b = action["base"] or {}
            if "height" in b:
                self.base.set_height(float(b["height"]))
            if "waist" in b:
                self.base.set_waist_pos(float(b["waist"]))
            if "head_yaw" in b:
                self.base.set_head_yaw(float(b["head_yaw"]))
            if "head_pitch" in b:
                self.base.set_head_pitch(float(b["head_pitch"]))
            if "chassis" in b:
                vx, vy, wz, mode = b["chassis"]
                self.base.set_chassis_cmd(float(vx), float(vy), float(wz), int(mode))
            try:
                self.base.loop()
            except Exception:
                pass
        dt = None
        if self.left:
            dt = getattr(self.left, "dt", None)
        if dt:
            time.sleep(dt)
        return self.get_observation()

    def close(self):
        if self.left:
            try:
                self.left.protect_mode()
            except Exception:
                pass
        if self.right:
            try:
                self.right.protect_mode()
            except Exception:
                pass
        if self.base:
            try:
                self.base.set_chassis_cmd(0, 0, 0, 2)
            except Exception:
                pass
        try:
            self._pipeline.stop()
        except Exception:
            pass


__all__ = ["ARXPythonEnv"]
