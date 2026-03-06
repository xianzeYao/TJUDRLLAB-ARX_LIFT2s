from typing import Any, Dict, Tuple, Iterable, Literal, Optional
import numpy as np
import subprocess
import time
from pathlib import Path
import shlex
import rclpy
from arx_ros2_env_utils import *


from arx5_arm_msg.msg._robot_cmd import RobotCmd
from arm_control.msg._pos_cmd import PosCmd


class ARXRobotEnv():
    """
    Concrete implementation of BasetEnv for ARX robot.
    """

    def __init__(self, duration_per_step: float = 0.02, min_steps: int = 10,
                 max_v_xyz: float = 0.25, max_v_rpy: float = 0.3,
                 max_a_xyz: float = 0.20, max_a_rpy: float = 1.00,
                 camera_type: Literal["color", "depth", "all"] = "all", camera_view: Iterable[str] = ("camera_l", "camera_h", "camera_r"),
                 dir: Optional[str] = None, video: bool = False, video_fps: float = 20.0,
                 video_name: Optional[str] = None,
                 img_size: Optional[Tuple[int, int]] = (224, 224)):
        super().__init__()
        self.camera_view = camera_view
        self.camera_type = camera_type
        self.dir = dir
        self.video = video
        self.video_fps = float(video_fps)
        self.video_name = video_name
        self.img_size = img_size
        self.duration_per_step = duration_per_step
        self.min_steps = min_steps

        self.max_v_xyz = max_v_xyz
        self.max_v_rpy = max_v_rpy
        self.max_a_xyz = max_a_xyz
        self.max_a_rpy = max_a_rpy

        # 1. Enable the robot
        success, error_message = self._enable_robot()
        if not success:
            raise RuntimeError(f"Failed to enable the robot: {error_message}")

    def reset(self) -> Dict[str, np.ndarray]:
        """
        Resets the robot to the initial pose.

        Returns:
            observation (Dict[str, np.ndarray]): The initial state of the robot.
        """
        time.sleep(2.5)
        # 1. Go to the initial pose
        success, error_message = self._go_to_initial_pose()
        if not success:
            raise RuntimeError(
                f"Failed to go to the initial pose: {error_message}")

        # 2. Stop the base and lift the base to a safe height
        self.step_lift(0.0)

        self.step_base(0.0, 0.0, 0.0, 0.5)

        # 3. Get the initial observation
        obs = self.get_observation()

        return obs

    def step_lift(self, height: float):
        """Adjust base hight"""
        success, error = self._apply_lift(height)
        if not success:
            raise RuntimeError(f"Failed to lift base: {error}")
        print(f"lift to height {height} done")

    def step_base(self, vx: float, vy: float, vz: float, duration: float):
        """Move base"""
        success, error = self._apply_base(vx, vy, vz, duration)
        if not success:
            raise RuntimeError(f"Failed to move base: {error}")

    def step(self, action: np.ndarray):
        """
        Execute one time step in the environment.

        Args:
            action (np.ndarray): Action provided by the agent. Shape (action_dim,).

        Returns:
            observation (np.ndarray): New state of the robot.
            reward (float): Scalar reward value.
            done (bool): Whether the episode has ended.
            info (dict): Diagnostic information.
        """

        # 1. Apply action
        success, error_message = self._apply_action(action)
        if not success:
            self.close()
            raise RuntimeError(f"Failed to apply action: {error_message}")

        # 2. State Observation
        # Retrieve latest data from ROS topics
        obs = self.get_observation()

        reward = 0.0
        is_done = False
        info = dict()
        # 3. Reward Calculation
        # reward = self._get_reward(obs, action)

        # 4. Termination Logic
        # is_done = self._get_termination(obs, action)

        # info = self._get_info()

        return obs, reward, is_done, info

    def get_observation(self, save_dir: Optional[str] = None,
                        video: Optional[bool] = None,
                        include_arm: bool = True,
                        include_camera: bool = True,
                        include_base: bool = True) -> Dict[str, np.ndarray]:
        """
        Fetch latest status/camera and pack into observation.

        save_dir: 默认 None 时沿用实例化时的 self.dir；显式传值覆盖；传入空串/False 可关闭保存。
        video: 默认 None 时沿用实例化时的 self.video；True 保存视频，False 保存图片。
        """
        camera_all, status_all = self.get_camera(
            save_dir=save_dir,
            video=video,
            target_size=self.img_size,
            return_status=True,
        )
        obs = build_observation(
            camera_all, status_all,
            include_arm=include_arm,
            include_camera=include_camera,
            include_base=include_base,
        )
        if not obs:
            try:
                self.close()
            except Exception:
                pass
            raise RuntimeError("Empty observation, node shutdown.")
        return obs

    def get_camera(self,
                   save_dir: Optional[str] = None,
                   video: Optional[bool] = None,
                   target_size: Optional[Tuple[int, int]] = None,
                   return_status: bool = False):
        """
        环境层统一相机入口。

        业务代码优先通过 env.get_camera()/env.get_observation() 取数据，
        不要直接下钻到 env.node.get_camera()。
        """
        real_save_dir = self.dir if save_dir is None else save_dir
        real_video = self.video if video is None else video
        real_target_size = self.img_size if target_size is None else target_size
        return self.node.get_camera(
            save_dir=real_save_dir,
            target_size=real_target_size,
            save_video=real_video,
            return_status=return_status,
        )

    def get_robot_status(self):
        """环境层统一状态入口，屏蔽底层 node 接口。"""
        return self.node.get_robot_status()

    def close(self):
        """
        Clean up resources and shut down ROS nodes.
        """

        # 1. Stop the base and make both arms go to initial pose and set height to 0
        self.step_base(0.0, 0.0, 0.0, 1.0)
        success, error_message = self._go_to_initial_pose()
        self.step_lift(0.0)
        if not success:
            raise RuntimeError(
                f"Failed to go to the initial pose: {error_message}")

        # 2. Disable the robot
        success, error_message = self._disable_robot()
        if not success:
            raise RuntimeError(f"Failed to disable the robot: {error_message}")

    # ----------------- Private helpers -----------------

    def _apply_lift(self, height: float) -> Tuple[bool, str | None]:
        """Internal: move base height with incremental steps."""
        msg = PosCmd()
        base_status = self.get_robot_status().get("base")
        if base_status is None:
            return (False, "base status unavailable")
        curr = float(base_status.height)
        target = float(height)
        step = 0.1 if target >= curr else -0.1
        while abs(curr - target) > 0.01:
            curr += step
            # 防止越界
            if (step > 0 and curr > target) or (step < 0 and curr < target):
                curr = target
            msg.height = curr
            if not self.node.send_base_msg(msg):
                return (False, "base command not sent")
            time.sleep(0.03)
        return (True, None)

    def _apply_base(self, vx: float, vy: float, vz: float, duration: float) -> Tuple[bool, str | None]:
        """Internal: move base chassis with timeout."""
        start = time.time()
        msg = PosCmd()
        while time.time() - start < duration:
            msg.chx = vx
            msg.chy = vy
            msg.chz = vz
            base_status = self.get_robot_status().get("base")
            msg.height = float(
                base_status.height) if base_status is not None else 0.0
            msg.mode1 = 1
            if not self.node.send_base_msg(msg):
                return (False, "base command not sent")
        # stop
        msg.chx = msg.chy = msg.chz = 0.0
        msg.mode1 = 2
        if not self.node.send_base_msg(msg):
            return (False, "base stop command not sent")
        return (True, None)

    def _apply_action(self, action: Dict[str, np.ndarray]) -> Tuple[bool, str | None]:
        """
        apply to the robot single step control command.
            action: the action provieded by the agent（step level）.
            end: x,y,z,roll,pitch,yaw,gripper
            joint: joint0,joint1,...,joint5,gripper

        """
        curr_obs = self.get_observation()
        limits = {
            "xyz": {"v": self.max_v_xyz, "a": self.max_a_xyz},
            "rpy": {"v": self.max_v_rpy, "a": self.max_a_rpy},
        }
        sequences = plan_action_sequences(
            curr_obs,
            action,
            self.duration_per_step,
            limits,
            self.min_steps,
        )
        lsequence = sequences.get("left") or []
        rsequence = sequences.get("right") or []
        max_len = max(len(lsequence), len(rsequence))
        for i in range(max_len):
            has_left = i < len(lsequence)
            has_right = i < len(rsequence)
            if not has_left and not has_right:
                continue
            t0 = time.time()
            if has_left:
                lmsg = RobotCmd()
                lmsg.mode = 4
                lmsg.end_pos = [float(x) for x in lsequence[i][:6]]
                lmsg.gripper = float(lsequence[i][6])
                lok = self.node.send_control_msg("left", lmsg)
                if not lok:
                    return (False, f"left: command not sent")
            if has_right:
                rmsg = RobotCmd()
                rmsg.mode = 4
                rmsg.end_pos = [float(x) for x in rsequence[i][:6]]
                rmsg.gripper = float(rsequence[i][6])
                rok = self.node.send_control_msg("right", rmsg)
                if not rok:
                    return (False, f"right: command not sent")
            dt = time.time() - t0
            sleep_need = self.duration_per_step - dt
            if sleep_need > 0:
                time.sleep(sleep_need)
        return (True, None)

    def _go_to_initial_pose(self) -> Tuple[bool, str | None]:
        """Move arms to initial pose."""
        home_action = {
            "left": np.array([0, 0, 0, 0, 0, 0, 0], dtype=np.float32),
            "right": np.array([0, 0, 0, 0, 0, 0, 0], dtype=np.float32),

        }
        success, error_message = self._apply_action(home_action)
        if not success:
            print(
                f"failed to go home: {error_message}, force switch to home mode")
            self.set_special_mode(1)
            return (False, f"failed to go home: {error_message}")
        else:
            print(f"both arms homed")
        return (True, None)

    def set_special_mode(self, mode: int) -> Tuple[bool, str | None]:
        """Set special mode, e.g. gravity."""
        mode_type = {0: "soft", 1: "home", 2: "protect", 3: "gravity"}
        cmd = RobotCmd()
        cmd.mode = mode
        # 0-soft,1-home,2-protect,3-gravity
        self.node.send_control_msg("left", cmd)
        self.node.send_control_msg("right", cmd)
        print(f"set mode for both arms {mode_type.get(mode, 'unknown')} done")
        return (True, None)

    def _setup_space(self):
        """Configure action/observation space."""
        # TODO 定义一下动作空间，可以快速check
        pass

    def _enable_robot(self) -> Tuple[bool, str | None]:
        rclpy.init()

        self.node, self.executor, self.executor_thread = start_robot_io(
            self.camera_type, self.camera_view, self.video, self.video_fps, self.dir, self.img_size, self.video_name)
        if self.node and self.executor:
            return (True, None)
        erorr = []
        if not self.node:
            erorr.append("IO node failed to start")
        if not self.executor:
            erorr.append("Executor init failed")
        if not rclpy.ok():
            erorr.append("ROS2 init failed")
        return (False, "reason: ".join(erorr) if erorr else "other error")

    def _disable_robot(self) -> Tuple[bool, str | None]:
        """Destroy comms node and executor."""

        if getattr(self, "node", None) is not None:
            try:
                self.node.stop_saver()
            except Exception:
                pass
            self.node.destroy_node()
            self.node = None
        if getattr(self, "executor", None) is not None:
            self.executor.shutdown()
            self.executor = None
        if getattr(self, "executor_thread", None) is not None:
            self.executor_thread.join(timeout=2.0)
            self.executor_thread = None
        if rclpy.ok():
            rclpy.shutdown()

        if self.node is None and self.executor is None and not rclpy.ok():
            return (True, None)
        erorr = []
        if self.node is not None:
            erorr.append("IO node destroy failed")
        if self.executor is not None:
            erorr.append("Executor destroy failed")
        if rclpy.ok():
            erorr.append("ROS2 shutdown failed")
        return (False, "reason: ".join(erorr) if erorr else "other error")


def main():
    arx = ARXRobotEnv(
        duration_per_step=1.0/20.0,
        min_steps=20,
        max_v_xyz=0.25, max_a_xyz=0.20,
        max_v_rpy=0.3, max_a_rpy=1.00,
        camera_type="all",
        camera_view=("camera_h",),
        dir="testdata",
        video=True,
        video_fps=30.0,
        video_name="test",
        img_size=(640, 480)
    )

    # arx = ARXRobotEnv(duration_per_step=1.0/20.0,
    #                   min_steps=20,
    #                   max_v_xyz=0.25, max_a_xyz=0.20,
    #                   max_v_rpy=0.3, max_a_rpy=1.00,
    #                   camera_type="all",
    #                   camera_view=("camera_h",),
    #                   dir="testdata",
    #                   img_size=(640, 480))

    obs = arx.reset()
    arx.step_lift(18.0)
    action = {
        "left": np.array([0.1, 0, 0.15, 0, 0, 0, -2.0], dtype=np.float32),
        "right": np.array([0.1, 0, 0.15, 0, 0, 0, -2.0], dtype=np.float32),
    }
    arx.step(action)
    action = {
        "left": np.array([0.2, 0, 0.17, 0, 0, 0, -3.4], dtype=np.float32),
        "right": np.array([0.2, 0, 0.17, 0, 0, 0, -3.4], dtype=np.float32),
    }
    arx.step(action)
    arx.close()


if __name__ == "__main__":
    main()
