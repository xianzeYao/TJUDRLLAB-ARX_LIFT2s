#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState
from moveit_msgs.srv import GetPositionIK
from moveit_msgs.msg import PositionIKRequest, RobotState
from geometry_msgs.msg import PoseStamped
from builtin_interfaces.msg import Duration
from urdf_parser_py.urdf import URDF
from moveit_msgs.action import MoveGroup
from moveit_msgs.msg import Constraints, JointConstraint
from rclpy.action import ActionClient
from Robotic_Arm.rm_robot_interface import RoboticArm, rm_thread_mode_e
import time
import numpy as np
from scipy.spatial.transform import Rotation as R

# ==========================================================
# 机器人关节名（和 URDF 保持一致）
# ==========================================================
JOINT_NAMES = [
    'joint1', 'joint2', 'joint3',
    'joint4', 'joint5', 'joint6', 'joint7'
]

# ==========================================================
# 将 xyz + rpy 转换为 PoseStamped
# ==========================================================
def xyzrpy_to_pose_stamped(xyzrpy, frame_id="base_link"):
    x, y, z, rx, ry, rz = xyzrpy
    pose = PoseStamped()
    pose.header.frame_id = frame_id
    pose.pose.position.x = x
    pose.pose.position.y = y
    pose.pose.position.z = z
    quat = R.from_euler("xyz", [rx, ry, rz]).as_quat()  # xyzw
    pose.pose.orientation.x = quat[0]
    pose.pose.orientation.y = quat[1]
    pose.pose.orientation.z = quat[2]
    pose.pose.orientation.w = quat[3]
    return pose

# ==========================================================
# 解析 URDF 获取 joint limits
# ==========================================================
def parse_urdf_limits(urdf_path, joint_names):
    with open(urdf_path, "rb") as f:
        xml_bytes = f.read()
    robot = URDF.from_xml_string(xml_bytes)
    lower_limits = []
    upper_limits = []
    for name in joint_names:
        joint = robot.joint_map.get(name, None)
        if joint is None:
            raise ValueError(f"Joint {name} not found in URDF")
        lower_limits.append(joint.limit.lower)
        upper_limits.append(joint.limit.upper)
    return lower_limits, upper_limits

# ==========================================================
# MoveIt2 + Realman 封装类
# ==========================================================
class MoveItToRealman(Node):
    def __init__(self):
        super().__init__('moveit_to_realman')

        # ----------------------------
        # 初始化 MoveGroup Action Client
        # ----------------------------
        self.client = ActionClient(self, MoveGroup, '/move_action')
        self.get_logger().info("MoveGroup client created, waiting for server...")
        self.client.wait_for_server()
        self.get_logger().info("MoveGroup server ready.")

        # ----------------------------
        # 初始化 Realman 机械臂
        # ----------------------------
        self.arm = RoboticArm(rm_thread_mode_e.RM_TRIPLE_MODE_E)
        self.arm_handle = self.arm.rm_create_robot_arm("192.168.101.19", 8080)
        if self.arm_handle.id < 0:
            raise RuntimeError("机械臂连接失败")

    def send_joint_goal(self, joint_names, joint_positions, velocity_scaling=0.2, acceleration_scaling=0.2):
        # 构造 MoveIt2 Goal
        goal = MoveGroup.Goal()
        goal.request.group_name = "rm_group"
        goal.request.num_planning_attempts = 5
        goal.request.allowed_planning_time = 5.0
        goal.request.start_state.joint_state.name = joint_names
        goal.request.start_state.joint_state.position = joint_positions

        # 设置 JointConstraints
        joint_constraints = []
        for name, pos in zip(joint_names, joint_positions):
            jc = JointConstraint()
            jc.joint_name = name
            jc.position = pos
            jc.tolerance_above = 0.001
            jc.tolerance_below = 0.001
            jc.weight = 1.0
            joint_constraints.append(jc)
        goal.request.goal_constraints = [Constraints(joint_constraints=joint_constraints)]

        goal.request.max_velocity_scaling_factor = velocity_scaling
        goal.request.max_acceleration_scaling_factor = acceleration_scaling

        # 发送 MoveIt2 Goal
        self.get_logger().info("Sending goal to MoveGroup...")
        future = self.client.send_goal_async(goal)
        rclpy.spin_until_future_complete(self, future)
        goal_handle = future.result()
        if not goal_handle.accepted:
            self.get_logger().error("Goal rejected by MoveGroup!")
            return

        self.get_logger().info("Goal accepted, waiting for result...")
        result_future = goal_handle.get_result_async()
        rclpy.spin_until_future_complete(self, result_future)
        result = result_future.result().result

        # 解析 MoveIt2 轨迹并发送到 Realman
        traj = result.planned_trajectory.joint_trajectory
        self.get_logger().info(f"Received planned trajectory with {len(traj.points)} points")

        for i, pt in enumerate(traj.points):
            positions = list(pt.positions)
            if i < len(traj.points) - 1:
                dt = (traj.points[i+1].time_from_start.sec + traj.points[i+1].time_from_start.nanosec*1e-9) - \
                     (pt.time_from_start.sec + pt.time_from_start.nanosec*1e-9)
            else:
                dt = 0.1

            ret = self.arm.rm_movej(positions, v=30, r=5, connect=1, block=0)
            if ret != 0:
                self.get_logger().warn(f"Point {i} send failed, ret={ret}")
            else:
                self.get_logger().info(f"Point {i} sent: {positions}")

            if dt > 0 and i != len(traj.points) - 1:
                time.sleep(dt)

    def shutdown(self):
        self.arm.rm_delete_robot_arm()
        self.get_logger().info("Disconnected Realman arm.")

# ==========================================================
# 主函数：先求 IK，再发 MoveIt+Realman
# ==========================================================
def main():
    rclpy.init()
    node = Node("moveit2_ik_demo")

    # -----------------------------
    # 1. 读取 URDF joint limits
    # -----------------------------
    URDF_PATH = "/home/lizy/ros2_ws/src/ros2_rm_robot/rm_description/urdf/rm_75.urdf"
    lower_limits, upper_limits = parse_urdf_limits(URDF_PATH, JOINT_NAMES)

    # -----------------------------
    # 2. 获取当前关节状态
    # -----------------------------
    current_joint_state = None
    def joint_state_cb(msg):
        nonlocal current_joint_state
        current_joint_state = msg
    sub = node.create_subscription(JointState, "/joint_states", joint_state_cb, 10)

    timeout = time.time() + 5.0
    while current_joint_state is None and time.time() < timeout:
        rclpy.spin_once(node, timeout_sec=0.1)
    if current_joint_state is None:
        node.get_logger().warn("Failed to get current joint_states, using zeros")
        current_positions = [0.0]*7
    else:
        current_positions = []
        for name in JOINT_NAMES:
            if name in current_joint_state.name:
                idx = current_joint_state.name.index(name)
                current_positions.append(current_joint_state.position[idx])
            else:
                current_positions.append(0.0)

    # -----------------------------
    # 3. 调用 IK 服务
    # -----------------------------
    cli = node.create_client(GetPositionIK, "/compute_ik")
    if not cli.wait_for_service(timeout_sec=5.0):
        node.get_logger().error("IK service not available!")
        return

    xyzrpy = [-0.240035, -0.256265, 0.364198, -2.851, -0.537, 3.107]  # 示例
    pose_stamped = xyzrpy_to_pose_stamped(xyzrpy)

    req = GetPositionIK.Request()
    ik_req = PositionIKRequest()
    ik_req.group_name = "rm_group"
    ik_req.pose_stamped = pose_stamped
    ik_req.timeout = Duration(sec=2, nanosec=0)

    robot_state = RobotState()
    robot_state.joint_state.name = JOINT_NAMES
    robot_state.joint_state.position = current_positions
    ik_req.robot_state = robot_state
    req.ik_request = ik_req

    future = cli.call_async(req)
    rclpy.spin_until_future_complete(node, future)
    result = future.result()

    if result.error_code.val != 1:
        node.get_logger().error(f"IK failed! error code: {result.error_code.val}")
        return

    # -----------------------------
    # 4. 裁剪 joint limits
    # -----------------------------
    joint_positions = []
    for i, pos in enumerate(result.solution.joint_state.position):
        clipped = min(max(pos, lower_limits[i]), upper_limits[i])
        joint_positions.append(clipped)

    # 打印 IK 输出
    print("IK joint_names = [")
    for name in JOINT_NAMES:
        print(f"    '{name}',")
    print("]")

    print("\nIK joint_positions = [")
    for pos in joint_positions:
        print(f"    {pos},")
    print("]")

    # -----------------------------
    # 5. 发 MoveIt2 + Realman
    # -----------------------------
    moveit_node = MoveItToRealman()
    velocity_scaling = 0.3
    acceleration_scaling = 0.05
    moveit_node.send_joint_goal(JOINT_NAMES, joint_positions, velocity_scaling, acceleration_scaling)
    moveit_node.shutdown()

    node.destroy_node()
    moveit_node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()

