
import time
import sys
sys.path.append("../ARX_Realenv/ROS2")  # noqa
from arx_ros2_env import ARXRobotEnv  # noqa
import numpy as np
from arx5_arm_msg.msg._robot_cmd import RobotCmd  # 控制命令

# joint_key: 0-6分别对应7个关节


def _normalize_joint_targets(joint_key, target_pos):
    if isinstance(target_pos, dict):
        keys = [int(k) for k in target_pos.keys()]
        targets = [float(target_pos[k]) for k in target_pos.keys()]
    else:
        if isinstance(joint_key, (list, tuple, np.ndarray)):
            keys = [int(k) for k in joint_key]
        else:
            keys = [int(joint_key)]
        if isinstance(target_pos, (list, tuple, np.ndarray)):
            if len(target_pos) != len(keys):
                raise ValueError("target_pos 长度与 joint_key 不一致")
            targets = [float(v) for v in target_pos]
        else:
            if len(keys) != 1:
                raise ValueError("多个 joint_key 时必须提供多个 target_pos")
            targets = [float(target_pos)]
    return keys, targets


def give_joint_action(arx: ARXRobotEnv, side="left", joint_key: int = 3,
                      num: int = 50, target_pos: float = -3.4, sleep_s: float = 0.02):
    key = f"{side}_joint_pos"
    curr_joint = arx.get_observation(
        include_base=False, include_camera=False).get(key)
    if curr_joint is None:
        raise RuntimeError(f"未获取到关节信息: {key}")
    keys, targets = _normalize_joint_targets(joint_key, target_pos)
    for k in keys:
        if k < 0 or k >= len(curr_joint):
            raise IndexError(f"joint_key 超出范围: {k}")
    start_vals = [float(curr_joint[k]) for k in keys]
    seqs = [np.linspace(s, t, num=num) for s, t in zip(start_vals, targets)]
    for step in range(num):
        cmd = RobotCmd()
        cmd.mode = 5  # END_CONTROL（枚举：0 SOFT,1 GO_HOME,2 PROTECT,3 G_COMP,4 END_CONTROL,5 POSITION_CONTROL）
        joint = curr_joint.copy()
        for idx, k in enumerate(keys):
            joint[k] = float(seqs[idx][step])
        cmd.joint_pos = joint
        cmd.gripper = 0.0
        arx.node.send_control_msg(side, cmd)
        time.sleep(sleep_s)


def hello(arx: ARXRobotEnv, side="left", close_robot=True, wave_cycles: int = 1):
    try:
        arx.step_lift(16.0)
        # JOINT控制抬升
        give_joint_action(arx=arx, side=side, joint_key=[1, 2, 3],
                          target_pos=[0.7, 0.65, 1.3], num=60, sleep_s=arx.duration_per_step)
        time.sleep(0.5)
        # 单纯手腕控制用joint进行控制
        give_joint_action(
            arx, side, joint_key=4, target_pos=-0.7, num=30, sleep_s=arx.duration_per_step)
        cycles = max(0, int(wave_cycles))
        for _ in range(cycles):
            give_joint_action(
                arx, side, joint_key=4, target_pos=0.7, num=60, sleep_s=arx.duration_per_step
            )
            give_joint_action(
                arx, side, joint_key=4, target_pos=-0.7, num=60, sleep_s=arx.duration_per_step
            )
        give_joint_action(
            arx, side, joint_key=4, target_pos=0, num=30, sleep_s=arx.duration_per_step
        )
        home = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        give_joint_action(arx=arx, side=side, joint_key=[
                          0, 1, 2, 3, 4, 5, 6], target_pos=home, num=60, sleep_s=arx.duration_per_step)
        time.sleep(1.0)
    except Exception as e:
        print(f"An error occurred: {e}")
    finally:
        if close_robot:
            arx.close()


def main():
    arx = ARXRobotEnv(duration_per_step=1.0/40.0,  # 就是插值里一步的时间，20Hz也就是0.05s
                      min_steps=20,
                      max_v_xyz=0.25, max_a_xyz=0.20,
                      max_v_rpy=0.3, max_a_rpy=1.00,
                      camera_type="all",
                      camera_view=("camera_h",),
                      img_size=(640, 480))
    arx.reset()
    hello(arx, side="left", close_robot=True, wave_cycles=2)


if __name__ == "__main__":
    main()
