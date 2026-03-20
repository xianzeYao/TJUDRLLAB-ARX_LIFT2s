
import time
import sys
sys.path.append("../ARX_Realenv/ROS2")  # noqa
from arx_ros2_env import ARXRobotEnv  # noqa
import numpy as np
from arx5_arm_msg.msg._robot_cmd import RobotCmd


def main():
    try:
        arx = ARXRobotEnv(duration_per_step=1.0/20.0,  # 就是插值里一步的时间，20Hz也就是0.05s
                          min_steps=20,
                          max_v_xyz=0.1, max_a_xyz=0.15,
                          max_v_rpy=0.3, max_a_rpy=1.00,
                          camera_type="all",
                          camera_view=("camera_h",),
                          img_size=(640, 480))
        arx.reset()
        arx.step_lift(18.0)
        open_action = {
            "left":  np.array([0, 0, 0, -1.571, 0, 0, -3.4], dtype=np.float32,),
            "right": np.array([0, 0, 0, 1.571, 0, 0, -3.4], dtype=np.float32),
        }
        arx.step_smooth_eef(open_action)
        print("请放取春联，10秒后开始夹取...")
        time.sleep(5.0)
        close_action = {
            "left":  np.array([0, 0, 0, -1.571, 0, 0, 0.0], dtype=np.float32,),
            "right": np.array([0, 0, 0, 1.571, 0, 0, 0.0], dtype=np.float32),
        }
        arx.step_smooth_eef(close_action)

        time.sleep(1.0)
        lift_action = {
            "left": np.array(
                [0.1, 0.05, 0.45, -1.571, 0, 0, 0.0], dtype=np.float32,),
            "right": np.array([0.1, -0.05, 0.45, 1.571, 0, 0, 0.0], dtype=np.float32),
        }
        arx.step_smooth_eef(lift_action)
        time.sleep(5.0)
        # 使用原厂IK抖动
        shake_action = {
            "left": np.array(
                [0.1, 0.05, 0.35, -1.571, 0.5, 0, 0], dtype=np.float32,),
            "right": np.array([0.1, -0.05, 0.35, 1.571, 0.5, 0, 0], dtype=np.float32),
        }
        lmsg = RobotCmd()
        lmsg.mode = 4
        lmsg.end_pos = list(shake_action["left"][:6])
        lmsg.gripper = float(shake_action["left"][6])
        arx.node.send_control_msg("left", lmsg)
        rmsg = RobotCmd()
        rmsg.mode = 4
        rmsg.end_pos = list(shake_action["right"][:6])
        rmsg.gripper = float(shake_action["right"][6])
        arx.node.send_control_msg("right", rmsg)
        time.sleep(5.0)
        arx.step_smooth_eef(open_action)
    except Exception as e:
        print(f"An error occurred: {e}")
    finally:
        arx.close()


if __name__ == "__main__":
    main()
