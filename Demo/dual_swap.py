
import time
import sys
sys.path.append("../ARX_Realenv/ROS2")  # noqa
from arx_ros2_env import ARXRobotEnv  # noqa
import numpy as np
import cv2
from arx5_arm_msg.msg._robot_cmd import RobotCmd
from motion_swap import build_swap_sequence  # 控制命令
from point2pos_utils import load_cam2ref, load_intrinsics, pixel_to_base_point, pixel_to_ref_point_safe
from arx_pointing import predict_point_from_rgb
from arm_control.msg._pos_cmd import PosCmd


def _get_frame(arx: ARXRobotEnv):
    while True:
        frames = arx.node.get_camera(
            target_size=(640, 480), return_status=False)
        color = frames.get("camera_h_color")
        depth = frames.get("camera_h_aligned_depth_to_color")
        if color is None or depth is None:
            cv2.waitKey(1)
            continue
        return color, depth


def main():
    try:
        arx = ARXRobotEnv(duration_per_step=1.0/20.0,  # 就是插值里一步的时间，20Hz也就是0.05s
                          min_steps=20,
                          max_v_xyz=0.15, max_a_xyz=0.20,
                          max_v_rpy=0.45, max_a_rpy=1.00,
                          camera_type="all",
                          camera_view=("camera_h",),
                          img_size=(640, 480))
        arx.reset()
        # arx.step_lift(5.0)
        open_action = {
            "left":  np.array([0, 0, 0, 0, 0, 0, -3.4], dtype=np.float32,),
            "right": np.array([0, 0, 0, 0, 0, 0, -3.4], dtype=np.float32),
        }
        arx.step(open_action)
        print("请放取扫把簸箕，5秒后开始夹取...")
        time.sleep(5.0)
        close_action = {
            "left":  np.array([0, 0, 0, 0, 0, 0, 0.0], dtype=np.float32,),
            "right": np.array([0, 0, 0, 0, 0, 0, 0.0], dtype=np.float32),
        }
        arx.step(close_action)
        time.sleep(5.0)
        lift_action = {"left": np.array([0, 0, 0.1, 0, 0, 0, 0.0], dtype=np.float32,),
                       "right": np.array([0, 0, 0.1, 0, 0, 0, 0.0], dtype=np.float32),
                       }
        arx.step(lift_action)
        time.sleep(1.0)
        # arx.step_base(vx=-0.5, vy=0.0, vz=0.5, duration=9.0)
        # arx.step_base(vx=0.75, vy=0.0, vz=0.0, duration=2.0)
        # arx.step_base(vx=-0.5, vy=0.0, vz=0.0, duration=5.0)
        # 简单detect白色纸团
        K = load_intrinsics()
        T_left = load_cam2ref(side="left")
        trash_prompt = "a white crumpled paper on the floor"
        while True:
            color, depth = _get_frame(arx)
            trash_point = predict_point_from_rgb(color, trash_prompt)
            u, v = int(round(trash_point[0])), int(round(trash_point[1]))
            raw_depth = float(depth[v, u])
            if not np.isfinite(raw_depth) or raw_depth <= 0:
                print(f"预测像素 {(u, v)} 深度无效({raw_depth})，按 r 刷新")
                continue
            trash_base_point = pixel_to_base_point((u, v), depth, K, T_left)
            print(f"trash_base_point: {trash_base_point}")
            vis = color.copy()
            cv2.circle(vis, (u, v), 3, (0, 0, 255), -1)
            win = "dual_swap_detect"
            cv2.namedWindow(win, cv2.WINDOW_NORMAL)
            cv2.imshow(win, vis)
            key = cv2.waitKey(0)
            if key == ord("r"):
                continue
            if key == ord("q"):
                return
            if key == ord("w"):
                arx.step_base(vx=0.5, vy=0.0, vz=0.0, duration=1.0)
                continue
            if key == ord("e"):
                break

        swap_seq = build_swap_sequence(trash_base_point)
        for action in swap_seq:
            arx.step(action)

    except Exception as e:
        print(f"An error occurred: {e}")
    finally:
        arx.step(open_action)
        cv2.destroyAllWindows()
        time.sleep(5.0)
        arx.close()


if __name__ == "__main__":
    main()
