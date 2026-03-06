from nav_utils import depth_to_meters, get_key, extract_actions, merge_forward_actions, path_to_actions, refine_trajectory_strict
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

# ===============================
# 相机内参
# ===============================
K = np.array([
    [391.9335632324219, 0.0, 320.5389099121094],
    [0.0, 391.6839294433594, 239.18849182128906],
    [0.0, 0.0, 1.0]
])

FX, FY = K[0, 0], K[1, 1]
CX, CY = K[0, 2], K[1, 2]

# ===============================
# 相机 → base_link 外参
# ===============================

T_CAM2REF_R = np.array([
    [-0.019024340515354288, -0.5083879840468148,
        0.860917959009319, 0.025652315727880005],
    [-0.9992674280090241, 0.038266714301299354,
        0.0005156518327227855, 0.2573767428832648],
    [-0.033206652769975364, -0.8602774646899676, -
        0.5087435522524251, 0.1459514185409872],
    [0.0, 0.0, 0.0, 1.0]
])

T_CAM2REF_L = np.array([
    [-0.01022451527760726, -0.5071681372702741,
        0.861786481574838, 0.019333535519116728],
    [-0.9997376669412061, -0.012479673613300601, -
        0.019205599325708644, -0.23751223916353767],
    [0.020495282049587504, -0.8617567744348205, -
        0.5069074916879823, 0.13595597780350663],
    [0.0, 0.0, 0.0, 1.0]
])

BIAS_REF2CAM_L = np.array([0.0, 0.48, 0.0, 0.0])

# BIAS_REF2CAM = np.array([0.0, 0.48, 0.0, 0.0])


class AutoNav_Robot():
    def __init__(self, camera_type="all", camera_view=("camera_h",), img_size=(640, 480), golden_point=False):
        # -- arx robot env --
        self.arx = ARXRobotEnv(
            duration_per_step=1.0 / 20.0,
            min_steps=20,
            max_v_xyz=0.15, max_a_xyz=0.20,
            max_v_rpy=0.5, max_a_rpy=1.00,
            camera_type=camera_type,
            camera_view=camera_view,
            img_size=img_size,
        )

        obs = self.arx.reset()

        # -- golden points --
        # go
        self.go_points = [(346, 423), (371, 400), (399, 376),
                          (431, 353), (463, 333), (495, 317), (527, 306), (578, 293)]

        # return
        self.return_points = [(304, 401), (292, 371), (280, 330),
                              (260, 301), (226, 283), (188, 267), (148, 269), (116, 280)]

        # -- emergency stop --
        self.running = True

        self.golden_point = golden_point

        # -- initial pose information --
        self.x_r = 0.0
        self.y_r = 0.0
        self.theta_r = 0.0
        self.pose_log = []

        # -- update frequency --
        self.dt = 0.05

        # -------- emergency stop ------
        # self.running = True
        self.kb_thread = threading.Thread(
            target=self._safe_key_listener,
            daemon=True
        )
        self.kb_thread.start()

        # ---------- save path ----------
        self.save_root = Path(
            "/home/arx/Robotbase_base/data/camera_record"
        )
        self.save_root.mkdir(parents=True, exist_ok=True)

        ts = time.strftime("%Y%m%d_%H%M%S")
        self.save_dir = self.save_root / ts
        self.save_dir.mkdir()

        self.rgb_dir = self.save_dir / "rgb"
        self.depth_dir = self.save_dir / "depth"
        self.rgb_dir.mkdir()
        self.depth_dir.mkdir()

        # -- save content --
        self.save_frequency = 100
        self.action_log = []   # [(chx, chy, chz, duration)]
        self.rgb_frames = []
        self.frame_id = 0

        # -- default height --
        self.default_height = 15.0

        self.lift_to_default_height()

    def get_color_depth(self):
        frames = self.arx.get_camera(
            target_size=(640, 480), return_status=False)
        color = frames.get("camera_h_color")
        depth = frames.get("camera_h_aligned_depth_to_color")
        return color, depth

    # slowly raise up
    def lift_to_default_height(self):
        self.arx.step_lift(self.default_height)

    # change pixel to world point
    def pixel_to_pw(self, pixel, depth, return_=False):
        u, v = pixel
        z = depth_to_meters(float(depth[int(v), int(u)]))
        while z <= 0:
            depth = self.get_color_depth()[1]
            z = depth_to_meters(float(depth[int(v), int(u)]))
        # 像素 → 相机坐标
        x = (u - CX) * z / FX
        y = (v - CY) * z / FY
        Pc = np.array([x, y, z, 1.0], dtype=np.float64)

        # 相机 → ref → base_link
        if return_:
            Pw = T_CAM2REF_R @ Pc
        else:
            Pw = T_CAM2REF_L @ Pc
            Pw = Pw + BIAS_REF2CAM_L
        # Pw = Pw_right + BIAS_REF2CAM

        return Pw

    # motion
    def stop(self):
        msg = PosCmd()
        msg.chx = msg.chy = msg.chz = 0.0
        base_status = self.arx.get_robot_status().get("base")
        msg.height = float(
            base_status.height) if base_status is not None else 0.0
        msg.mode1 = 2
        self.arx.node.send_base_msg(msg)

    def run_for_1s(self, chx=0.0, chy=0.0, chz=0.0, duration=1.0, record=True):
        v = 0.24 * chx**2
        omega = chz * (2 * math.pi / 20.6)

        msg = PosCmd()
        msg.chx = chx
        msg.chy = chy
        msg.chz = chz
        base_status = self.arx.get_robot_status().get("base")
        msg.height = float(
            base_status.height) if base_status is not None else 0.0
        msg.mode1 = 1
        self.arx.node.send_base_msg(msg)

        if record and self.running:
            self.integrate_motion(v, omega, duration)

        self.stop()

        if record and self.running:
            self.action_log.append((chx, chy, chz, duration))

    def run_for_1s_return(self, chx=0.0, chy=0.0, chz=0.0, duration=1.0, record=True):
        msg = PosCmd()
        msg.chx = chx
        msg.chy = chy
        msg.chz = chz
        msg.mode1 = 1
        base_status = self.arx.get_robot_status().get("base")
        msg.height = float(
            base_status.height) if base_status is not None else 0.0
        self.arx.node.send_base_msg(msg)
        start_time = time.time()

        if duration < 0.1:
            time.sleep(duration)
            return

        else:
            while time.time() - start_time < duration and self.running:
                a = 1
            return

    def judge_goal(self, goal):
        judge_prompt = f"""Is there a {goal} in the picture? If you think there is no {goal}, output 'False'; if you think there is {goal}, output 'True'."""
        color, depth = self.get_color_depth()
        points, generated_content = predict_multi_points_from_rgb(
            color,
            text_prompt="",
            all_prompt=judge_prompt,
            base_url="http://172.28.102.11:22002/v1",
            model_name="Embodied-R1.5-SFT-0128",
            api_key="EMPTY",
            assume_bgr=False,
            return_raw=True
        )

        print(generated_content)

        if generated_content == "True":
            return True
        else:
            return False

    def nav_plan(self, user_instruction):
        self.turn_left(math.pi / 2.0)
        self.turn_right_corner()
        # if not self.judge_goal("center of red circular landmark on the ground"):
        # self.run_for_1s(chz=-0.5, duration=20.6/6.0)
        # if not self.golden_point:
        #     self.go_to_goal("center of red circular landmark on the ground")
        # else:
        #     actions = [(0.0, 0.0, -0.5, 0.7873), (1.0, 0.0, 0.0, 9.326)]
        #     for chx, chy, chz, duration in actions:
        #         self.run_for_1s(chx, chy, chz, duration)
        self.go_to_goal("center of red circular landmark on the ground")
        # action_return = self.go_to_table()
        self.go_to_table()
        # self.save_traj()
        print("Successfully save trajectory!")

        # return action_return

    def save_traj(self):
        # 覆盖保存（'w' 模式会覆盖原文件）
        with open('/home/arx/Arx_Lift/Testdata4Nav/data.json', 'w', encoding='utf-8') as f:
            json.dump(self.action_log, f, ensure_ascii=False,
                      indent=2)  # indent 使文件可读

    def load_traj(self):
        with open('/home/arx/Arx_Lift/Testdata4Nav/data.json', 'r', encoding='utf-8') as f:
            loaded_list = json.load(f)
        return loaded_list

    def turn_left(self, angle):
        print(f"Turn left 90°......")
        duration_time = 20.6 * float(angle / math.pi)
        self.run_for_1s(chz=0.5, duration=duration_time)

    def go_to_table(self):
        self.run_for_1s(chz=-0.5, duration=20.6 / 2.5)

        self.arx.step_lift(19.0)

        self.run_for_1s(chx=0.5, duration=6.5)

        # color, depth = self.get_color_depth()
        # points = self.detect_goal(color, "the brown round coaster on edge of the table")
        # goal_pw = self.pixel_to_pw(points[0], depth, return_=True)
        # goal_pw[0] += 0.25
        # goal_pw[1] -= 0.25
        # start = (0, 0)
        # goal = (goal_pw[0], -goal_pw[1])

        # path = [start, goal]
        # actions = path_to_actions(path)
        # actions = merge_forward_actions(actions)

        # cv2.circle(
        #     color,
        #     center=(int(points[0][0]), int(points[0][1])),
        #     radius=5,
        #     color=(0, 0, 255),
        #     thickness=-1  # -1 表示实心圆
        # )

        # cv2.imwrite("../Testdata4Nav/test_3.png", color)

        # for action, action_content in actions:
        #     if action == "rotate":
        #         if action_content <= 0:
        #             duration = max(float((-(action_content)/(0.5 * 2*math.pi / 20.6))), 0.0)
        #             self.run_for_1s(chz=-0.5, duration=duration)
        #             action_return = (0.5, duration)
        #         else:
        #             duration = action_content/(0.5 * 2*math.pi / 20.6)
        #             self.run_for_1s(chz=0.5, duration=action_content/(0.5 * 2*math.pi / 20.6))
        #             action_return = (-0.5, duration)

        # return action_return

        # time.sleep(10.0)
        # -- turn right end--

        # foward a little
        # self.run_for_1s(chx=0.5, duration=2.2)

    # intelligent turn right

    def turn_right_until_see_goal(self, goal, max_angle):
        # start_turn_right
        msg = PosCmd()
        msg.chx = 0.0
        msg.chy = 0.0
        msg.chz = -0.3
        msg.mode1 = 1
        self.arx.node.send_base_msg(msg)
        start_time = time.time()

        max_turn_time = max_angle / (0.6 * math.pi / 20.6)

#         detect_prompt = """Is there {goal}? If you think there is, ouput the point coordinates on the center of it; if you think there is not, the output point coordinates should be (1000, 1000).
#         Output format:
# Return the result in JSON format as:
# [
#   {"point_2d": [x, y]}
# ]""".replace("{goal}", goal)

        judge_prompt = f"""
Is there a {goal} in the picture? If you think there is no {goal}, output 'False'; if you think there is {goal}, output 'True'.
"""
        detect_prompt = prompt_format = (
            "Provide one or more points coordinate of objects region this sentence describes: "
            f"{goal}. "
            'The answer should be presented in JSON format as follows: [{"point_2d": [x, y]}].'
        )
        # detect_prompt = detect_prompt.replace("{goal}", goal)

        print(judge_prompt)

        print(detect_prompt)

        detect_flag = False

        while not detect_flag and time.time() - start_time < max_turn_time:
            color, depth = self.get_color_depth()
            # h, w = color.shape[:2]
            points, generated_content = predict_multi_points_from_rgb(
                color,
                text_prompt="",
                all_prompt=judge_prompt,
                base_url="http://172.28.102.11:22002/v1",
                model_name="Embodied-R1.5-SFT-0128",
                api_key="EMPTY",
                assume_bgr=False,
                return_raw=True
            )

            print(generated_content)

            if generated_content == "True":
                break

        while not detect_flag and time.time() - start_time < max_turn_time:
            color, depth = self.get_color_depth()
            h, w = color.shape[:2]
            points = predict_multi_points_from_rgb(
                color,
                text_prompt="",
                all_prompt=detect_prompt,
                base_url="http://172.28.102.11:22002/v1",
                model_name="Embodied-R1.5-SFT-0128",
                api_key="EMPTY",
                assume_bgr=False
            )

            print(points[0][0])

            if points[0][0] > w / 4.0 and points[0][0] < (w * 3.0) / 4.0:
                print(points[0])
                self.action_log.append(
                    (0.0, 0.0, -0.5, time.time() - start_time))
                detect_flag = True

        self.stop()
        if not detect_flag:
            self.action_log.append((0.0, 0.0, -0.5, time.time() - start_time))

        cv2.circle(
            color,
            center=(int(points[0][0]), int(points[0][1])),
            radius=5,
            color=(0, 0, 255),
            thickness=-1  # -1 表示实心圆
        )

        cv2.imwrite("../Testdata4Nav/test_2.png", color)

        return points, detect_flag, color

    def turn_right_corner(self):
        print("Turn right corner......")
        self.initialize_pose()
        color, depth = self.get_color_depth()
        prompt = """**Task**

Given an image captured from a top-mounted robot camera,Use 2D points to trace the movement trajectory as it moves.

**Trajectory requirements**

- Output **exactly 8 points** on the **ground (floor)** that form a single continuous trajectory.
- The **first point** must be at the **bottom center of the image**, representing the robot’s current position.
- The last point must be located on the right image boundary, below the vertical midpoint.
- The trajectory must represent **a clear forward motion first, followed by a right turn**.
- The **first 2–3 points** should lie approximately on a **straight forward path** before any noticeable rightward deviation.
- The right turn should **start later**, not immediately near the starting point.
Output format:
Return the result in JSON format as:
[
  {"point_2d": [x, y]}
]"""

        points = predict_multi_points_from_rgb(
            color,
            text_prompt="",
            all_prompt=prompt,
            base_url="http://172.28.102.11:22014/v1",
            model_name="Qwen3-VL-8B-Instruct",
            api_key="EMPTY",
            temperature=0.2
            # assume_bgr=False
        )

        order_num = 0.0

        revised_points = []

        for (u, v) in points:
            u += 80
            v += 30
            u = min(638, u)
            v = min(478, v)
            cv2.circle(
                color,
                center=(int(u), int(v)),
                radius=5,
                color=(order_num, order_num, 255 - order_num),
                thickness=-1  # -1 表示实心圆
            )
            order_num += 30
            revised_points.append((u, v))

        cv2.imwrite("../Testdata4Nav/test_1.png", color)

        path_xy = []

        # -- pixel to wolrd point --
        if self.golden_point:
            revised_points = self.go_points
        for point in revised_points:
            Pw = self.pixel_to_pw(point, depth)
            path_xy.append((Pw[0], Pw[1]))

        print(path_xy)

        actions = [(0.7359800721939873, 0.0, 0.6557183655386088, 0.05), (0.7359800721939873, 0.0, 0.6557183655386088, 0.05), (0.7359800721939873, 0.0, 0.6557183655386088, 0.05), (0.7359800721939873, 0.0, 0.6557183655386088, 0.05), (0.7359800721939873, 0.0, 0.6557183655386088, 0.05), (0.7359800721939873, 0.0, 0.6557183655386088, 0.05), (0.7359800721939873, 0.0, 0.6557183655386088, 0.05), (0.7359800721939873, 0.0, 0.6557183655386088, 0.05), (0.7359800721939873, 0.0, 0.6557183655386088, 0.05), (0.7359800721939873, 0.0, 0.6557183655386088, 0.05), (0.7359800721939873, 0.0, 0.6557183655386088, 0.05), (0.7359800721939873, 0.0, 0.6557183655386088, 0.05), (0.7359800721939873, 0.0, 0.6557183655386088, 0.05), (0.7359800721939873, 0.0, 0.6557183655386088, 0.05), (0.7359800721939873, 0.0, 0.6308599694777695, 0.05), (0.7359800721939873, 0.0, 0.5302139449429619, 0.05), (0.7387243656990787, 0.0, 0.44476455976276025, 0.05), (0.7469198024600913, 0.0, 0.3724194716883983, 0.05), (0.7539055944988764, 0.0, 0.31137645857458107, 0.05), (0.7598456410355648, 0.0, 0.25991466608148034, 0.05), (0.7648843664373753, 0.0, 0.2165758874626728, 0.05), (0.7691483995150151, 0.0, 0.18012279384835456, 0.05), (0.7727483085640727, 0.0, 0.1495042607855178, 0.05), (0.7757802984122815, 0.0, 0.12382654376727571, 0.05), (0.7783278163083042, 0.0, 0.10232925215254925, 0.05), (0.7804630392595283, 0.0, 0.08436525429707528, 0.05), (0.7822482314944901, 0.0, 0.06938380678856716, 0.05), (0.7837369705046264, 0.0, 0.05691633515113945, 0.05), (0.78497524586555, 0.0, 0.04656440372715104, 0.05), (0.7860024382242343, 0.0, 0.03798950164752962, 0.05), (0.7868521874208758, 0.0, 0.030904343242346997, 0.05), (0.7875531593074567, 0.0, 0.02506543814609773, 0.05), (0.7881297208218957, 0.0, 0.020266731559623462, 0.05), (0.7886025325262471, 0.0, 0.016334151039129226, 0.05), (0.7889890672764157, 0.0, 0.013120924755367797, 0.05), (0.7893040630569199, 0.0, 0.010503558980702498, 0.05), (0.7895599173472984, 0.0, 0.008378380869684253, 0.05), (0.7897670297229197, 0.0, 0.00665856738260768, 0.05), (0.7899340987532673, 0.0, 0.00527159322723957, 0.05), (0.790068378656832, 0.0, 0.0041570405558227375, 0.05), (0.7901759006093091, 0.0, 0.0032647213116185833, 0.05), (0.7902616630829891, 0.0, 0.0025530699257863693, 0.05), (0.7903297951199951, 0.0, 0.0019877697940891502, 0.05), (0.7903836960090149, 0.0, 0.001540581824084459, 0.05), (0.7904261544423029, 0.0, 0.0011883474996625578, 0.05), (0.7904594498745426, 0.0, 0.0009121424870832622, 0.05), (0.7904854384850281, 0.0, 0.0006965599033123015, 0.05), (0.7905056258568821, 0.0, 0.0005291050608904999, 0.05), (0.7905212282290962, 0.0, 0.00039968585487784006, 0.05), (0.7905332239465119, 0.0, 0.00030018501597093585, 0.05), (0.7905423965271323, 0.0, 0.0002241022594684438, 0.05), (0.7905493705830753, 0.0, 0.00016625594517708777, 0.05), (0.7905546416689879, 0.0, 0.00012253525618447365, 0.05), (0.7905586009878878, 0.0, 8.969512750117092e-05, 0.05), (0.7905615557573547, 0.0, 6.51872291339175e-05, 0.05), (0.7905637459271408, 0.0, 4.7021248855219786e-05, 0.05), (0.790565357841008, 0.0, 3.36515433211121e-05, 0.05), (0.790566535349587, 0.0, 2.3884945048969867e-05, 0.05), (0.790567388805945, 0.0, 1.6806139140994978e-05, 0.05), (0.790568002310201, 0.0, 1.171756784925628e-05, 0.05), (0.7905684395128305, 0.0, 8.091292645988442e-06, 0.05), (0.7905687482373129, 0.0, 5.530650665473234e-06, 0.05), (0.7905689641405694, 0.0, 3.739892916698685e-06, 0.05), (0.7905691135934406, 0.0, 2.500292229865169e-06, 0.05), (0.7905692159325102, 0.0, 1.6514657084830435e-06, 0.05), (0.7905692852082669, 0.0, 1.0768748557084562e-06, 0.05), (0.7905693315322899, 0.0, 6.926515828046543e-07, 0.05), (0.7905693621073501, 0.0, 4.3905424115765526e-07, 0.05), (0.7905693820085484, 0.0, 2.739886352372281e-07, 0.05), (0.7905693947704512, 0.0, 1.6813816614008452e-07, 0.05), (0.7905694028242698, 0.0, 1.013377472936186e-07, 0.05), (0.7905694078201186, 0.0, 5.990090656830676e-08, 0.05), (0.7905694108620159, 0.0, 3.467063676713075e-08, 0.05), (0.7905694126772781, 0.0, 1.961439121865431e-08, 0.05), (0.7905694137371093, 0.0, 1.0823881280079142e-08, 0.05), (0.790569414341301, 0.0, 5.8125620809744785e-09, 0.05), (0.7905694146768563, 0.0, 3.029381344994486e-09, 0.05), (0.7905694148579335, 0.0, 1.5274804545550331e-09, 0.05), (0.7905694149525883, 0.0, 7.423904576685489e-10, 0.05), (0.790569415000346, 0.0, 3.462751395798129e-10, 0.05), (0.7905694150235044, 0.0, 1.5419396003426347e-10, 0.05), (0.7905694150342423, 0.0, 6.51310611982815e-11, 0.05), (0.7905694150389734, 0.0, 2.589001564393232e-11, 0.05), (0.7905694150409383, 0.0, 9.592799285876133e-12, 0.05), (0.7905694150417005, 0.0, 3.2703551161011395e-12, 0.05), (0.790569415041973, 0.0, 1.0107669579848877e-12, 0.05), (0.7905694150420615, 0.0, 2.7743478664632754e-13, 0.05), (0.790569415042087, 0.0, 6.470467755982645e-14, 0.05), (0.7905694150420943, 0.0, 4.940617080535903e-15, 0.05), (0.7905694150420944, 0.0, 3.713803511445171e-15, 0.05), (0.7359800721939873, 0.0, -0.6557183655386088, 0.05), (0.7359800721939873, 0.0, -0.6557183655386088, 0.05), (0.7359800721939873, 0.0, -0.6557183655386088, 0.05), (0.7359800721939873, 0.0, -0.6557183655386088, 0.05), (0.7359800721939873, 0.0, -0.6557183655386088, 0.05), (0.7359800721939873, 0.0, -0.6557183655386088, 0.05), (0.7359800721939873, 0.0, -0.6557183655386088, 0.05), (0.7359800721939873, 0.0, -0.6557183655386088, 0.05), (0.7359800721939873, 0.0, -0.6557183655386088, 0.05), (0.7359800721939873, 0.0, -0.6557183655386088, 0.05), (0.7359800721939873, 0.0, -0.6557183655386088, 0.05), (0.7359800721939873, 0.0, -0.6557183655386088, 0.05), (0.7359800721939873, 0.0, -0.6557183655386088, 0.05), (0.7359800721939873, 0.0, -0.6557183655386088, 0.05), (0.7359800721939873, 0.0, -0.6557183655386088, 0.05), (0.7359800721939873, 0.0, -0.6557183655386088, 0.05), (0.7359800721939873, 0.0, -0.6557183655386088, 0.05), (0.7359800721939873, 0.0, -0.6557183655386088, 0.05), (0.7359800721939873, 0.0, -0.6557183655386088, 0.05), (0.7359800721939873, 0.0, -0.6557183655386088, 0.05),
                   (0.7359800721939873, 0.0, -0.6557183655386088, 0.05), (0.7359800721939873, 0.0, -0.6557183655386088, 0.05), (0.7359800721939873, 0.0, -0.6557183655386088, 0.05), (0.7359800721939873, 0.0, -0.6557183655386088, 0.05), (0.7359800721939873, 0.0, -0.6557183655386088, 0.05), (0.7359800721939873, 0.0, -0.6557183655386088, 0.05), (0.7359800721939873, 0.0, -0.6557183655386088, 0.05), (0.7359800721939873, 0.0, -0.6557183655386088, 0.05), (0.7359800721939873, 0.0, -0.6557183655386088, 0.05), (0.7359800721939873, 0.0, -0.6557183655386088, 0.05), (0.7359800721939873, 0.0, -0.6557183655386088, 0.05), (0.7359800721939873, 0.0, -0.6557183655386088, 0.05), (0.7359800721939873, 0.0, -0.6557183655386088, 0.05), (0.7359800721939873, 0.0, -0.6557183655386088, 0.05), (0.7359800721939873, 0.0, -0.6557183655386088, 0.05), (0.7359800721939873, 0.0, -0.6557183655386088, 0.05), (0.7359800721939873, 0.0, -0.6557183655386088, 0.05), (0.7359800721939873, 0.0, -0.6557183655386088, 0.05), (0.7359800721939873, 0.0, -0.6557183655386088, 0.05), (0.7359800721939873, 0.0, -0.6557183655386088, 0.05), (0.7359800721939873, 0.0, -0.6557183655386088, 0.05), (0.7359800721939873, 0.0, -0.6557183655386088, 0.05), (0.7359800721939873, 0.0, -0.6557183655386088, 0.05), (0.7359800721939873, 0.0, -0.6557183655386088, 0.05), (0.7359800721939873, 0.0, -0.6557183655386088, 0.05), (0.7359800721939873, 0.0, -0.6557183655386088, 0.05), (0.7359800721939873, 0.0, -0.6557183655386088, 0.05), (0.7359800721939873, 0.0, -0.6557183655386088, 0.05), (0.7359800721939873, 0.0, -0.6557183655386088, 0.05), (0.7359800721939873, 0.0, -0.6557183655386088, 0.05), (0.7359800721939873, 0.0, -0.6557183655386088, 0.05), (0.7359800721939873, 0.0, -0.6557183655386088, 0.05), (0.7359800721939873, 0.0, -0.6557183655386088, 0.05), (0.7359800721939873, 0.0, -0.6557183655386088, 0.05), (0.7359800721939873, 0.0, -0.6557183655386088, 0.05), (0.7359800721939873, 0.0, -0.6557183655386088, 0.05), (0.7359800721939873, 0.0, -0.6557183655386088, 0.05), (0.7359800721939873, 0.0, -0.6557183655386088, 0.05), (0.7359800721939873, 0.0, -0.6557183655386088, 0.05), (0.7359800721939873, 0.0, -0.6557183655386088, 0.05), (0.7359800721939873, 0.0, -0.6557183655386088, 0.05), (0.7359800721939873, 0.0, -0.6557183655386088, 0.05), (0.7359800721939873, 0.0, -0.6557183655386088, 0.05), (0.7359800721939873, 0.0, -0.6557183655386088, 0.05), (0.7359800721939873, 0.0, -0.6557183655386088, 0.05), (0.7359800721939873, 0.0, -0.6557183655386088, 0.05), (0.7359800721939873, 0.0, -0.6557183655386088, 0.05), (0.7359800721939873, 0.0, -0.6557183655386088, 0.05), (0.7359800721939873, 0.0, -0.6557183655386088, 0.05), (0.7359800721939873, 0.0, -0.6557183655386088, 0.05), (0.7359800721939873, 0.0, -0.6557183655386088, 0.05), (0.7359800721939873, 0.0, -0.6557183655386088, 0.05), (0.7359800721939873, 0.0, -0.6557183655386088, 0.05), (0.7359800721939873, 0.0, -0.6557183655386088, 0.05), (0.7359800721939873, 0.0, -0.6557183655386088, 0.05), (0.7359800721939873, 0.0, -0.6557183655386088, 0.05), (0.7359800721939873, 0.0, -0.6557183655386088, 0.05), (0.7359800721939873, 0.0, -0.6557183655386088, 0.05), (0.7359800721939873, 0.0, -0.6557183655386088, 0.05), (0.7359800721939873, 0.0, -0.6557183655386088, 0.05), (0.7359800721939873, 0.0, -0.6557183655386088, 0.05), (0.7359800721939873, 0.0, -0.6557183655386088, 0.05), (0.7359800721939873, 0.0, -0.6557183655386088, 0.05), (0.7359800721939873, 0.0, -0.6557183655386088, 0.05), (0.7359800721939873, 0.0, -0.6557183655386088, 0.05), (0.7359800721939873, 0.0, -0.6557183655386088, 0.05), (0.7359800721939873, 0.0, -0.6557183655386088, 0.05), (0.7359800721939873, 0.0, -0.6557183655386088, 0.05), (0.7359800721939873, 0.0, -0.6557183655386088, 0.05), (0.7359800721939873, 0.0, -0.6557183655386088, 0.05), (0.7359800721939873, 0.0, -0.6557183655386088, 0.05), (0.7359800721939873, 0.0, -0.6557183655386088, 0.05), (0.7359800721939873, 0.0, -0.6557183655386088, 0.05), (0.7359800721939873, 0.0, -0.6557183655386088, 0.05), (0.7359800721939873, 0.0, -0.6557183655386088, 0.05), (0.7359800721939873, 0.0, -0.6557183655386088, 0.05), (0.7359800721939873, 0.0, -0.6557183655386088, 0.05), (0.7359800721939873, 0.0, -0.6557183655386088, 0.05), (0.7359800721939873, 0.0, -0.6557183655386088, 0.05), (0.7359800721939873, 0.0, -0.6557183655386088, 0.05), (0.7359800721939873, 0.0, -0.6557183655386088, 0.05), (0.7359800721939873, 0.0, -0.6557183655386088, 0.05), (0.7359800721939873, 0.0, -0.6557183655386088, 0.05), (0.7359800721939873, 0.0, -0.6557183655386088, 0.05), (0.7359800721939873, 0.0, -0.6557183655386088, 0.05), (0.7359800721939873, 0.0, -0.6557183655386088, 0.05), (0.7359800721939873, 0.0, -0.6557183655386088, 0.05), (0.7359800721939873, 0.0, -0.6557183655386088, 0.05), (0.7359800721939873, 0.0, -0.6557183655386088, 0.05), (0.7359800721939873, 0.0, -0.6557183655386088, 0.05), (0.7359800721939873, 0.0, -0.6557183655386088, 0.05), (0.7359800721939873, 0.0, -0.6557183655386088, 0.05), (0.7359800721939873, 0.0, -0.6557183655386088, 0.05), (0.7359800721939873, 0.0, -0.6557183655386088, 0.05), (0.7359800721939873, 0.0, -0.6557183655386088, 0.05), (0.7359800721939873, 0.0, -0.6557183655386088, 0.05), (0.7359800721939873, 0.0, -0.6557183655386088, 0.05), (0.7359800721939873, 0.0, -0.6557183655386088, 0.05), (0.7359800721939873, 0.0, -0.6557183655386088, 0.05), (0.7359800721939873, 0.0, -0.6557183655386088, 0.05), (0.7359800721939873, 0.0, -0.6557183655386088, 0.05), (0.7359800721939873, 0.0, -0.6557183655386088, 0.05), (0.7359800721939873, 0.0, -0.6557183655386088, 0.05), (0.7359800721939873, 0.0, -0.6557183655386088, 0.05), (0.7359800721939873, 0.0, -0.6557183655386088, 0.05), (0.7359800721939873, 0.0, -0.6557183655386088, 0.05), (0.7359800721939873, 0.0, -0.6557183655386088, 0.05), (0.7359800721939873, 0.0, -0.6557183655386088, 0.05), (0.7359800721939873, 0.0, -0.6557183655386088, 0.05), (0.7359800721939873, 0.0, -0.6557183655386088, 0.05), (0.7359800721939873, 0.0, -0.6557183655386088, 0.05), (0.7359800721939873, 0.0, -0.6557183655386088, 0.05)]

        turn_right_action_log = self.follow_path(
            path_xy, lookahead=0.12, v_max=0.15, v_min=0.13, reach_dis=0.09, show_index=False)

        # for chx, chy, chz, duration in actions:
        #     msg = PosCmd()
        #     msg.chx = chx
        #     msg.chy = chy
        #     msg.chz = chz
        #     base_status = self.arx.get_robot_status().get("base")
        #     msg.height = float(
        #         base_status.height) if base_status is not None else 0.0
        #     msg.mode1 = 1
        #     self.arx.node.send_base_msg(msg)
        #     self.action_log.append((chx, chy, chz, duration))
        #     time.sleep(self.dt)

        # self.stop()

        # return turn_right_action_log

    def go_to_goal(self, goal, left_side=False):
        print(f"Go to goal {goal}......")
        color, depth = self.get_color_depth()
        prompt = """Provide one or more points coordinate of objects region this sentence describes: {goal}.
        Output format: Return the result in JSON format as:
        [ 
            {"point_2d": [x, y]}
        ]
        """.replace("{goal}", goal)

        points = predict_multi_points_from_rgb(
            color,
            text_prompt="",
            all_prompt=prompt,
            base_url="http://172.28.102.11:22002/v1",
            model_name="Embodied-R1.5-SFT-0128",
            api_key="EMPTY",
            assume_bgr=False
        )

        cv2.circle(
            color,
            center=(int(points[0][0]), int(points[0][1])),
            radius=5,
            color=(0, 0, 255),
            thickness=-1  # -1 表示实心圆
        )

        # cv2.imwrite("../Testdata4Nav/test_2.png", color)

        goal_pw = self.pixel_to_pw(points[0], depth, return_=True)
        if left_side:
            goal_pw[0] += 0.25
            goal_pw[1] -= 0.25
            cv2.imwrite("../Testdata4Nav/test_3.png", color)
        else:
            goal_pw[0] += 0.25
            goal_pw[1] -= 0.25
            cv2.imwrite("../Testdata4Nav/test_2.png", color)
        start = (0, 0)
        goal = (goal_pw[0], -goal_pw[1])

        path = [start, goal]
        actions = path_to_actions(path)
        actions = merge_forward_actions(actions)

        # -- move to goal --
        for action, action_content in actions:
            if action == "forward":
                self.run_for_1s(chx=1.0, duration=(action_content-0.10)/0.247)
            elif action == "rotate":
                if action_content <= 0:
                    self.run_for_1s(
                        chz=-0.5, duration=max(float((-action_content/(0.5 * 2*math.pi / 20.6))) - 0.5, 0.0))
                else:
                    self.run_for_1s(
                        chz=0.5, duration=action_content/(0.5 * 2*math.pi / 20.6))

    def detect_goal(self, color, goal):
        prompt = """Provide one point coordinate of object region this sentence describes: {goal}.
        Output format: Return the result in JSON format as:
        [ 
            {"point_2d": [x, y]}
        ]
        """.replace("{goal}", goal)

        points = predict_multi_points_from_rgb(
            color,
            text_prompt="",
            all_prompt=prompt,
            base_url="http://172.28.102.11:22002/v1",
            model_name="Embodied-R1.5-SFT-0128",
            api_key="EMPTY",
            assume_bgr=False
        )

        return points

    def turn_left_corner(self, turn_right_action_log):
        print("Turn left corner......")
        self.initialize_pose()
        color, depth = self.get_color_depth()
        # -- one --
#         prompt = """Task
# Given an image captured from a top-mounted robot camera, use 2D points to trace the movement trajectory as it moves.
# Trajectory requirements
# Output exactly 8 points on the ground (floor) that form a single continuous trajectory.
# The first point must be at the bottom center of the image, representing the robot’s current position.
# The last point must be located on the left image boundary, below the vertical midpoint (to complete the bypass).
# The trajectory must represent a clear forward motion first, followed by a left turn to navigate around the table on the left.
# The first 2–3 points should lie approximately on a straight forward path to establish clearance before initiating the turn.
# The left turn should start mid-trajectory, angling toward the left boundary to successfully bypass the obstacle.
# Surface Constraint: All points, especially the final destination point, must be located strictly within the blue floor area. Avoid any points overlapping with the table or non-floor surfaces.
# Output format:
# Return the result in JSON format"""

#         points = predict_multi_points_from_rgb(
#             color,
#             text_prompt="",
#             all_prompt=prompt,
#             base_url="http://172.28.102.11:22002/v1",
#             model_name="Embodied-R1.5-SFT-0128",
#             api_key="EMPTY",
#             temperature=0.2,
#             assume_bgr=False
#         )

        # -- two --
        # prompt = """
        #     Task
        # Given an image captured from a top-mounted robot camera, use 2D points to trace the movement trajectory as it moves.
        # Trajectory requirements
        # Output exactly 8 points on the ground (floor) that form a single continuous trajectory.
        # The first point must be at the bottom center of the image.
        # The last point must be located on the left image boundary, below the vertical midpoint (to complete the bypass).
        # The trajectory must represent a clear forward motion first, followed by a left turn to navigate around the table on the left.
        # The first 2–3 points should lie approximately on a straight forward path to establish clearance before initiating the turn.
        # The left turn should start mid-trajectory, angling toward the left boundary to successfully bypass the obstacle.
        # Smooth Curve: The path should form a continuous smooth arc leaning left, avoiding any straight vertical segments at the start.
        # Surface Constraint: All points, especially the final destination point, must be located strictly within the blue floor area. Avoid any points overlapping with the table or non-floor surfaces.
        # Output format:
        # Return the result in JSON format
        # """

        # points_ = predict_multi_points_from_rgb(
        #     color,
        #     text_prompt="",
        #     all_prompt=prompt,
        #     base_url="http://172.28.102.11:22002/v1",
        #     model_name="Embodied-R1.5-SFT-0128",
        #     api_key="EMPTY",
        #     temperature=0.2,
        #     assume_bgr=False
        # )
        # w, h = color.shape[:2]
        # points=points_
        # points=refine_trajectory_strict(points_,w,h)

        # -- three --
    #     prompt = """Task
    # Given an image captured from a top-mounted robot camera, use 2D points to trace the movement trajectory as it moves.
    # Trajectory requirements
    # Output exactly 8 points on the ground (floor) that form a single continuous trajectory.
    # The first point must be at the bottom center of the image, representing the robot’s current position.
    # The last point must be located on the left image boundary, below the vertical midpoint (to complete the bypass).
    # The trajectory must represent a clear forward motion first, followed by a left turn to navigate around the table on the left.
    # The first 2–3 points should lie approximately on a straight forward path to establish clearance before initiating the turn.
    # The left turn should start mid-trajectory, angling toward the left boundary to successfully bypass the obstacle.
    # Surface Constraint: All points, especially the final destination point, must be located strictly within the blue floor area. Avoid any points overlapping with the table or non-floor surfaces.
    # Output format:
    # Return the result in JSON format"""

    #     points_all = []
    #     for i in range(10):
    #         points = predict_multi_points_from_rgb(
    #             color,
    #             text_prompt="",
    #             all_prompt=prompt,
    #             base_url="http://172.28.102.11:22002/v1",
    #             model_name="Embodied-R1.5-SFT-0128",
    #             api_key="EMPTY",
    #             assume_bgr=False
    #         )
    #         points_all.append(points)

    #     points_all_np = np.array(points_all, dtype=np.float32)
    #     # shape: (10, 8, 2)

    #     # 对 10 次取平均
    #     points_avg = points_all_np.mean(axis=0)
    #     # shape: (8, 2)

    #     # 转回 python list
    #     points = [(float(u), float(v)) for u, v in points_avg]

    #     w, h = color.shape[:2]
    #     points=refine_trajectory_strict(points,w,h)

    #     order_num = 0.0

    #     revised_points = []

    #     for (u, v) in points:
    #         cv2.circle(
    #             color,
    #             center=(int(u), int(v)),
    #             radius=5,
    #             color=(order_num, order_num, 255 - order_num),
    #             thickness=-1  # -1 表示实心圆
    #         )
    #         order_num += 30
    #         revised_points.append((u, v))

    #     cv2.imwrite("../Testdata4Nav/test_4.png", color)

        path_xy = []

        if self.golden_point:
            revised_points = self.return_points

        # -- pixel to wolrd point --
        for point in revised_points:
            Pw = self.pixel_to_pw(point, depth, return_=True)
            path_xy.append((Pw[0]+0.24, Pw[1]-0.24))

        print(path_xy[:7])

        theta_turn = self.follow_path(
            path_xy[:7], lookahead=0.12, v_max=0.15, v_min=0.13, reach_dis=0.09, show_index=False, return_=True)

        # -- four --
        # for chx, chy, chz, duration in reversed(turn_right_action_log):
        #     # ignore still motion
        #     if abs(chx) < 1e-3 and abs(chz) < 1e-3:
        #         continue

        #     # print((chx, chy, -chz, duration))

        #     self.run_for_1s_return(chx, chy, -chz, duration)

        #     if not self.running:
        #         break

        return theta_turn

    # emergency read keyboard
    def keyboard_listener(self):
        while self.running:
            try:
                ch = get_key()
            except Exception:
                continue

            if ch == 'q':
                # print("Key 'q' pressed! Emergency stop!")
                # raise RuntimeError("Key 'q' pressed! Emergency stop!")
                self.running = False
                self.arx.close()
                # self.running = False
                self.stop()
                break

    # compute pose
    def initialize_pose(self):
        self.x_r = 0.0
        self.y_r = 0.0
        self.theta_r = 0.0
        self.pose_log = []

    def update_pose(self, v, omega):

        # 更新机器人位姿
        self.x_r += v * math.cos(self.theta_r) * self.dt
        self.y_r += v * math.sin(self.theta_r) * self.dt
        self.theta_r += omega * self.dt

        self.pose_log.append((self.x_r, self.y_r, self.theta_r))

    def integrate_motion(self, v, omega, duration):
        t = 0.0
        while t < duration and self.running:
            self.update_pose(v, omega)
            time.sleep(self.dt)
            t += self.dt

    # get robot pose
    def get_robot_pose(self):
        return self.x_r, self.y_r, self.theta_r

    def get_lookahead_point(self, path_xy, lookahead, _index):
        x_r, y_r, theta_r = self.get_robot_pose()
        for index, (x_g, y_g) in enumerate(path_xy):
            # change coordinate
            x_t = math.cos(-theta_r)*(x_g - x_r) - \
                math.sin(-theta_r)*(y_g - y_r)
            y_t = math.sin(-theta_r)*(x_g - x_r) + \
                math.cos(-theta_r)*(y_g - y_r)
            dist = math.hypot(x_t, y_t)
            if dist >= lookahead and x_t > 0 and index >= _index:
                return x_t, y_t, dist, index
        # return the final point
        x_t = math.cos(-theta_r)*(path_xy[-1][0] - x_r) - \
            math.sin(-theta_r)*(path_xy[-1][1] - y_r)
        y_t = math.sin(-theta_r)*(path_xy[-1][0] - x_r) + \
            math.cos(-theta_r)*(path_xy[-1][1] - y_r)
        dist = math.hypot(x_t, y_t)
        return x_t, y_t, dist, len(path_xy) - 1

    # pure pursuite follow path
    def follow_path(self, path_xy, lookahead=0.6, v_max=0.12, v_min=0.10, omega_max=0.2, reach_dis=0.08, return_=False, show_index=False):
        # reset pose
        # self.x_r = 0.0
        # self.y_r = 0.0
        # self.theta_r = 0.0

        index = 0
        rate = self.dt
        final_count = 0
        if return_:
            max_final_count = (math.hypot(abs(
                path_xy[-2][0] - path_xy[-1][0]), abs(path_xy[-2][1] - path_xy[-1][1])) / 0.05)
        else:
            max_final_count = (math.hypot(abs(
                path_xy[-2][0] - path_xy[-1][0]), abs(path_xy[-2][1] - path_xy[-1][1])) / 0.06)

        action_log = []
        while self.running:
            # 获取目标点
            x_t, y_t, dist, index = self.get_lookahead_point(
                path_xy, lookahead, index)
            if index == len(path_xy) - 1:
                final_count += 1
            if show_index:
                print(index)

            # 非常接近终点，允许真正停下
            if dist < reach_dis or final_count > max_final_count:
                break

            # Pure Pursuit 曲率
            curvature = 2 * y_t / (dist**2 + 1e-6)
            omega = 1.2 * curvature

            # 原始速度衰减
            v = v_max * math.exp(-abs(omega))

            # -------- 独立限制线速度和角速度 --------
            v = max(min(v, v_max), v_min)             # v ∈ [v_min, v_max]
            # omega ∈ [-omega_max, omega_max]
            omega = max(min(omega, omega_max), -omega_max)
            # -----------------------------------------

            # 遥控信号映射
            msg = PosCmd()
            msg.chx = math.sqrt(v / 0.24)          # 前后速度
            msg.chz = omega / (2 * math.pi / 20.6)  # 正数向左
            base_status = self.arx.get_robot_status().get("base")
            msg.height = float(
                base_status.height) if base_status is not None else 0.0
            msg.mode1 = 1
            self.arx.node.send_base_msg(msg)
            self.action_log.append((msg.chx, msg.chy, msg.chz, rate))
            action_log.append((msg.chx, msg.chy, msg.chz, rate))
            # print(math.sqrt(v / 0.24))
            # print(omega / (2 * math.pi / 20.6))

            # 更新位姿
            self.update_pose(v, omega)
            time.sleep(rate)

        self.stop()
        return action_log

    # Motion Inversion with Forward-only Constraint
    def motion_inversion(self):
        # turn back
        # self.run_for_1s(chz=-0.5, duration=20.6)
        # print(self.action_log)
        go_action_log = self.load_traj()
        action_log = go_action_log[1:-2].copy()
        # print(self.action_log)
        # action_log.append(self.action_log[-4])
        for chx, chy, chz, duration in reversed(action_log):
            # ignore still motion
            if abs(chx) < 1e-3 and abs(chz) < 1e-3:
                continue

            # print((chx, chy, -chz, duration))

            self.run_for_1s_return(chx, chy, -chz, duration)

            if not self.running:
                break

    def _safe_key_listener(self):
        """
        safe key listener
        """
        fd = sys.stdin.fileno()
        old_settings = termios.tcgetattr(fd)

        try:
            while self.running:
                # 0.05s超时轮询，不阻塞主线程
                rlist, _, _ = select.select([sys.stdin], [], [], 0.05)
                if rlist:
                    ch = sys.stdin.read(1)
                    if ch == 'k':
                        print("\n[Emergency Stop] 'k' pressed.")
                        self.arx.close()
                        self.running = False
                        self.stop()
                        break
        finally:
            # 确保退出时终端状态恢复
            termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)

    def nav_back(self):
        """
        return back
        """
        # -- adjust face --
        # self.run_for_1s(chz=action_return[0], duration=action_return[1])

        # -- step back a little --
        self.run_for_1s(chx=-0.5, duration=7.2)

        # -- turn right --
        self.run_for_1s(chz=-0.5, duration=20.6/2.0)

        # -- go to table corner --
        self.run_for_1s(chx=0.5, duration=12.5)

        # -- turn left corner --
        theta_turn = self.turn_left_corner()

        # -- turn left to see landmark --
        self.run_for_1s(chz=0.5, duration=20.6/4.0)

        # -- go to landmark
        self.go_to_goal(
            "center of red circular landmark on the ground", left_side=True)

        # -- turn left to face the table --
        self.run_for_1s(chz=0.5, duration=((20.6*5.0)/12.0-0.5))

    def back_origin_path(self):

        print("Go back......")

        # print(action_return)

        # self.run_for_1s(chz=action_return[0], duration=action_return[1])

        self.run_for_1s(chx=-0.5, duration=6.5)

        self.run_for_1s(chz=-0.5, duration=20.6 - 20.6 / 2.5)

        self.motion_inversion()

        self.run_for_1s(chx=0.5, duration=5.0)

        self.run_for_1s(chz=0.5, duration=10.3)

    def go_follow_golden_path(self):
        # self.run_for_1s(chz=0.5, duration=10.3)
        trajectory = self.load_traj()
        self.run_for_1s(chx=trajectory[0][0], chy=trajectory[0]
                        [1], chz=trajectory[0][2], duration=trajectory[0][3])
        for chx, chy, chz, duration in trajectory[1:-4]:
            msg = PosCmd()
            msg.chx = chx
            msg.chy = chy
            msg.chz = chz
            base_status = self.arx.get_robot_status().get("base")
            msg.height = float(
                base_status.height) if base_status is not None else 0.0
            msg.mode1 = 1
            self.arx.node.send_base_msg(msg)
            # self.action_log.append((chx, chy, chz, duration))
            time.sleep(self.dt)

        self.stop()
        for i in reversed(range(4)):
            i += 1
            self.run_for_1s(chx=trajectory[-i][0], chy=trajectory[-i]
                            [1], chz=trajectory[-i][2], duration=trajectory[-i][3])

    def return_follow_reversed_path(self):
        trajectory = self.load_traj()
        self.run_for_1s(chx=-trajectory[-1][0], chy=trajectory[-1]
                        [1], chz=trajectory[-1][2], duration=trajectory[-1][3])
        self.run_for_1s(chx=-trajectory[-2][0], chy=trajectory[-2][1],
                        chz=trajectory[-2][2], duration=20.6 - trajectory[-2][3])
        self.run_for_1s(chx=trajectory[-3][0], chy=trajectory[-3]
                        [1], chz=trajectory[-3][2], duration=trajectory[-3][3])
        self.run_for_1s(chx=trajectory[-4][0], chy=trajectory[-4]
                        [1], chz=-trajectory[-4][2], duration=trajectory[-4][3])
        for chx, chy, chz, duration in reversed(trajectory[1:-4]):
            msg = PosCmd()
            msg.chx = chx
            msg.chy = chy
            msg.chz = -chz
            base_status = self.arx.get_robot_status().get("base")
            msg.height = float(
                base_status.height) if base_status is not None else 0.0
            msg.mode1 = 1
            self.arx.node.send_base_msg(msg)
            # self.action_log.append((chx, chy, -chz, duration))
            time.sleep(self.dt)

        self.stop()
        self.run_for_1s(chx=0.5, duration=5.0)

        self.run_for_1s(chz=0.5, duration=10.3)

        self.run_for_1s(chx=0.5, duration=1.0)
