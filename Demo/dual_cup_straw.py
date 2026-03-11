
import time
import sys
sys.path.append("../ARX_Realenv/ROS2")  # noqa
from arx_ros2_env import ARXRobotEnv  # noqa
import numpy as np
from demo_utils import step_base_duration
from single_arm_pick_place import single_arm_pick_place

OPEN = -3.4
CLOSE = -2.2


def special_give(arx: ARXRobotEnv, cup_side: str):
    straw_side = "right" if cup_side == "left" else "left"
    success, error_message = arx.set_special_mode(1, side=straw_side)
    if not success:
        raise RuntimeError(f"Failed to home {straw_side} arm: {error_message}")
    step_base_duration(arx, vx=0.5, vy=0.0, vz=-0.0, duration=2.0)
    # 往前递杯子
    if cup_side == "left":
        give_action = {cup_side: np.array(
            [0.4, 0, 0.25, 0, 0, 0, CLOSE], dtype=np.float32)}
    else:
        give_action = {cup_side: np.array(
            [0.4, 0, 0.25, 0, 0, 0, CLOSE], dtype=np.float32)}
    arx.step(give_action)
    if cup_side == "left":
        give2_action = {cup_side: np.array(
            [0.4, 0, 0.2, 0, 0, 0, CLOSE], dtype=np.float32)}
    else:
        give2_action = {cup_side: np.array(
            [0.4, 0, 0.2, 0, 0, 0, CLOSE], dtype=np.float32)}
    arx.step(give2_action)
    time.sleep(3.0)
    if cup_side == "left":
        open_action = {cup_side: np.array(
            [0.4, 0, 0.2, 0, 0, 0, OPEN], dtype=np.float32)}
    else:
        open_action = {cup_side: np.array(
            [0.4, 0, 0.2, 0, 0, 0, OPEN], dtype=np.float32)}
    arx.step(open_action)


def dual_cup_straw(
    arx: ARXRobotEnv,
    cup_side: str = "left",
    debug_pick_place: bool = True,
    depth_median_n: int = 10,
):
    try:
        # arx.step_lift(14.0)
        # single_arm_pick_place(
        #     arx,
        #     arm="left",
        #     pick_prompt="the red cup",
        #     place_prompt="",
        #     reset_robot=False,
        #     close_robot=False,
        #     debug=True,
        # )
        # step_base_duration(arx, vx=0.0, vy=0.0, vz=-0.5, duration=10.3)
        # step_base_duration(arx, vx=0.5, vy=0.0, vz=0.0, duration=10.0)
        # step_base_duration(arx, vx=0.0, vy=0.0, vz=0.5, duration=10.3)
        arx.step_lift(16.0)
        straw_side = "right" if cup_side == "left" else "left"
        pick_straw_prompt = f"the top of the nearest black straw in the cup"
        place_straw_prompt = f"the exact center of the cup's opening"
        single_arm_pick_place(
            arx,
            pick_prompt=pick_straw_prompt,
            place_prompt="",
            arm_side=straw_side,
            item_type="straw",
            debug=debug_pick_place,
            depth_median_n=depth_median_n,
        )

        # 右转90度
        step_base_duration(arx, vx=0.0, vy=0.0, vz=-0.5, duration=10.0)
        # 杯子放到摄像头中央
        if cup_side == "left":
            suit_action = {cup_side: np.array(
                [0.35, -0.125, -0.05, 0, 0, -1.571, CLOSE], dtype=np.float32)}
        else:
            suit_action = {cup_side: np.array(
                [0.35, 0.125, -0.05, 0, 0, 1.571, CLOSE], dtype=np.float32)}
        arx.step(suit_action)
        single_arm_pick_place(
            arx,
            pick_prompt="",
            place_prompt=place_straw_prompt,
            arm_side=straw_side,
            item_type="straw",
            debug=debug_pick_place,
            depth_median_n=depth_median_n,
        )

    except Exception as e:
        print(f"An error occurred: {e}")


def main():
    arx = ARXRobotEnv(duration_per_step=1.0/20.0,  # 就是插值里一步的时间，20Hz也就是0.05s
                      min_steps=20,
                      max_v_xyz=0.15, max_a_xyz=0.20,
                      max_v_rpy=0.5, max_a_rpy=1.00,
                      camera_type="all",
                      camera_view=("camera_h",),
                      img_size=(640, 480))
    try:
        single_arm_pick_place(
            arx,
            arm="left",
            pick_prompt="the green cup on the left coaster",
            place_prompt="",
            reset_robot=True,
            close_robot=False,
            debug=False,
            depth_median_n=10,
        )
        dual_cup_straw(arx, cup_side="left",
                       debug_pick_place=True, depth_median_n=5)
        special_give(arx, cup_side="left")
    finally:
        arx.close()


if __name__ == "__main__":
    main()
