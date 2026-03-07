import math
import cv2
import time
from single_arm_pick_place import single_arm_pick_place
from autonav_robot import AutoNav_Robot
from nav_utils import path_to_actions, merge_forward_actions, index_resample
from dual_cup_pick_planning import dual_arm_pick_planning
from dual_cup_pick_planning_parallel import dual_arm_pick_planning_parallel
from dual_cup_straw import dual_cup_straw


def main():
    arx_nav_robot = AutoNav_Robot()
    try:
        #
        user_instruction = """
        Now I need you to navigate to the bubble tea preparation area.
        To bypass the current table, first turn left and then move forward-right; And Then you will find the red dot.
        Once the red dot is in sight, head directly toward it. Upon reaching the red dot, perform a turn so that the bubble tea preparation area (which will be on your right) is directly in front of you.
        """
        # dual arm pick planning and execute sequencly or parallelly
        # dual_arm_pick_planning(
        # arx_nav_robot.arx, goal="red cup", reset_robot=False, close_robot=False, no_last_place=True)
        dual_arm_pick_planning_parallel(
            arx_nav_robot.arx, goal="red cup", no_last_place=True, single_test=True)
        # nav go
        arx_nav_robot.arx.step_lift(15.0)
        # time.sleep(90.0)
        # arx_nav_robot.arx.step_base(1.0, 0.0, 0.0, 10.0)
        action_return = arx_nav_robot.go_follow_golden_path()
        # arx_nav_robot.arx.step_lift(18.0)
        # single_arm_pick_place(
        #     arx_nav_robot.arx,
        #     arm="right",
        #     pick_prompt="the red cup",
        #     place_prompt="",
        #     reset_robot=False,
        #     close_robot=False,
        #     debug=True,
        # )
        # place empty cup to making area and pick the bubble tea
        arx_nav_robot.arx.step_lift(18.0)
        time.sleep(3.0)
        single_arm_pick_place(
            arx_nav_robot.arx,
            arm="right",
            pick_prompt="",
            place_prompt="the center part of the brown coaster on the right side",
            debug=True,
        )
        time.sleep(3.0)
        # pick bubble tea cup
        single_arm_pick_place(
            arx_nav_robot.arx,
            arm="left",
            pick_prompt="the red cup on the left brown coaster",
            place_prompt="",
            debug=True,
        )
        # nav back
        arx_nav_robot.arx.step_lift(15.0)
        arx_nav_robot.return_follow_reversed_path()
        # arx_nav_robot.back_origin_path()
        # # arx_nav_robot.back_origin_path(action_return)
        # # place the bubble tea cup to the customer area and insert a straw
        dual_cup_straw(arx_nav_robot.arx, cup_side="left")
    finally:
        arx_nav_robot.arx.close()


if __name__ == "__main__":
    main()
