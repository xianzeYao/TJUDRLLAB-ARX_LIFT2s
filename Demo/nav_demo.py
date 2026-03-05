from autonav_robot import AutoNav_Robot
from nav_utils import path_to_actions, merge_forward_actions, index_resample
import time
import cv2
import math

def main():
    arx_nav_robot = AutoNav_Robot(golden_point=True)
    try:
        user_instruction = """
        Now I need you to navigate to the bubble tea preparation area.
        To bypass the current table, first turn left and then move forward-right; And Then you will find the red dot.
        Once the red dot is in sight, head directly toward it. Upon reaching the red dot, perform a turn so that the bubble tea preparation area (which will be on your right) is directly in front of you.
        """
        # # go
        # # arx_nav_robot.run_for_1s(chx=1.0, duration=1.0)
        action_return, turn_right_action_log = arx_nav_robot.nav_plan(user_instruction)

        arx_nav_robot.arx.step_lift(15.0)

        # # back
        arx_nav_robot.(action_return, turn_right_action_log)

        # -- put cup start --

        # -- put cup end --

    finally:
        arx_nav_robot.arx.close()

if __name__ == "__main__":
    main()