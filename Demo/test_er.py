import cv2
import numpy as np

from arx_pointing import predict_multi_points_from_rgb, predict_point_from_rgb
import sys
## """Curren Goal is: pick the red cup. I need to pick up the cups from top to the red cup. What is the picking plan steps to finish the goal?""",
## """Curren Goal is: pick the purple cup. I need to pick up the cups from top to the purple cup. What is the picking plan steps to finish the goal?""",
# """You are currently a robot performing robotic manipulation tasks. 
#         The task instruction is: place the purple cup on the coaster with the label of number 3. 
#         Use 2D points to mark the manipulated object-centric waypoints to guide the robot to successfully complete the task. 
#         You must provide the points in the order of the trajectory, and the number of points must be 6."""
# """Point out the coaster with the label of number 3 and the Purple cup"""
def main():
    color = cv2.imread("../Testdata4Mani/point.png")
    points, message = predict_multi_points_from_rgb(
        image=color,
        text_prompt="",
        all_prompt="""Curren Goal is: pick the red cup. I need to pick up the cups from top to the red cup. What is the picking plan steps to finish the goal?""",
        assume_bgr=False,
        return_raw=True
    )
    i = 1
    for (u, v) in points:
        cv2.circle(
            color,
            center=(int(u), int(v)),
            radius=5,
            color=(0, 0, 255),
            thickness=-1  # -1 表示实心圆
        )
        cv2.putText(
            color,
            text=f"{i}",
            org=(int(u)-3, int(v)+1),
            fontFace=cv2.FONT_HERSHEY_SIMPLEX,
            fontScale=0.3,
            color=(0, 255, 0),
            thickness=1
        )
        i += 1

    print(f"Predicted Points: {points}")
    print(f"Generated message: {message}")
    cv2.imshow("Predicted Points", color)
    cv2.imwrite("../Testdata4Mani/cup_bottom_out.png", color)


if __name__ == "__main__":
    main()
