from __future__ import annotations

import sys
import time
from pathlib import Path


ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.append(str(ROOT_DIR))

from ARX_Realenv.ROS2.arx_ros2_env import ARXRobotEnv  # noqa
from single_arm_pick_place import single_arm_pick_place  # noqa


def main():
    arx = ARXRobotEnv(
        duration_per_step=1.0 / 20.0,
        min_steps=20,
        max_v_xyz=0.15,
        max_a_xyz=0.20,
        max_v_rpy=0.45,
        max_a_rpy=1.00,
        camera_type="all",
        camera_view=("camera_l", "camera_h", "camera_r"),
        img_size=(640, 480),
    )

    try:
        arx.reset()
        arx.step_lift(20.0)

        pick_prompts = [
            "the blue square box",
            "the green square box",
            "the pink square box",
        ]

        for idx, pick_prompt in enumerate(pick_prompts, start=1):
            print(f"[{idx}/3] pick only: {pick_prompt}")
            _, _, arm_used = single_arm_pick_place(
                arx,
                pick_prompt=pick_prompt,
                place_prompt="",
                arm_side="right",
                item_type="deepbox",
                debug=True,
                depth_median_n=10,
                release_after_pick=True,
            )
            if arm_used is None:
                print(f"pick canceled, stop remaining tasks: {pick_prompt}")
                break

            print(f"set special mode after pick: home {arm_used}")
            success, error_message = arx.set_special_mode(1, side=arm_used)
            if not success and error_message:
                print(
                    f"set_special_mode(1, side={arm_used!r}) returned: {error_message}")

        print("set special mode: home")
        success, error_message = arx.set_special_mode(1)
        if not success and error_message:
            print(f"set_special_mode(1) returned: {error_message}")

        time.sleep(5.0)
    finally:
        arx.close()


if __name__ == "__main__":
    main()
