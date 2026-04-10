from __future__ import annotations

import sys
import time
from pathlib import Path
from typing import Literal, Optional

import cv2
import numpy as np

from utils import (
    VisualizeContext,
    dispatch_debug_image,
    estimate_lift_from_goal_z,
    execute_pick_place_hug_sequence,
    get_aligned_frames,
    pixel_to_base_point_safe,
    pixel_to_ref_point_safe,
    predict_multi_points_from_rgb,
    predict_point_from_rgb,
    render_multi_points_debug_view,
    should_stop,
)

CURRENT_DIR = Path(__file__).resolve().parent
ROOT_DIR = CURRENT_DIR.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.append(str(ROOT_DIR))
from ARX_Realenv.ROS2.arx_ros2_env import ARXRobotEnv  # noqa


def _sort_left_right_points(
    points: list[tuple[float, float]],
) -> tuple[tuple[int, int], tuple[int, int]]:
    if len(points) < 2:
        raise RuntimeError(f"需要两个点，当前只得到 {len(points)} 个点")
    pts = sorted(points[:2], key=lambda p: p[0])
    left_px = (int(round(pts[0][0])), int(round(pts[0][1])))
    right_px = (int(round(pts[1][0])), int(round(pts[1][1])))
    return left_px, right_px


def predict_hug_points(
    color: np.ndarray,
    object_prompt: str,
) -> tuple[tuple[int, int], tuple[int, int]]:
    prompt = (
        "Detect one target object and return exactly two points in JSON list format: "
        '[{"point_2d":[x,y]}, {"point_2d":[x,y]}]. '
        f"Target object description: {object_prompt}. "
        "Point 1 must be the center of the left side of the object. "
        "Point 2 must be the center of the right side of the object. "
        "Point 1 and Point 2 must be at different side. "
        "Return only JSON."
    )
    points = predict_multi_points_from_rgb(
        color,
        text_prompt="",
        all_prompt=prompt,
        assume_bgr=False,
        temperature=0.0,
    )
    return _sort_left_right_points(points)


def predict_center_point(
    color: np.ndarray,
    object_prompt: str,
) -> tuple[int, int]:
    prompt = f"Point to the center of this object: {object_prompt}. Return only JSON."
    u, v = predict_point_from_rgb(
        color,
        text_prompt="",
        all_prompt=prompt,
        assume_bgr=False,
        temperature=0.0,
    )
    return int(round(u)), int(round(v))


def _execute_item_sequence(
    arx: ARXRobotEnv,
    item_type: Literal["hug"],
    left_ref: np.ndarray,
    right_ref: np.ndarray,
) -> None:
    if item_type != "hug":
        raise ValueError(
            f"unknown item_type: {item_type!r}, dual_arm_hug_pick_place only supports 'hug'"
        )
    execute_pick_place_hug_sequence(
        arx=arx,
        left_ref=left_ref,
        right_ref=right_ref,
        do_pick=True,
        do_place=False,
    )


def _adjust_lift_from_center_z(
    arx: ARXRobotEnv,
    center_goal_z: float,
    visualize: Optional[VisualizeContext] = None,
) -> None:
    del visualize
    base_status = arx.get_robot_status().get("base")
    if base_status is None or not hasattr(base_status, "height"):
        raise RuntimeError("base status unavailable, cannot adjust lift")
    current_lift = float(base_status.height)
    target_lift = estimate_lift_from_goal_z(
        goal_z=float(center_goal_z),
        current_lift=current_lift,
        target_goal_z=0.05,
    )
    arx.step_lift(target_lift)
    time.sleep(1.0)


def dual_arm_hug_pick_place_once(
    arx: ARXRobotEnv,
    object_prompt: str,
    item_type: Literal["hug"] = "hug",
    debug: bool = True,
    depth_median_n: int = 10,
    visualize: Optional[VisualizeContext] = None,
) -> tuple[Optional[np.ndarray], Optional[np.ndarray]]:
    while True:
        if should_stop(visualize):
            return None, None

        # 阶段1：先打中间点并据此调高度
        time.sleep(1.0)
        color, depth = get_aligned_frames(arx, depth_median_n=depth_median_n)
        if color is None or depth is None:
            continue

        try:
            center_px = predict_center_point(color=color, object_prompt=object_prompt)
        except Exception as exc:
            print(f"中心点检测失败，重试中: {exc}")
            continue

        center_pw = pixel_to_base_point_safe(center_px, depth, robot_part="center")
        if center_pw is None:
            print(f"中心点像素转坐标失败，重试中: center_px={center_px}")
            continue

        if debug:
            center_vis = render_multi_points_debug_view(
                color=color,
                points=[center_px],
                title=f"hug-center: {object_prompt}",
            )
            center_debug_result = dispatch_debug_image(
                visualize,
                source="dual_arm_hug_pick_place",
                panel="manip",
                image=center_vis,
                window_name="dual_arm_hug_pick_place_center",
                object_prompt=object_prompt,
            )
            if not center_debug_result:
                continue
            if center_debug_result is None:
                return None, None

        try:
            _adjust_lift_from_center_z(
                arx=arx,
                center_goal_z=float(center_pw[2]),
                visualize=visualize,
            )
        except Exception as exc:
            print(f"根据中心点调高度失败，重试中: {exc}")
            continue

        # 阶段2：再打左右两侧点并继续原流程
        time.sleep(1.0)
        color, depth = get_aligned_frames(arx, depth_median_n=depth_median_n)
        if color is None or depth is None:
            continue

        try:
            left_px, right_px = predict_hug_points(color=color, object_prompt=object_prompt)
        except Exception as exc:
            print(f"两点检测失败，重试中: {exc}")
            continue

        left_ref = pixel_to_ref_point_safe(left_px, depth, robot_part="left")
        right_ref = pixel_to_ref_point_safe(right_px, depth, robot_part="right")
        if left_ref is None or right_ref is None:
            print(
                "像素转参考系失败，重试中: "
                f"left_px={left_px}, right_px={right_px}"
            )
            continue

        if debug:
            vis = render_multi_points_debug_view(
                color=color,
                points=[left_px, right_px],
                title=f"hug: {object_prompt}",
            )
            debug_result = dispatch_debug_image(
                visualize,
                source="dual_arm_hug_pick_place",
                panel="manip",
                image=vis,
                window_name="dual_arm_hug_pick_place",
                arm="dual",
                object_prompt=object_prompt,
            )
            if not debug_result:
                continue
            if debug_result is None:
                return None, None
        else:
            cv2.destroyAllWindows()

        try:
            _execute_item_sequence(
                arx=arx,
                item_type=item_type,
                left_ref=left_ref,
                right_ref=right_ref,
            )
        except Exception as exc:
            print(f"hug 动作执行失败: {exc}")
            return left_ref, right_ref
        return left_ref, right_ref


def dual_arm_hug_pick_place(
    arx: ARXRobotEnv,
    object_prompt: str,
    item_type: Literal["hug"] = "hug",
    debug: bool = True,
    depth_median_n: int = 10,
    visualize: Optional[VisualizeContext] = None,
) -> tuple[Optional[np.ndarray], Optional[np.ndarray]]:
    try:
        left_ref, right_ref = dual_arm_hug_pick_place_once(
            arx=arx,
            object_prompt=object_prompt,
            item_type=item_type,
            debug=debug,
            depth_median_n=depth_median_n,
            visualize=visualize,
        )
        return left_ref, right_ref
    finally:
        cv2.destroyAllWindows()


def main() -> None:
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
    arx.reset()
    # a plastic basket
    try:
        while True:
            arx.step_lift(10.0)
            time.sleep(1.0)
            left_ref, right_ref = dual_arm_hug_pick_place(
                arx=arx,
                object_prompt="a wood box",
                item_type="hug",
                debug=True,
                depth_median_n=10,
            )
            print(f"hug result left_ref={left_ref}, right_ref={right_ref}")
            cmd = input("任务执行完成。输入 c 继续下一次，输入 q 退出并关闭：").strip().lower()
            if cmd != "c":
                break
    finally:
        arx.close()


if __name__ == "__main__":
    main()
