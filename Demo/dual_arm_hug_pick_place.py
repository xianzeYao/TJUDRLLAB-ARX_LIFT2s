from __future__ import annotations

import sys
import time
from typing import Literal, Optional

import cv2
import numpy as np

from arx_pointing import predict_multi_points_from_rgb, predict_point_from_rgb
from demo_utils import estimate_lift_from_goal_z, execute_pick_place_hug_sequence
from point2pos_utils import (
    get_aligned_frames,
    pixel_to_base_point_safe,
    pixel_to_ref_point_safe,
)
from visualize_utils import (
    VisualizeContext,
    dispatch_debug_image,
    render_multi_points_debug_view,
    should_stop,
)

sys.path.append("../ARX_Realenv/ROS2")  # noqa
from arx_ros2_env import ARXRobotEnv  # noqa


def _sort_left_right_points(
    points: list[tuple[float, float]],
) -> tuple[tuple[int, int], tuple[int, int]]:
    if len(points) < 2:
        raise RuntimeError(f"需要两个点，当前只得到 {len(points)} 个点")
    pts = sorted(points[:2], key=lambda p: p[0])
    left_px = (int(round(pts[0][0])), int(round(pts[0][1])))
    right_px = (int(round(pts[1][0])), int(round(pts[1][1])))
    return left_px, right_px


def _predict_hug_points(
    color: np.ndarray,
    depth: np.ndarray,
    object_prompt: str,
) -> tuple[tuple[int, int], tuple[int, int], np.ndarray, np.ndarray]:
    # 原提示（保留）：
    # "Detect one target object and return exactly two points in JSON list format: "
    # '[{"point_2d":[x,y]}, {"point_2d":[x,y]}]. '
    # f"Target object description: {object_prompt}. "
    # "Point 1 must be the midpoint of the left vertical front edge of the object. "
    # "Point 2 must be the midpoint of the right vertical front edge of the object. "
    # "Point 1 and Point 2 must be on different front vertical edges. "
    # "Return only JSON."
    prompt = (
        f"{object_prompt}. "
        "Return exactly two points in JSON: [{\"point_2d\":[x,y]}, {\"point_2d\":[x,y]}]. "
        "Point 1 belongs to the object's LEFT front vertical edge and must be the midpoint of that edge. "
        "Point 2 belongs to the object's RIGHT front vertical edge and must be the midpoint of that edge. "
        "Only JSON."
    )
    points = predict_multi_points_from_rgb(
        color,
        text_prompt="",
        all_prompt=prompt,
        assume_bgr=False,
        temperature=0.0,
    )
    left_px, right_px = _sort_left_right_points(points)
    left_ref = pixel_to_ref_point_safe(left_px, depth, robot_part="left")
    right_ref = pixel_to_ref_point_safe(right_px, depth, robot_part="right")
    if left_ref is None or right_ref is None:
        raise RuntimeError(
            f"像素转参考系失败: left_px={left_px}, right_px={right_px}"
        )
    pick_z = min(float(left_ref[2]), float(right_ref[2]))
    left_ref = np.asarray(left_ref, dtype=np.float32).copy()
    right_ref = np.asarray(right_ref, dtype=np.float32).copy()
    left_ref[2] = pick_z
    right_ref[2] = pick_z
    return left_px, right_px, left_ref, right_ref


def _predict_center_point(
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


def _predict_place_point(
    color: np.ndarray,
    place_prompt: str,
) -> tuple[int, int]:
    prompt = f"Point to the center of this place target: {place_prompt}. Return only JSON."
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
    do_pick: bool,
    do_place: bool,
) -> None:
    if item_type != "hug":
        raise ValueError(
            f"unknown item_type: {item_type!r}, dual_arm_hug_pick_place only supports 'hug'"
        )
    left_pose = None
    right_pose = None
    if do_place:
        try:
            status = arx.get_robot_status()
            left_status = status.get("left")
            right_status = status.get("right")
            left_end = np.asarray(left_status.end_pos, dtype=np.float32).reshape(-1)
            right_end = np.asarray(right_status.end_pos, dtype=np.float32).reshape(-1)
            left_joint = np.asarray(left_status.joint_pos, dtype=np.float32).reshape(-1)
            right_joint = np.asarray(right_status.joint_pos, dtype=np.float32).reshape(-1)
            left_pose = np.concatenate([left_end[:6], left_joint[6:7]], dtype=np.float32)
            right_pose = np.concatenate([right_end[:6], right_joint[6:7]], dtype=np.float32)
        except Exception as exc:
            raise RuntimeError("failed to read dual-arm current pose for place") from exc

    execute_pick_place_hug_sequence(
        arx=arx,
        left_ref=left_ref,
        right_ref=right_ref,
        do_pick=do_pick,
        do_place=do_place,
        left_pose=left_pose,
        right_pose=right_pose,
    )


def _adjust_lift_from_center_z(
    arx: ARXRobotEnv,
    center_goal_z: float,
    visualize: Optional[VisualizeContext] = None,
) -> None:
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


def _dual_arm_hug_pick_once(
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
            center_px = _predict_center_point(color=color, object_prompt=object_prompt)
            print(f"[hug-pick][center] uv={center_px}")
        except Exception as exc:
            continue

        center_pw = pixel_to_base_point_safe(center_px, depth, robot_part="center")
        if center_pw is None:
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
            continue

        # 阶段2：再打左右两侧点并继续原流程
        time.sleep(1.0)
        color, depth = get_aligned_frames(arx, depth_median_n=depth_median_n)
        if color is None or depth is None:
            continue

        try:
            left_px, right_px, left_ref, right_ref = _predict_hug_points(
                color=color,
                depth=depth,
                object_prompt=object_prompt,
            )
            print(f"[hug-pick][points] left_uv={left_px}, right_uv={right_px}")
        except Exception as exc:
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
                do_pick=True,
                do_place=False,
            )
        except Exception as exc:
            return left_ref, right_ref
        return left_ref, right_ref


def dual_arm_hug_pick(
    arx: ARXRobotEnv,
    object_prompt: str,
    item_type: Literal["hug"] = "hug",
    debug: bool = True,
    depth_median_n: int = 10,
    visualize: Optional[VisualizeContext] = None,
) -> tuple[Optional[np.ndarray], Optional[np.ndarray]]:
    try:
        left_ref, right_ref = _dual_arm_hug_pick_once(
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


def _dual_arm_hug_place_once(
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

        # 阶段1：第一次打放置中心点并调高度
        time.sleep(1.0)
        color, depth = get_aligned_frames(arx, depth_median_n=depth_median_n)
        if color is None or depth is None:
            continue

        try:
            place_px_1 = _predict_place_point(color=color, place_prompt=object_prompt)
            print(f"[hug-place][point-1] uv={place_px_1}")
        except Exception as exc:
            continue

        place_pw_1 = pixel_to_base_point_safe(place_px_1, depth, robot_part="center")
        if place_pw_1 is None:
            continue

        if debug:
            place_vis_1 = render_multi_points_debug_view(
                color=color,
                points=[place_px_1],
                title=f"hug-place-1: {object_prompt}",
            )
            place_debug_result_1 = dispatch_debug_image(
                visualize,
                source="dual_arm_hug_pick_place",
                panel="manip",
                image=place_vis_1,
                window_name="dual_arm_hug_place_point_1",
                object_prompt=object_prompt,
            )
            if not place_debug_result_1:
                continue
            if place_debug_result_1 is None:
                return None, None

        # try:
        #     _adjust_lift_from_center_z(
        #         arx=arx,
        #         center_goal_z=float(place_pw_1[2]),
        #         visualize=visualize,
        #     )
        # except Exception as exc:
        #     continue

        # 阶段2：再次打放置点，转换到左右臂坐标系并执行 place
        time.sleep(1.0)
        color, depth = get_aligned_frames(arx, depth_median_n=depth_median_n)
        if color is None or depth is None:
            continue

        try:
            place_px_2 = _predict_place_point(color=color, place_prompt=object_prompt)
            print(f"[hug-place][point-2] uv={place_px_2}")
        except Exception as exc:
            continue

        left_place_ref = pixel_to_ref_point_safe(place_px_2, depth, robot_part="left")
        right_place_ref = pixel_to_ref_point_safe(place_px_2, depth, robot_part="right")
        if left_place_ref is None or right_place_ref is None:
            continue

        if debug:
            place_vis_2 = render_multi_points_debug_view(
                color=color,
                points=[place_px_2],
                title=f"hug-place-2: {object_prompt}",
            )
            place_debug_result_2 = dispatch_debug_image(
                visualize,
                source="dual_arm_hug_pick_place",
                panel="manip",
                image=place_vis_2,
                window_name="dual_arm_hug_place_point_2",
                object_prompt=object_prompt,
            )
            if not place_debug_result_2:
                continue
            if place_debug_result_2 is None:
                return None, None

        try:
            _execute_item_sequence(
                arx=arx,
                item_type=item_type,
                left_ref=left_place_ref,
                right_ref=right_place_ref,
                do_pick=False,
                do_place=True,
            )
        except Exception as exc:
            return left_place_ref, right_place_ref
        return left_place_ref, right_place_ref


def dual_arm_hug_place(
    arx: ARXRobotEnv,
    object_prompt: str,
    item_type: Literal["hug"] = "hug",
    debug: bool = True,
    depth_median_n: int = 10,
    visualize: Optional[VisualizeContext] = None,
) -> tuple[Optional[np.ndarray], Optional[np.ndarray]]:
    try:
        left_ref, right_ref = _dual_arm_hug_place_once(
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
            arx.step_lift(16.0)
            time.sleep(1.0)
            pick_left_ref, pick_right_ref = dual_arm_hug_pick(
                arx=arx,
                object_prompt="a wood box",
                item_type="hug",
                debug=False,
                depth_median_n=10,
            )
            print(f"hug pick result left_ref={pick_left_ref}, right_ref={pick_right_ref}")
            cmd = input("Pick完成。输入 p 执行place，输入 q 退出并关闭：").strip().lower()
            if cmd == "q":
                break
            if cmd != "p":
                continue
            place_left_ref, place_right_ref = dual_arm_hug_place(
                arx=arx,
                object_prompt="a gray coaster",
                item_type="hug",
                debug=False,
                depth_median_n=10,
            )
            print(f"hug place result left_ref={place_left_ref}, right_ref={place_right_ref}")
            cmd = input("Place完成。输入 c 继续下一次，输入 q 退出并关闭：").strip().lower()
            if cmd != "c":
                break
    finally:
        arx.close()


if __name__ == "__main__":
    main()
