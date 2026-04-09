"""
货架双物体抓取：检测两个物体，按距离排序，依次导航并抓取。
"""
from __future__ import annotations

import math
import sys
import time
from typing import Literal, Optional

import numpy as np

sys.path.append("../ARX_Realenv/ROS2")  # noqa

from arx_pointing import predict_multi_points_from_rgb
from arx_ros2_env import ARXRobotEnv
from demo_utils import estimate_lift_from_goal_z, step_base_duration
from nav_goal import (
    _vote_goal_presence,
    nav_to_goal,
)
from point2pos_utils import get_aligned_frames, pixel_to_base_point_safe
from single_arm_pick_place import single_arm_pick_place

# 与 nav_goal._select_goal_point(..., offset=...) 一致：像素→基座时 camera bias 的 y 分量（米）。
# 阶段1 与 nav_to_goal 必须用同一值，否则「排序用的 xyz」与「导航用的 xyz」不是同一套几何。
SHELF_PIXEL_OFFSET_Y = 0.24


def _predict_two_points(
    color: np.ndarray,
    obj1_desc: str,
    obj2_desc: str,
) -> tuple[tuple[float, float], tuple[float, float]]:
    """VLM 预测两个物体的像素坐标，返回 (px1, px2)。"""
    full_prompt = (
        "Provide exactly two points coordinate of objects region this sentence describes: "
        f"{obj1_desc} and {obj2_desc}. "
        'The answer should be presented in JSON format as follows: [{"point_2d": [x, y]}]. '
        "Return only JSON. First point is object 1, second point is object 2."
    )
    points = predict_multi_points_from_rgb(
        color,
        text_prompt="",
        all_prompt=full_prompt,
        assume_bgr=False,
        temperature=0.0,
    )
    if len(points) < 2:
        raise RuntimeError(f"未解析到足够坐标: {len(points)}")
    px1 = (float(points[0][0]), float(points[0][1]))
    px2 = (float(points[1][0]), float(points[1][1]))
    return px1, px2


def shelf_dual_pick(
    arx: ARXRobotEnv,
    obj1_desc: str,
    obj2_desc: str,
    item_type: Literal["cup", "straw", "deepbox"] = "cup",
    target_goal_z: float = 0.35,
    distance: float = 0.55,
    lateral_vy_cmd: float = 0.75,
    lateral_speed: float = 0.125,
    depth_median_n: int = 10,
    vote_times: int = 5,
    debug: bool = True,
) -> Optional[list[tuple[str, np.ndarray, tuple[float, float]]]]:
    """货架双物体抓取主流程：检测、排序、依次导航并抓取。

    逻辑分块（为何升降要「本文件自己算」而不交给 nav_to_goal 里的 use_goal_z_for_lift）：

    1) **阶段1（远景）** 用双物体专用 prompt 得到两个像素点，经深度得到基座系 ``pw``，
       用 ``math.hypot(pw[0], pw[1])`` 排序，得到先抓谁、后抓谁。这里的 ``pw[2]`` 是后续
       **升降的锚定 z**，必须与 ``pixel_to_base_point_safe`` 的 offset 和 ``nav_to_goal``
       的 ``offset`` 一致（见 ``SHELF_PIXEL_OFFSET_Y``）。

    2) **nav_to_goal** 内部会再用「单目标 + 实时点」算 ``goal_pw``，其 ``goal_pw[2]`` 与阶段1
       的 ``pw[2]`` 往往不一致（提示词不同、多候选点取最近、深度噪声）。若打开
       ``use_goal_z_for_lift=True``，导航结束时会用 **实时** ``goal_pw[2]`` 覆盖升降，
       就会出现日志里「打印的目标升降」和「实际 lift to 0」矛盾。因此本脚本对第一次导航
       传入 ``use_goal_z_for_lift=False``，导航只负责 **平面接近**；导航结束后用阶段1 保存的
       ``ordered[0][1][2]`` 再调用 ``estimate_lift_from_goal_z`` + ``step_lift``，与打印一致。

    3) **第一次抓取**：左臂 ``single_arm_pick_place``。

    4) **第二次**：先 ``step_lift`` 便于抬头看货架，再按两物体远景 ``delta_y`` 侧向平移，
       再用 ``ordered[1][1][2]`` 算升降并 ``step_lift``，右臂抓取。

    Args:
        arx: 已初始化的 ARXRobotEnv 实例，需已完成 reset()。
        obj1_desc: 第一个物体的视觉描述。
        obj2_desc: 第二个物体的视觉描述。
        item_type: "cup"、"straw" 或 "deepbox"。
        target_goal_z: 目标物体 z 的参考值，用于 estimate_lift_from_goal_z 计算升降高度。
        distance: 导航停止距离。
        lateral_vy_cmd: 左右平移时的速度指令值。
        lateral_speed: 左右平移时的期望速度
        depth_median_n: 深度中值滤波帧数。
        vote_times: 存在性检测投票次数。
        debug: 是否显示调试窗口。

    Returns:
        成功时返回 ordered 列表；用户按 q 取消时返回 None。
    """
    if not obj1_desc or not obj2_desc:
        raise ValueError("obj1_desc 和 obj2_desc 不能为空")

    # 阶段1：双物体检测与打点、按距离排序
    ordered: list[tuple[str, np.ndarray, tuple[float, float]]] = []
    while True:

        time.sleep(1.5)
        color, depth = get_aligned_frames(
            arx, depth_median_n=depth_median_n
        )
        if color is None or depth is None:
            continue
        # 修改逻辑
        if not _vote_goal_presence(color, obj1_desc, vote_times=vote_times):
            print(f"未检测到物体: {obj1_desc}，自动刷新")
            continue
        print(f"检测到物体: {obj1_desc}")
        if not _vote_goal_presence(color, obj2_desc, vote_times=vote_times):
            print(f"未检测到物体: {obj2_desc}，自动刷新")
            continue
        print(f"检测到物体: {obj2_desc}")
        try:
            px1, px2 = _predict_two_points(color, obj1_desc, obj2_desc)
        except RuntimeError as exc:
            print(f"点位预测失败，自动刷新：{exc}")
            continue

        pw1 = pixel_to_base_point_safe(
            px1,
            depth,
            robot_part="center",
            offset=[0.0, SHELF_PIXEL_OFFSET_Y, 0.0],
        )
        pw2 = pixel_to_base_point_safe(
            px2,
            depth,
            robot_part="center",
            offset=[0.0, SHELF_PIXEL_OFFSET_Y, 0.0],
        )
        if pw1 is None:
            print(f"预测像素 {px1} 深度无效或像素越界，自动刷新")
            continue
        if pw2 is None:
            print(f"预测像素 {px2} 深度无效或像素越界，自动刷新")
            continue

        # 显示基座坐标系下的 xyz
        print(f"[物体1] {obj1_desc} -> xyz = ({pw1[0]:.4f}, {pw1[1]:.4f}, {pw1[2]:.4f})")
        print(f"[物体2] {obj2_desc} -> xyz = ({pw2[0]:.4f}, {pw2[1]:.4f}, {pw2[2]:.4f})")

        dist1 = math.hypot(pw1[0], pw1[1])
        dist2 = math.hypot(pw2[0], pw2[1])
        if dist1 <= dist2:
            ordered = [
                (obj1_desc, pw1, px1),
                (obj2_desc, pw2, px2),
            ]
            print(f"按距离排序: 先抓 [{obj1_desc}]（距 {dist1:.4f}m），后抓 [{obj2_desc}]（距 {dist2:.4f}m）")
        else:
            ordered = [
                (obj2_desc, pw2, px2),
                (obj1_desc, pw1, px1),
            ]
            print(f"按距离排序: 先抓 [{obj2_desc}]（距 {dist2:.4f}m），后抓 [{obj1_desc}]（距 {dist1:.4f}m）")
        break
    # 首先调整高度并导航
    base_status = arx.get_robot_status().get("base")
    current_lift = float(base_status.height)
    obj1_lift = estimate_lift_from_goal_z(
        goal_z=float(ordered[0][1][2]),
        current_lift=current_lift,
        target_goal_z=target_goal_z,
    )
    print(
        f"物体 z={ordered[0][1][2]:.4f}, 当前升降={current_lift:.2f}, 目标升降={obj1_lift:.2f}"
    )
    # 导航阶段：不启用 nav 内按 live goal_pw[2] 调升降（与阶段1 锚定 z 易冲突），只接近目标。
    nav_result = nav_to_goal(
        arx,
        goal=ordered[0][0],
        distance=distance,
        lift_height=17.0,
        offset=SHELF_PIXEL_OFFSET_Y,
        use_goal_z_for_lift=True,
        target_goal_z=target_goal_z,
        rotate_recover=True,
        continuous=False,
        debug_raw=False,
        depth_median_n=depth_median_n,
        vote_times=vote_times,
    )
    if nav_result is None:
        return None
    # 抓取
    order_str = "第1个"
    arm_str = "右臂"
    arm_side = "right"
    print(f"--- 正在抓取{order_str}: [{ordered[0][0]}]（{arm_str}）---")
    _pick_ref, _place_ref, arm_used = single_arm_pick_place(
            arx,
            pick_prompt=ordered[0][0],
            place_prompt="",
            arm_side=arm_side,
            item_type=item_type,
            debug=debug,
            depth_median_n=depth_median_n,
            release_after_pick=False,
        )
    if arm_used is None:
        return None
    # 只平移不导航,先抬升让摄像头看到，然后平移，最后调整高度
    arx.step_lift(13.0)
    time.sleep(1.0)
    delta_y = float(ordered[1][1][1] - ordered[0][1][1])
    dur = abs(delta_y) / lateral_speed
    vy = -lateral_vy_cmd if delta_y > 0 else lateral_vy_cmd
    print(f"左右平移: delta_y={delta_y:.4f}m, 持续{dur:.2f}s")
    step_base_duration(
        arx, vx=0.0, vy=vy, vz=0.0, duration=dur
    )
    base_status = arx.get_robot_status().get("base")
    current_lift = float(base_status.height)
    obj2_lift = estimate_lift_from_goal_z(
        goal_z=float(ordered[1][1][2]),
        current_lift=current_lift,
        target_goal_z=target_goal_z,
    )
    print(
        f"物体 z={ordered[1][1][2]:.4f}, 当前升降={current_lift:.2f}, 目标升降={obj2_lift:.2f}"
    )
    arx.step_lift(obj2_lift)
    time.sleep(1.0)
    # 抓取
    order_str = "第2个"
    arm_str = "左臂"
    arm_side = "left"
    print(f"--- 正在抓取{order_str}: [{ordered[1][0]}]（{arm_str}）---")
    _pick_ref, _place_ref, arm_used = single_arm_pick_place(
            arx,
            pick_prompt=ordered[1][0],
            place_prompt="",
            arm_side=arm_side,
            item_type=item_type,
            debug=debug,
            depth_median_n=depth_median_n,
            release_after_pick=True,
        )
    if arm_used is None:
        return None
    return ordered


def main() -> None:
    arx = ARXRobotEnv(
        duration_per_step=1.0 / 20.0,
        min_steps=20,
        max_v_xyz=0.15,
        max_a_xyz=0.20,
        max_v_rpy=0.45,
        max_a_rpy=1.00,
        camera_type="all",
        camera_view=("camera_h",),
        img_size=(640, 480),
    )
    try:
        arx.reset()
        arx.step_lift(17.0)
        time.sleep(1.0)
        '''
        shelf project pick_prompt:
        1. a tennis ball
        2. a yellow glue
        3. a sponge
        4. a white cup
        5. a brown hourse
        6. a pink soda can
        7. a rubik's cube
        8. a pink box
        9. a red hammer
        10. an apple
        11. tissues
        12. a green jar
        13. the middle-topper metal component
        '''
        shelf_dual_pick(
            arx,
            obj1_desc="a pink soda can",
            obj2_desc="a red Potato chip canister",
            item_type="cup",
            target_goal_z=0,
            distance=0.53,
            lateral_vy_cmd=0.75,
            lateral_speed=0.125,
            depth_median_n=5,
            vote_times=5,
            debug=False,
        )
    finally:
        arx.close()


if __name__ == "__main__":
    main()
