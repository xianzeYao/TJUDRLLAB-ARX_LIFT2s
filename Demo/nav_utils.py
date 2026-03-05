import numpy as np
import sys
import termios
import tty
from typing import List, Tuple
import math
import re

def refine_trajectory_strict(points, img_height=1000, img_width=1000):
    """
    1. 起点下移：确保起点在底部 20% 区域。
    2. 跨度约束：确保 Y 跨度在 40% 到 50% 之间。
    3. X 轴强制递减：从第一点开始持续向左偏。
    4. 末端平滑回钩：y(7) > y(6)。
    """
    pts = np.array(points, dtype=float)
    
    # --- 1. 整体偏移向下 (起点修正) ---
    y_threshold = img_height * 0.8
    if pts[0, 1] < y_threshold:
        offset_y = y_threshold - pts[0, 1]
        pts[:, 1] += offset_y

    # --- 2. Y 坐标跨度约束 (40% - 50%) ---
    y_min, y_max = np.min(pts[:, 1]), np.max(pts[:, 1])
    y_span = y_max - y_min
    
    max_allowed_span = img_height * 0.4
    min_allowed_span = img_height * 0.3
    
    # 情况 A: 跨度太大 -> 压缩
    if y_span > max_allowed_span:
        scale = max_allowed_span / y_span
        pts[:, 1] = y_max - (y_max - pts[:, 1]) * scale
    # 情况 B: 跨度太小 -> 拉伸
    elif y_span < min_allowed_span and y_span > 0:
        scale = min_allowed_span / y_span
        pts[:, 1] = y_max - (y_max - pts[:, 1]) * scale

    # --- 3. X 坐标强制递减逻辑 ---
    x_step_min = img_width * 0.02  # 每步最小左偏 2%
    for i in range(1, len(pts)):
        if pts[i, 0] > (pts[i-1, 0] - x_step_min):
            pts[i, 0] = pts[i-1, 0] - x_step_min

    # --- 4. 末端修正 (索引 5, 6, 7) ---
    # 加大末端左偏力度
    for i in range(5, 8):
        pts[i, 0] = np.minimum(pts[i, 0], pts[i-1, 0] - img_width * 0.04)

    # 修正 Y 轴：使变化平缓并形成回钩
    # 注意：这里使用 pts[5,1] + 小增量 确保 y 坐标变大（即向下移动）
    pts[6, 1] = pts[5, 1] + abs(pts[5, 1] - pts[4, 1]) * 0.1 
    pts[7, 1] = pts[6, 1] + 10 

    # 5. 边界保护
    pts[:, 0] = np.clip(pts[:, 0], 0, img_width - 1)
    pts[:, 1] = np.clip(pts[:, 1], 0, img_height - 1)
    
    return pts.tolist()

def extract_actions(instruction_str):
    # Step 1: Use regular expression to extract lines that represent actions
    action_lines = re.findall(r'\d+\.\s*(.*)', instruction_str)

    # Step 2: Return the list of extracted actions
    return action_lines

def index_resample(points, num_points=25, gamma=1.8):
    """
    仅按索引非均匀采样：
    - 前面点密
    - 后面点稀
    """
    n = len(points)
    if n <= num_points:
        return points

    u = np.linspace(0, 1, num_points)
    idx = (u ** gamma) * (n - 1)
    idx = np.round(idx).astype(int)

    # 防止重复索引（可选但推荐）
    idx[0] = 0
    idx[-1] = n - 1
    for i in range(1, len(idx)):
        if idx[i] <= idx[i - 1]:
            idx[i] = idx[i - 1] + 1
    idx = np.clip(idx, 0, n - 1)

    return [points[i] for i in idx]

def depth_to_meters(raw_depth: float) -> float:
    """兼容毫米与米的深度值。"""
    if not np.isfinite(raw_depth) or raw_depth < 0:
        raise ValueError(f"无效深度值: {raw_depth}")
    if raw_depth > 10.0:
        return float(raw_depth) / 1000.0
    return float(raw_depth)

def get_key():
    fd = sys.stdin.fileno()
    old_settings = termios.tcgetattr(fd)
    try:
        tty.setraw(fd)
        ch = sys.stdin.read(1)
    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)
    return ch

def path_to_actions(
    path: List[Tuple[int, int]],
    init_yaw: float = 0.0,
):
    actions = []
    cur_yaw = init_yaw

    for i in range(1, len(path)):
        x0, y0 = path[i - 1]
        x1, y1 = path[i]

        # dx = (x1 - x0) * cell_size
        # dy = (y1 - y0) * cell_size
        dx = x1 - x0
        dy = y1 - y0

        target_yaw = math.atan2(dy, dx)
        d_yaw = target_yaw - cur_yaw

        # 归一化到 [-pi, pi]
        while d_yaw > math.pi:
            d_yaw -= 2 * math.pi
        while d_yaw < -math.pi:
            d_yaw += 2 * math.pi

        dist = math.hypot(dx, dy)

        if abs(d_yaw) > 1e-3:
            actions.append(("rotate", -d_yaw))
            cur_yaw = target_yaw

        if dist > 1e-3:
            actions.append(("forward", dist))

    return actions

Action = Tuple[str, float]

def merge_forward_actions(actions: List[Action]) -> List[Action]:
    merged = []
    forward_acc = 0.0

    for act, val in actions:
        if act == "forward":
            forward_acc += val
        else:
            # 遇到非 forward，先把累计的 forward 放进去
            if forward_acc > 0:
                merged.append(("forward", forward_acc))
                forward_acc = 0.0
            merged.append((act, val))

    # 结尾如果是 forward
    if forward_acc > 0:
        merged.append(("forward", forward_acc))

    return merged