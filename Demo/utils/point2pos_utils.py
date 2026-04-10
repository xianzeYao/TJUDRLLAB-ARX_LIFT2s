"""
将 2D 像素点 + 对齐深度转换为基坐标系下的 6 维末端姿态。（部分做clip）

默认使用的相机的标定文件：
- 内参: instrics_right4camerah.json
- 外参:
  - left: new_extrinsics_cam_h_left.json
  - right: new_extrinsics_cam_h_right.json
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Literal, Tuple, List

import numpy as np

WORKSPACE = Path(__file__).resolve().parents[2]
DEFAULT_INTRINSICS = WORKSPACE / "ARX_Realenv/Tools/instrinsics_camerah.json"
DEFAULT_LEFT_EXTRINSICS = WORKSPACE / \
    "ARX_Realenv/Tools/extrinsics_cam_h_left.json"
DEFAULT_RIGHT_EXTRINSICS = WORKSPACE / \
    "ARX_Realenv/Tools/extrinsics_cam_h_right.json"

BIAS_REF2CAM = np.array([0.0, 0.24, 0.0, 0.0])


def depth_to_meters(raw_depth: float) -> float:
    """将深度值转换为米单位，兼容毫米与米输入。"""
    if not np.isfinite(raw_depth) or raw_depth <= 0:
        raise ValueError(f"无效深度值: {raw_depth}")
    if raw_depth > 10.0:
        return float(raw_depth) / 1000.0
    return float(raw_depth)


def load_intrinsics(path: Path | str | None = None) -> np.ndarray:
    """读取 3x3 相机内参矩阵。"""
    intr_path = Path(path) if path else DEFAULT_INTRINSICS
    data = json.loads(intr_path.read_text())
    K = np.asarray(data["camera_matrix"], dtype=np.float64)
    if K.shape != (3, 3):
        raise ValueError(f"内参矩阵形状异常: {K.shape}")
    return K


def _load_cam2ref_matrix(ext_path: Path) -> np.ndarray:
    payload = json.loads(ext_path.read_text())
    T = np.asarray(payload.get("T_cam2ref"), dtype=np.float64)
    if T.shape != (4, 4):
        raise ValueError(f"外参矩阵形状异常: {T.shape}")
    return T


def load_cam2ref(
    path: Path | str | None = None,
    side: Literal["left", "right"] | None = None,
) -> np.ndarray | tuple[np.ndarray, np.ndarray]:
    """
    读取 T_cam2ref。

    规则:
    - 传 `path`：返回该文件的单个 4x4 矩阵；
    - 不传 `path` 且传 `side`：返回对应 side 的默认 4x4 矩阵；
    - 默认（path/side 都不传）：返回 (T_left, T_right) 两个 4x4 矩阵。
    """
    if path is not None:
        return _load_cam2ref_matrix(Path(path))

    if side == "left":
        return _load_cam2ref_matrix(DEFAULT_LEFT_EXTRINSICS)
    if side == "right":
        return _load_cam2ref_matrix(DEFAULT_RIGHT_EXTRINSICS)
    if side is not None:
        raise ValueError(f"side must be 'left' or 'right', got: {side!r}")

    return (
        _load_cam2ref_matrix(DEFAULT_LEFT_EXTRINSICS),
        _load_cam2ref_matrix(DEFAULT_RIGHT_EXTRINSICS),
    )


def get_aligned_frames(
    arx,
    depth_median_n: int = 1,
) -> Tuple[np.ndarray | None, np.ndarray | None]:
    """读取 `camera_h` 的彩色图和对齐深度图，可选做多帧深度中值融合。"""
    frames = arx.get_camera(target_size=(640, 480), return_status=False)
    color = frames.get("camera_h_color")
    depth = frames.get("camera_h_aligned_depth_to_color")
    if color is None or depth is None or depth_median_n <= 1:
        return color, depth

    depths = [depth]
    for _ in range(depth_median_n - 1):
        frames = arx.get_camera(target_size=(640, 480), return_status=False)
        depth_i = frames.get("camera_h_aligned_depth_to_color")
        if depth_i is not None:
            depths.append(depth_i)

    depth_stack = np.stack(depths, axis=0).astype(np.float32, copy=False)
    valid_mask = np.isfinite(depth_stack) & (depth_stack > 0)
    valid_count = np.count_nonzero(valid_mask, axis=0)
    masked_depth = np.where(valid_mask, depth_stack, np.nan)
    median_depth = np.nanmedian(masked_depth, axis=0)
    median_depth = np.where(valid_count > 0, median_depth, 0.0)
    return color, median_depth


def _pixel_to_camera_point(
    pixel: Tuple[int, int],
    depth_image: np.ndarray,
    K: np.ndarray,
) -> np.ndarray:
    """把像素坐标和深度值投影到相机坐标系齐次点。"""
    u, v = int(round(pixel[0])), int(round(pixel[1]))
    H, W = depth_image.shape
    if not (0 <= u < W and 0 <= v < H):
        raise ValueError(f"像素越界: {(u, v)} not in [0,{W})x[0,{H})")

    z = depth_to_meters(float(depth_image[v, u]))
    fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]
    x_cam = (u - cx) * z / fx
    y_cam = (v - cy) * z / fy
    return np.array([x_cam, y_cam, z, 1.0], dtype=np.float64)


def _normalize_center_offset(
    offset: np.ndarray | Tuple[float, ...] | List[float] | None,
) -> np.ndarray:
    """把 center 模式下的 offset 统一整理成 4 维齐次偏移。"""
    if offset is None:
        return BIAS_REF2CAM.astype(np.float64, copy=True)

    offset_arr = np.asarray(offset, dtype=np.float64)
    if offset_arr.shape == (3,):
        return np.concatenate([offset_arr, [0.0]])
    if offset_arr.shape == (4,):
        return offset_arr
    raise ValueError(
        f"offset shape must be (3,) or (4,), got {offset_arr.shape}")


def pixel_to_ref_point(
    pixel: Tuple[int, int],
    depth_image: np.ndarray,
    robot_part: Literal["left", "right"] = "left",
    K: np.ndarray | None = None,
    T_left: np.ndarray | None = None,
    T_right: np.ndarray | None = None,
) -> np.ndarray:
    """像素 + 深度 -> 左/右参考坐标系 3D 点。"""
    if K is None:
        K = load_intrinsics()
    cam_point = _pixel_to_camera_point(pixel, depth_image, K)
    if robot_part == "left":
        T_cam2ref = T_left if T_left is not None else load_cam2ref(side="left")
    elif robot_part == "right":
        T_cam2ref = T_right if T_right is not None else load_cam2ref(
            side="right")
    else:
        raise ValueError(f"robot_part must be left/right, got {robot_part!r}")
    ref_point = T_cam2ref @ cam_point
    return ref_point[:3]


def pixel_to_ref_point_safe(
    pixel: Tuple[int, int],
    depth_image: np.ndarray,
    robot_part: Literal["left", "right"] = "left",
    K: np.ndarray | None = None,
    T_left: np.ndarray | None = None,
    T_right: np.ndarray | None = None,
) -> np.ndarray | None:
    """`pixel_to_ref_point()` 的安全版本，失败时返回 `None`。"""
    try:
        return pixel_to_ref_point(
            pixel,
            depth_image,
            robot_part=robot_part,
            K=K,
            T_left=T_left,
            T_right=T_right,
        )
    except ValueError:
        return None


def pixel_to_base_point(
    pixel: Tuple[int, int],
    depth_image: np.ndarray,
    robot_part: Literal["center", "left", "right"] = "center",
    offset: np.ndarray | Tuple[float, ...] | List[float] | None = None,
    K: np.ndarray | None = None,
    T_left: np.ndarray | None = None,
    T_right: np.ndarray | None = None,
) -> np.ndarray:
    """像素 + 深度 -> 机器人工作坐标系 3D 点。"""
    if K is None:
        K = load_intrinsics()
    cam_point = _pixel_to_camera_point(pixel, depth_image, K)

    if robot_part == "right":
        T_cam2ref = T_right if T_right is not None else load_cam2ref(
            side="right")
        ref_point = T_cam2ref @ cam_point
    else:
        T_cam2ref = T_left if T_left is not None else load_cam2ref(side="left")
        ref_point = T_cam2ref @ cam_point
        if robot_part == "center":
            ref_point = ref_point + _normalize_center_offset(offset)
        elif robot_part != "left":
            raise ValueError(
                f"robot_part must be center/left/right, got {robot_part!r}")

    return ref_point[:3]


def pixel_to_base_point_safe(
    pixel: Tuple[int, int],
    depth_image: np.ndarray,
    robot_part: Literal["center", "left", "right"] = "center",
    offset: np.ndarray | Tuple[float, ...] | List[float] | None = None,
    K: np.ndarray | None = None,
    T_left: np.ndarray | None = None,
    T_right: np.ndarray | None = None,
) -> np.ndarray | None:
    """`pixel_to_base_point()` 的安全版本，失败时返回 `None`。"""
    try:
        return pixel_to_base_point(
            pixel,
            depth_image,
            robot_part=robot_part,
            offset=offset,
            K=K,
            T_left=T_left,
            T_right=T_right,
        )
    except ValueError:
        return None


__all__ = [
    "get_aligned_frames",
    "load_intrinsics",
    "load_cam2ref",
    "depth_to_meters",
    "pixel_to_ref_point",
    "pixel_to_ref_point_safe",
    "pixel_to_base_point",
    "pixel_to_base_point_safe",
]
