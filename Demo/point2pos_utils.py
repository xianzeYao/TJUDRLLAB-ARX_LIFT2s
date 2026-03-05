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

WORKSPACE = Path(__file__).resolve().parent.parent
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


def pixel_to_ref_point(
    pixel: Tuple[int, int],
    depth_image: np.ndarray,
    K: np.ndarray,
    T_cam2ref: np.ndarray,
) -> np.ndarray:
    """像素 + 深度 -> 基坐标系 3D 点。"""
    u, v = pixel
    H, W = depth_image.shape
    if not (0 <= u < W and 0 <= v < H):
        raise ValueError(f"像素越界: {(u, v)} not in [0,{W})x[0,{H})")
    z = depth_to_meters(float(depth_image[v, u]))
    fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]
    x_cam = (u - cx) * z / fx
    y_cam = (v - cy) * z / fy
    cam_point = np.array([x_cam, y_cam, z, 1.0], dtype=np.float64)
    ref_point = T_cam2ref @ cam_point
    return ref_point[:3]


def pixel_to_ref_point_safe(
    pixel: Tuple[int, int],
    depth_image: np.ndarray,
    K: np.ndarray,
    T_cam2ref: np.ndarray,
) -> np.ndarray | None:
    """像素 + 深度 -> 基坐标系 3D 点。深度无效则返回 None。"""
    try:
        return pixel_to_ref_point(pixel, depth_image, K, T_cam2ref)
    except ValueError:
        return None


def filter_valid_points(
    uv_list: List[Tuple[int, int]],
    depth: np.ndarray,
    K: np.ndarray,
    T_cam2ref: np.ndarray,
) -> Tuple[List[Tuple[int, int]], List[np.ndarray]]:
    """过滤深度无效的像素点并转换为 ref 坐标系 3D 点。"""
    valid_uvs: List[Tuple[int, int]] = []
    pt_refs: List[np.ndarray] = []
    for uv in uv_list:
        raw_depth = depth[uv[1], uv[0]]
        if np.isnan(raw_depth) or raw_depth == 0:
            print(f"预测像素 {uv} 深度无效({raw_depth})，跳过该点")
            continue
        ref = pixel_to_ref_point_safe(uv, depth, K, T_cam2ref)
        if ref is None:
            continue
        pt_refs.append(ref)
        valid_uvs.append(uv)
    return valid_uvs, pt_refs


def pixel_to_base_point(
    pixel: Tuple[int, int],
    depth_image: np.ndarray,
    K: np.ndarray,
    T_cam2ref: np.ndarray,
) -> np.ndarray:
    """像素 + 深度 -> 基坐标系 3D 点。"""
    u, v = pixel
    H, W = depth_image.shape
    if not (0 <= u < W and 0 <= v < H):
        raise ValueError(f"像素越界: {(u, v)} not in [0,{W})x[0,{H})")
    z = depth_to_meters(float(depth_image[v, u]))
    fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]
    x_cam = (u - cx) * z / fx
    y_cam = (v - cy) * z / fy
    cam_point = np.array([x_cam, y_cam, z, 1.0], dtype=np.float64)
    base_point = T_cam2ref @ cam_point + BIAS_REF2CAM
    return base_point[:2]


__all__ = [
    "load_intrinsics",
    "load_cam2ref",
    "depth_to_meters",
    "pixel_to_ref_point",
    "pixel_to_ref_point_safe",
    "filter_valid_points",
    "pixel_to_base_point",
]
