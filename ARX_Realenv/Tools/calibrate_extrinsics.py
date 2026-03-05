"""
基于 collect_calibration.py 采集的数据：仅做数据提取与角点识别，不做 hand-eye 求解与可视化。

使用前提：
- 样本是单张 RGB 图 + end_pos（相对于固定 ref 坐标系），棋盘规格由参数指定
- 相机内参通过 JSON/XML 提供

约定：
end_pos 使用“初始法兰固定帧 R0”作为参考系（固定坐标系）。
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, Iterable, Tuple

import cv2
import numpy as np


def rpy_to_matrix(roll: float, pitch: float, yaw: float) -> np.ndarray:
    """将 roll/pitch/yaw(ZYX) 转为旋转矩阵，也就是"""
    cx, sx = np.cos(roll), np.sin(roll)
    cy, sy = np.cos(pitch), np.sin(pitch)
    cz, sz = np.cos(yaw), np.sin(yaw)
    rx = np.array([[1, 0, 0], [0, cx, -sx], [0, sx, cx]])
    ry = np.array([[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]])
    rz = np.array([[cz, -sz, 0], [sz, cz, 0], [0, 0, 1]])
    return rz @ ry @ rx


DEFAULT_INTR_PATH = Path(__file__).resolve().parent / "right4camerah.json"


def load_intrinsics(path: Path | None) -> Tuple[np.ndarray, np.ndarray]:
    """从 JSON（camera_matrix, dist_coeffs）或 OpenCV XML 读取 K 与 D；默认 right4camerah.json。"""
    use_path = path if path is not None else DEFAULT_INTR_PATH
    if use_path.suffix.lower() == ".json":
        data = json.loads(use_path.read_text())
        K = np.asarray(data["camera_matrix"], dtype=np.float64)
        dist = np.asarray(data["dist_coeffs"], dtype=np.float64)
        # 兼容 dist 展平成一维
        if dist.ndim == 1:
            dist = dist.reshape(1, -1)
        return K, dist
    # 兼容 XML
    fs = cv2.FileStorage(str(use_path), cv2.FILE_STORAGE_READ)
    if not fs.isOpened():
        raise RuntimeError(f"无法打开内参文件: {use_path}")
    K = fs.getNode("camera_matrix").mat()
    dist = fs.getNode("dist_coeffs").mat()
    fs.release()
    if K is None or dist is None:
        raise RuntimeError(f"内参文件缺少 camera_matrix 或 dist_coeffs: {use_path}")
    return K, dist


def pick_image(meta: Dict, sample_dir: Path) -> Tuple[Path, np.ndarray] | None:
    """返回 (图像路径, image)。当前目录只存单张彩色图，若有 color 关键字则优先。"""
    files = sorted(sample_dir.glob("*.png"))
    if not files:
        return None
    color_files = [f for f in files if "color" in f.name]
    if color_files:
        files = color_files
    img_path = files[0]
    if img_path.suffix == ".npy":
        img = np.load(img_path)
    else:
        img = cv2.imread(str(img_path), cv2.IMREAD_COLOR)
    if img is None:
        return None
    return img_path, img


def detect_chessboard(
    img: np.ndarray,
    K: np.ndarray,
    dist: np.ndarray,
    board_cols: int,
    board_rows: int,
    square_size: float,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray] | None:
    """
    棋盘角点 -> PnP 估计 target(棋盘) -> cam 的位姿。
    board_cols/rows 为内角点数量（与 OpenCV 一致），square_size 为单格边长（米）。
    """
    # img 来自 cv2.imread，通道顺序为 BGR
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    ret, corners = cv2.findChessboardCorners(
        gray, (board_cols, board_rows), None)
    if not ret:
        return None
    corners_sub = cv2.cornerSubPix(
        gray,
        corners,
        (5, 5),
        (-1, -1),
        (cv2.TERM_CRITERIA_MAX_ITER | cv2.TERM_CRITERIA_EPS, 30, 0.001),
    )
    objp = np.zeros((board_cols * board_rows, 3), np.float32)
    objp[:, :2] = np.mgrid[0:board_cols,
                           0:board_rows].T.reshape(-1, 2) * square_size
    ok, rvec, tvec = cv2.solvePnP(
        objp, corners_sub, K, dist, flags=cv2.SOLVEPNP_ITERATIVE)
    if not ok:
        return None
    # print("rvec:", rvec)
    R, _ = cv2.Rodrigues(rvec)
    # print("R:", R)
    t = tvec.reshape(3)
    return R, t, corners_sub


def draw_axes_bgr(
    img: np.ndarray,
    T_cam_obj: np.ndarray,
    K: np.ndarray,
    scale: float = 0.05,
    thickness: int = 4,
    warn_prefix: str | None = None,
) -> np.ndarray:
    """
    在 BGR 图像上绘制物体坐标系 XYZ 轴（OpenCV 风格，X=红，Y=绿，Z=蓝）。
    T_cam_obj: 4x4，obj 坐标系在相机坐标系下的位姿。
    K: 3x3 内参。
    """
    axes = np.array(
        [
            [0.0, 0.0, 0.0, 1.0],
            [scale, 0.0, 0.0, 1.0],
            [0.0, scale, 0.0, 1.0],
            [0.0, 0.0, scale, 1.0],
        ],
        dtype=np.float64,
    )  # shape (4,4)
    pts_cam = (T_cam_obj @ axes.T).T  # (4,4)
    if pts_cam[0, 2] <= 1e-6:
        return img
    origin_px = K @ (pts_cam[0, :3] / pts_cam[0, 2])
    origin_px = (int(origin_px[0]), int(origin_px[1]))
    colors = [(0, 0, 255), (0, 255, 0), (255, 0, 0)]  # x,y,z
    skipped = []
    for p, color in zip(pts_cam[1:], colors):
        if p[2] <= 1e-6:
            skipped.append(color)
            continue
        uvw = K @ (p[:3] / p[2])
        pt = (int(uvw[0]), int(uvw[1]))
        cv2.arrowedLine(img, origin_px, pt, color, thickness, tipLength=0.25)
        cv2.circle(img, pt, 3,  color, -1)  # endpoint dot to确保颜色可见
    if skipped and warn_prefix:
        # 仅打印一次该样本的缺失轴颜色
        color_map = {(0, 0, 255): "X(red)", (0, 255, 0)                     : "Y(green)", (255, 0, 0): "Z(blue)"}
        print(
            f"{warn_prefix}: 轴未绘制 {', '.join(color_map[c] for c in skipped)} (Z<=0 或投影失败)")
    return img


def ref_gripper_transforms(end_pos: Iterable[float]) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    返回 (R_g2ref, t_g2ref, R_ref2g, t_ref2g)。
    end_pos 记录的是“gripper 在 ref 下的位姿”（这里 ref 为固定 R0），因此需要取逆得到 gripper->ref。
    """
    pose = np.asarray(end_pos, dtype=np.float64).flatten()
    R_ref2g = rpy_to_matrix(pose[3], pose[4], pose[5])
    t_ref2g = pose[:3]
    # return R_ref2g, t_ref2g
    R_g2ref = R_ref2g.T
    t_g2ref = -R_g2ref @ t_ref2g
    return R_g2ref, t_g2ref, R_ref2g, t_ref2g


def project_ref_point_to_image(
    img: np.ndarray,
    point_ref: np.ndarray,
    R_ref2cam: np.ndarray,
    t_ref2cam: np.ndarray,
    K: np.ndarray,
) -> tuple[np.ndarray, tuple[int, int] | None, bool]:
    """将 ref 坐标系下的 3D 点投影到图像；返回绘制后的图、像素坐标及是否发生过边框填充。"""
    point_cam = R_ref2cam @ point_ref.reshape(3, 1) + t_ref2cam.reshape(3, 1)
    if point_cam[2, 0] <= 1e-6:
        return img, None, False
    uvw = K @ (point_cam[:3] / point_cam[2, 0])
    u, v = int(uvw[0]), int(uvw[1])
    H, W = img.shape[:2]
    pad_top = max(0, -v)
    pad_left = max(0, -u)
    pad_bottom = max(0, v - (H - 1))
    pad_right = max(0, u - (W - 1))
    padded = bool(pad_top or pad_bottom or pad_left or pad_right)
    if pad_top or pad_bottom or pad_left or pad_right:
        img = cv2.copyMakeBorder(
            img,
            pad_top,
            pad_bottom,
            pad_left,
            pad_right,
            cv2.BORDER_CONSTANT,
            value=(0, 0, 0),
        )
        u += pad_left
        v += pad_top
    cv2.circle(img, (u, v), 3, (255, 0, 255), -1)  # ref 点标粉色
    return img, (u, v), padded


def main():
    parser = argparse.ArgumentParser(description="眼在手外数据解析（仅输出可知的 R/T）")
    parser.add_argument(
        "--data-dir", type=Path, default=Path("calibration_data"), help="采集数据目录"
    )
    parser.add_argument(
        "--intrinsics",
        type=Path,
        default=Path("instrinsics_camerah.json"),
        help="相机内参（json 或 xml），默认使用 right4camerah.json",
    )
    parser.add_argument(
        "--camera-label",
        type=str,
        default="cam_h",
        help="相机标签（如 h/l/r），用于输出文件名和键名",
    )
    parser.add_argument(
        "--board-cols",
        type=int,
        default=6,
        help="棋盘内角点列数（X 方向）",
    )
    parser.add_argument(
        "--board-rows",
        type=int,
        default=4,
        help="棋盘内角点行数（Y 方向）",
    )
    parser.add_argument(
        "--square-size",
        type=float,
        default=0.02,
        help="棋盘单格边长（米）",
    )
    parser.add_argument(
        "--side",
        choices=["left", "right"],
        default="right",
        help="使用哪个末端的姿态（与采集时一致），默认 right",
    )
    args = parser.parse_args()

    K, dist = load_intrinsics(args.intrinsics)

    sample_metas = sorted((args.data_dir).glob("sample_*/meta.json"))
    if not sample_metas:
        raise RuntimeError(f"未在 {args.data_dir} 找到 sample_*/meta.json")
    total_samples = len(sample_metas)
    detected_samples = 0
    detected_corners = 0

    records: list[Dict] = []
    R_t2c_all: list[np.ndarray] = []
    t_t2c_all: list[np.ndarray] = []
    R_g2r_all: list[np.ndarray] = []
    t_g2r_all: list[np.ndarray] = []
    R_r2g_all: list[np.ndarray] = []
    t_r2g_all: list[np.ndarray] = []
    vis_records: list[tuple[Path, np.ndarray]] = []

    # 遍历数据
    for meta_path in sample_metas:
        meta = json.loads(meta_path.read_text())
        sample_dir = meta_path.parent
        end_pos = meta.get("end_pos")
        if end_pos is None:
            print(f"{meta_path} 缺少 end_pos，跳过")
            continue
        img_pick = pick_image(meta, sample_dir)
        if img_pick is None:
            print(f"{meta_path} 无可用相机帧，跳过")
            continue
        img_path, img = img_pick
        cam_key = img_path.stem
        det = detect_chessboard(
            img,
            K,
            dist,
            board_cols=args.board_cols,
            board_rows=args.board_rows,
            square_size=args.square_size,
        )
        if det is None:
            print(f"{meta_path} [{cam_key}] 未检测到棋盘角点，跳过")
            continue

        R_t2c, t_t2c, corners_sub = det  # target->cam
        t_t2c = np.asarray(t_t2c, dtype=np.float64).reshape(3)
        # 保存调试图：角点与棋盘坐标系
        debug_dir = sample_dir / "debug"
        debug_dir.mkdir(parents=True, exist_ok=True)
        dbg = cv2.drawChessboardCorners(
            img.copy(), (args.board_cols, args.board_rows), corners_sub, True)
        cv2.imwrite(str(debug_dir / f"{cam_key}_chessboard.jpg"), dbg)
        T_cam_target = np.eye(4, dtype=np.float64)
        T_cam_target[:3, :3] = R_t2c
        T_cam_target[:3, 3] = t_t2c
        axes_vis = draw_axes_bgr(img.copy(), T_cam_target, K, scale=0.05)
        cv2.imwrite(str(debug_dir / f"{cam_key}_axes.jpg"), axes_vis)

        R_g2ref, t_g2ref, R_ref2g, t_ref2g = ref_gripper_transforms(end_pos)
        t_g2ref = np.asarray(t_g2ref, dtype=np.float64).reshape(3)

        records.append({
            "sample_dir": str(sample_dir),
            "image": str(img_path),
            "camera_key": cam_key,
            "R_target2cam": R_t2c.tolist(),
            "t_target2cam": t_t2c.tolist(),
            "R_gripper2ref": R_g2ref.tolist(),
            "t_gripper2ref": t_g2ref.tolist(),
            "R_ref2gripper": R_ref2g.tolist(),
            "t_ref2gripper": t_ref2g.tolist(),
            "corner_count": len(corners_sub) if corners_sub is not None else 0,
            "board_cols": args.board_cols,
            "board_rows": args.board_rows,
            "square_size": args.square_size,
        })
        R_t2c_all.append(R_t2c)
        t_t2c_all.append(t_t2c)
        R_g2r_all.append(R_g2ref)
        t_g2r_all.append(t_g2ref)
        R_r2g_all.append(R_ref2g)
        t_r2g_all.append(t_ref2g)
        vis_records.append((img_path, t_ref2g))
        detected_samples += 1
        detected_corners += len(corners_sub) if corners_sub is not None else 0

    if len(R_g2r_all) < 2:
        raise RuntimeError("有效样本不足（<2），无法估计相机外参")

    R_cam2ref, t_cam2ref = cv2.calibrateHandEye(
        R_g2r_all,
        t_g2r_all,
        R_t2c_all,
        t_t2c_all,
    )
    t_cam2ref = np.asarray(t_cam2ref, dtype=np.float64).reshape(3)
    R_cam2ref = np.asarray(R_cam2ref, dtype=np.float64)
    R_ref2cam = R_cam2ref.T
    t_ref2cam = -R_ref2cam @ t_cam2ref
    T_ref2cam = np.eye(4, dtype=np.float64)
    T_ref2cam[:3, :3] = R_ref2cam
    T_ref2cam[:3, 3] = t_ref2cam
    T_cam2ref = np.eye(4, dtype=np.float64)
    T_cam2ref[:3, :3] = R_cam2ref
    T_cam2ref[:3, 3] = t_cam2ref

    failed_samples = total_samples - detected_samples
    out_path = Path(f"extrinsics_{args.camera_label}_{args.side}.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "camera_label": args.camera_label,
        "side": args.side,
        "R_cam2ref": R_cam2ref.tolist(),
        "t_cam2ref": t_cam2ref.tolist(),
        "R_ref2cam": R_ref2cam.tolist(),
        "t_ref2cam": t_ref2cam.tolist(),
        "T_cam2ref": T_cam2ref.tolist(),
        "T_ref2cam": T_ref2cam.tolist(),
    }
    out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    print("标定完成：")
    print(
        f"样本总数: {total_samples}, 成功检测: {detected_samples}, 失败: {failed_samples}, 总角点数: {detected_corners}")
    print(f"结果已保存到 {out_path}")
    print("齐次矩阵字段已写入 JSON: T_ref2cam, T_cam2ref")
    # 可视化：将 ref->gripper 平移（末端原点）投影回对应 RGB 图
    for img_path, t_ref2g in vis_records:
        img = cv2.imread(str(img_path), cv2.IMREAD_COLOR)
        if img is None:
            continue
        img_vis, uv, padded = project_ref_point_to_image(
            img, t_ref2g, R_ref2cam, t_ref2cam, K
        )
        vis_dir = img_path.parent / "visual"
        vis_dir.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(vis_dir / f"{img_path.stem}_ref_origin.jpg"), img_vis)
        msg = f"{img_path}: ref->gripper 平移投影到 {uv}" if uv else f"{img_path}: ref->gripper 平移未能投影"
        if uv and padded:
            msg += "（已添加黑边）"
        print(msg)


if __name__ == "__main__":
    main()
