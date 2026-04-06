from __future__ import annotations

import sys
import time
from pathlib import Path
from typing import Literal, Optional

import cv2
import numpy as np

CURRENT_DIR = Path(__file__).resolve().parent
ROOT_DIR = CURRENT_DIR.parent
ROS2_DIR = ROOT_DIR / "ARX_Realenv" / "ROS2"

if str(CURRENT_DIR) not in sys.path:
    sys.path.append(str(CURRENT_DIR))
if str(ROS2_DIR) not in sys.path:
    sys.path.append(str(ROS2_DIR))

from arx_ros2_env import ARXRobotEnv  # noqa: E402
from point2pos_utils import depth_to_meters, load_cam2ref, load_intrinsics  # noqa: E402


COLOR_PRESETS = {
    "red1": ((0, 120, 70), (10, 255, 255)),
    "red2": ((170, 120, 70), (180, 255, 255)),
    "green": ((35, 70, 70), (90, 255, 255)),
    "blue": ((90, 70, 70), (130, 255, 255)),
    "yellow": ((18, 90, 90), (40, 255, 255)),
}

if hasattr(cv2, "aruco"):
    ARUCO_DICT_NAMES = {
        "4x4_50": cv2.aruco.DICT_4X4_50,
        "4x4_100": cv2.aruco.DICT_4X4_100,
        "5x5_50": cv2.aruco.DICT_5X5_50,
        "5x5_100": cv2.aruco.DICT_5X5_100,
        "6x6_50": cv2.aruco.DICT_6X6_50,
        "6x6_100": cv2.aruco.DICT_6X6_100,
    }
else:
    ARUCO_DICT_NAMES = {}

COLOR_MIN_AREA = 500.0


def _default_grasp_offset(arm: str) -> np.ndarray:
    return np.array([0.05, 0.0, 0.0], dtype=np.float32)


def _draw_text_lines(
    image: np.ndarray,
    lines: list[str],
    origin: tuple[int, int] = (12, 24),
    line_height: int = 22,
    color: tuple[int, int, int] = (0, 0, 255),
) -> None:
    for idx, line in enumerate(lines):
        cv2.putText(
            image,
            line,
            (origin[0], origin[1] + idx * line_height),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.58,
            color,
            2,
        )


def _resolve_hsv_range(
    color_preset: Literal["red", "green", "blue", "yellow"],
) -> list[tuple[np.ndarray, np.ndarray]]:
    if color_preset == "red":
        return [
            (
                np.array(COLOR_PRESETS["red1"][0], dtype=np.uint8),
                np.array(COLOR_PRESETS["red1"][1], dtype=np.uint8),
            ),
            (
                np.array(COLOR_PRESETS["red2"][0], dtype=np.uint8),
                np.array(COLOR_PRESETS["red2"][1], dtype=np.uint8),
            ),
        ]

    lower_key = color_preset
    upper_key = color_preset
    lower, upper = COLOR_PRESETS[lower_key]
    return [(np.array(lower, dtype=np.uint8), np.array(upper, dtype=np.uint8))]


def _build_aruco_detector(dict_name: str):
    aruco_dict = cv2.aruco.getPredefinedDictionary(ARUCO_DICT_NAMES[dict_name])
    params = cv2.aruco.DetectorParameters()
    if hasattr(cv2.aruco, "ArucoDetector"):
        return cv2.aruco.ArucoDetector(aruco_dict, params), aruco_dict, params
    return None, aruco_dict, params


def _detect_target_pixel_aruco(
    image_bgr: np.ndarray,
    marker_id: int,
    detector_bundle,
) -> tuple[Optional[tuple[int, int]], Optional[np.ndarray]]:
    detector, aruco_dict, params = detector_bundle
    if detector is not None:
        corners, ids, _ = detector.detectMarkers(image_bgr)
    else:
        corners, ids, _ = cv2.aruco.detectMarkers(
            image_bgr, aruco_dict, parameters=params
        )

    vis = image_bgr.copy()
    if ids is None or len(ids) == 0:
        return None, vis

    ids_flat = ids.reshape(-1)
    cv2.aruco.drawDetectedMarkers(vis, corners, ids)
    for idx, current_id in enumerate(ids_flat):
        if int(current_id) != int(marker_id):
            continue
        pts = np.asarray(corners[idx], dtype=np.float32).reshape(-1, 2)
        center = np.mean(pts, axis=0)
        center_px = (int(round(center[0])), int(round(center[1])))
        return center_px, vis
    return None, vis


def _detect_target_pixel_color(
    image_bgr: np.ndarray,
    hsv_ranges: list[tuple[np.ndarray, np.ndarray]],
    min_area: float,
) -> tuple[Optional[tuple[int, int]], Optional[np.ndarray]]:
    hsv = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HSV)
    mask = np.zeros(hsv.shape[:2], dtype=np.uint8)
    for lower, upper in hsv_ranges:
        mask = cv2.bitwise_or(mask, cv2.inRange(hsv, lower, upper))

    kernel = np.ones((5, 5), dtype=np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

    contours, _ = cv2.findContours(
        mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    vis = image_bgr.copy()
    if not contours:
        return None, vis

    best = max(contours, key=cv2.contourArea)
    area = float(cv2.contourArea(best))
    if area < float(min_area):
        return None, vis

    x, y, w, h = cv2.boundingRect(best)
    cx = int(round(x + w / 2.0))
    cy = int(round(y + h / 2.0))
    cv2.rectangle(vis, (x, y), (x + w, y + h), (0, 255, 255), 2)
    cv2.circle(vis, (cx, cy), 5, (0, 255, 0), -1)
    return (cx, cy), vis


def _detect_target_pixel_pointing(
    image_rgb: np.ndarray,
    text_prompt: str,
    all_prompt: str,
) -> tuple[Optional[tuple[int, int]], Optional[np.ndarray], Optional[str]]:
    from arx_pointing import predict_point_from_rgb

    try:
        point_result = predict_point_from_rgb(
            image_rgb,
            text_prompt=text_prompt,
            all_prompt=all_prompt,
            assume_bgr=False,
            return_raw=True,
        )
    except Exception as exc:
        return None, image_rgb.copy(), f"pointing failed: {exc}"

    point, raw = point_result
    vis = image_rgb.copy()
    if point is None:
        return None, vis, raw

    center_px = (int(round(point[0])), int(round(point[1])))
    cv2.circle(vis, center_px, 6, (255, 255, 0), -1)
    return center_px, vis, raw


def _median_depth_in_patch(depth_image: np.ndarray, center_px: tuple[int, int], radius: int) -> float:
    u, v = center_px
    height, width = depth_image.shape[:2]
    x0 = max(0, u - radius)
    x1 = min(width, u + radius + 1)
    y0 = max(0, v - radius)
    y1 = min(height, v + radius + 1)
    patch = depth_image[y0:y1, x0:x1].astype(np.float32, copy=False)
    valid = patch[np.isfinite(patch) & (patch > 0)]
    if valid.size == 0:
        raise ValueError("no valid depth in patch")
    return depth_to_meters(float(np.median(valid)))


def _pixel_to_ref_point_with_patch(
    pixel: tuple[int, int],
    depth_image: np.ndarray,
    intrinsics: np.ndarray,
    cam2ref: np.ndarray,
    patch_radius: int,
) -> np.ndarray:
    u, v = int(round(pixel[0])), int(round(pixel[1]))
    height, width = depth_image.shape[:2]
    if not (0 <= u < width and 0 <= v < height):
        raise ValueError(f"pixel out of bounds: {(u, v)}")

    z = _median_depth_in_patch(depth_image, (u, v), patch_radius)
    fx, fy, cx, cy = (
        float(intrinsics[0, 0]),
        float(intrinsics[1, 1]),
        float(intrinsics[0, 2]),
        float(intrinsics[1, 2]),
    )
    x_cam = (u - cx) * z / fx
    y_cam = (v - cy) * z / fy
    cam_point = np.array([x_cam, y_cam, z, 1.0], dtype=np.float64)
    ref_point = cam2ref @ cam_point
    return ref_point[:3].astype(np.float32)


def _clip_delta(delta_xyz: np.ndarray, max_step: float) -> np.ndarray:
    return np.clip(delta_xyz, -float(max_step), float(max_step)).astype(np.float32)


def _read_arm_state(status, arm: str) -> tuple[np.ndarray, float]:
    arm_status = status.get(arm) if isinstance(status, dict) else None
    if arm_status is None:
        raise RuntimeError(f"{arm} arm status unavailable")

    end_pos = np.asarray(getattr(arm_status, "end_pos", None),
                         dtype=np.float32).reshape(-1)
    joint_pos = np.asarray(
        getattr(arm_status, "joint_pos", None), dtype=np.float32).reshape(-1)
    if end_pos.shape[0] < 6 or joint_pos.shape[0] < 7:
        raise RuntimeError(f"{arm} arm status malformed")
    return end_pos[:6].copy(), float(joint_pos[6])


FIXED_RPY = np.array([0.0, 0.0, 0.0], dtype=np.float32)


def run_track_object(
    arx: ARXRobotEnv,
    arm: Literal["left", "right"] = "right",
    tracker: Literal["aruco", "color", "pointing"] = "color",
    marker_id: int = 0,
    aruco_dict: str = "4x4_50",
    color_preset: Literal["red", "green", "blue", "yellow"] = "green",
    hz: float = 15.0,
    kp: float = 0.35,
    max_step: float = 0.005,
    deadband: float = 0.005,
    ema_alpha: float = 0.2,
    depth_patch: int = 2,
    offset_x: float = 0.05,
    offset_y: float = 0.0,
    offset_z: float = 0.0,
    lost_stop_frames: int = 100,
    pointing_prompt: str = "",
    pointing_all_prompt: str = "",
    show_window: bool = False,
) -> None:
    if not hasattr(cv2, "aruco") and tracker == "aruco":
        raise RuntimeError(
            "cv2.aruco is unavailable, please install opencv-contrib-python")
    if tracker == "pointing" and not (pointing_prompt or pointing_all_prompt):
        raise ValueError(
            "tracker='pointing' requires pointing_prompt or pointing_all_prompt")

    intrinsics = load_intrinsics()
    cam2ref = load_cam2ref(side=arm)
    target_offset = _default_grasp_offset(arm)
    target_offset[0] = float(offset_x)
    target_offset[1] = float(offset_y)
    target_offset[2] = float(offset_z)

    hz = max(float(hz), 1.0)
    loop_dt = 1.0 / hz
    ema_alpha = float(np.clip(ema_alpha, 0.0, 1.0))

    camera_size = getattr(arx, "img_size", None)
    if camera_size is None:
        raise RuntimeError("arx.img_size is unavailable")

    hsv_ranges = _resolve_hsv_range(color_preset=color_preset) if tracker == "color" else []
    aruco_detector = _build_aruco_detector(
        aruco_dict) if tracker == "aruco" else None

    filtered_target_ref: Optional[np.ndarray] = None
    lost_frames = 0
    while True:
        tick_t0 = time.monotonic()
        frames, status = arx.get_camera(
            target_size=tuple(camera_size),
            return_status=True,
        )
        color = frames.get("camera_h_color") if isinstance(
            frames, dict) else None
        depth = (
            frames.get("camera_h_aligned_depth_to_color")
            if isinstance(frames, dict)
            else None
        )
        if color is None or depth is None:
            print("camera_h frame unavailable, retrying")
            time.sleep(loop_dt)
            continue

        if tracker == "aruco":
            center_px, vis = _detect_target_pixel_aruco(
                color,
                marker_id=marker_id,
                detector_bundle=aruco_detector,
            )
            pointing_raw = None
        elif tracker == "color":
            center_px, vis = _detect_target_pixel_color(
                color,
                hsv_ranges=hsv_ranges,
                min_area=COLOR_MIN_AREA,
            )
            pointing_raw = None
        else:
            center_px, vis, pointing_raw = _detect_target_pixel_pointing(
                color,
                text_prompt=pointing_prompt,
                all_prompt=pointing_all_prompt,
            )

        try:
            end_pos, current_gripper = _read_arm_state(status, arm)
        except RuntimeError as exc:
            print(f"{exc}, retrying")
            time.sleep(loop_dt)
            continue
        current_xyz = end_pos[:3]

        err_xyz = None
        servo_delta = np.zeros(3, dtype=np.float32)
        raw_target_ref = None

        if center_px is None:
            lost_frames += 1
            if lost_frames >= int(lost_stop_frames):
                filtered_target_ref = None
        else:
            lost_frames = 0
            try:
                raw_target_ref = _pixel_to_ref_point_with_patch(
                    center_px,
                    depth_image=depth,
                    intrinsics=intrinsics,
                    cam2ref=cam2ref,
                    patch_radius=max(int(depth_patch), 0),
                )
            except ValueError:
                raw_target_ref = None

            if raw_target_ref is not None:
                if filtered_target_ref is None:
                    filtered_target_ref = raw_target_ref.copy()
                else:
                    filtered_target_ref = (
                        (1.0 - ema_alpha) * filtered_target_ref
                        + ema_alpha * raw_target_ref
                    ).astype(np.float32)

        if filtered_target_ref is not None:
            desired_xyz = filtered_target_ref + target_offset
            err_xyz = desired_xyz - current_xyz
            if float(np.linalg.norm(err_xyz)) >= float(deadband):
                servo_delta = _clip_delta(
                    float(kp) * err_xyz,
                    max_step=float(max_step),
                )
                next_xyz = current_xyz + servo_delta
                arx.step_raw_eef(
                    {
                        arm: np.array(
                            [
                                next_xyz[0],
                                next_xyz[1],
                                next_xyz[2],
                                FIXED_RPY[0],
                                FIXED_RPY[1],
                                FIXED_RPY[2],
                                current_gripper,
                            ],
                            dtype=np.float32,
                        )
                    }
                )

        if vis is None:
            vis = color.copy()
        if center_px is not None:
            cv2.circle(vis, center_px, 6, (0, 255, 0), -1)

        lines = [
            f"arm={arm} tracker={tracker} hz={hz:.1f}",
            f"lost_frames={lost_frames}",
            f"offset=({target_offset[0]:+.3f}, {target_offset[1]:+.3f}, {target_offset[2]:+.3f})",
        ]
        if tracker == "pointing" and pointing_prompt:
            lines.append(f"prompt={pointing_prompt}")
        if raw_target_ref is not None:
            lines.append(
                f"obj_ref=({raw_target_ref[0]:+.3f}, {raw_target_ref[1]:+.3f}, {raw_target_ref[2]:+.3f})"
            )
        if err_xyz is not None:
            lines.append(
                f"err=({err_xyz[0]:+.3f}, {err_xyz[1]:+.3f}, {err_xyz[2]:+.3f})"
            )
            lines.append(
                f"delta=({servo_delta[0]:+.3f}, {servo_delta[1]:+.3f}, {servo_delta[2]:+.3f})"
            )
        if tracker == "pointing" and pointing_raw:
            raw_short = pointing_raw.replace("\n", " ").strip()
            lines.append(f"raw={raw_short[:80]}")
        _draw_text_lines(vis, lines)

        if show_window:
            cv2.imshow("track_object", vis)
            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), 27):
                break

        elapsed = time.monotonic() - tick_t0
        sleep_time = loop_dt - elapsed
        if sleep_time > 0:
            time.sleep(sleep_time)


def main() -> None:
    arm: Literal["left", "right"] = "right"
    tracker: Literal["aruco", "color", "pointing"] = "pointing"
    marker_id = 0
    aruco_dict = "4x4_50"
    color_preset: Literal["red", "green", "blue", "yellow"] = "green"
    hz = 15.0
    kp = 0.35
    max_step = 0.005
    deadband = 0.005
    ema_alpha = 0.2
    depth_patch = 2
    lift = 0.0
    offset_x = -0.15
    offset_y = 0.0
    offset_z = 0.0
    lost_stop_frames = 100
    pointing_prompt = ""
    pointing_all_prompt = ""
    show_window = True
    skip_reset = False

    env = ARXRobotEnv(
        duration_per_step=1.0 / 20.0,
        min_steps=10,
        max_v_xyz=0.20,
        max_a_xyz=0.20,
        max_v_rpy=0.30,
        max_a_rpy=1.00,
        camera_type="all",
        camera_view=("camera_h",),
        img_size=(640, 480),
    )
    try:
        if not skip_reset:
            env.reset()
        if abs(float(lift)) > 1e-6:
            env.step_lift(float(lift))
        run_track_object(
            arx=env,
            arm=arm,
            tracker=tracker,
            marker_id=marker_id,
            aruco_dict=aruco_dict,
            color_preset=color_preset,
            hz=hz,
            kp=kp,
            max_step=max_step,
            deadband=deadband,
            ema_alpha=ema_alpha,
            depth_patch=depth_patch,
            offset_x=offset_x,
            offset_y=offset_y,
            offset_z=offset_z,
            lost_stop_frames=lost_stop_frames,
            pointing_prompt=pointing_prompt,
            pointing_all_prompt=pointing_all_prompt,
            show_window=show_window,
        )
    finally:
        if show_window:
            cv2.destroyAllWindows()
        if env is not None:
            env.close()


if __name__ == "__main__":
    main()
