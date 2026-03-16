from __future__ import annotations

import importlib
import json
import select
import sys
import threading
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, Literal, Optional

import numpy as np


ROOT_DIR = Path(__file__).resolve().parent.parent
ROS2_ROOT = (ROOT_DIR / "ARX_Realenv" / "ROS2").resolve()
if str(ROS2_ROOT) not in sys.path:
    sys.path.insert(0, str(ROS2_ROOT))

from arx_ros2_env import ARXRobotEnv
from arx_ros2_env_utils import _quat_from_rpy, _quat_multiply, _rpy_from_quat
from arm_control.msg._pos_cmd import PosCmd


KIWI_WHEEL_RADIUS_M = 0.15
KIWI_BASE_RADIUS_M = 0.376
SQRT3 = float(np.sqrt(3.0))
EPISODE_SCHEMA_VERSION = "collect/v3"
GRIPPER_OPEN_VALUE = -3.4
GRIPPER_CLOSED_VALUE = 0.0


def _stamp_to_float(stamp_like: Any) -> Optional[float]:
    if stamp_like is None:
        return None
    if hasattr(stamp_like, "sec") and hasattr(stamp_like, "nanosec"):
        return float(stamp_like.sec) + float(stamp_like.nanosec) * 1e-9
    if hasattr(stamp_like, "stamp"):
        return _stamp_to_float(stamp_like.stamp)
    return None


def _topic_stamps_to_serializable(topic_stamps: Dict[str, Optional[float]]) -> Dict[str, Optional[float]]:
    return {
        str(key): (None if value is None else float(value))
        for key, value in topic_stamps.items()
    }


def poll_stdin_line() -> Optional[str]:
    if not sys.stdin.isatty():
        return None
    ready, _, _ = select.select([sys.stdin], [], [], 0.0)
    if not ready:
        return None
    return sys.stdin.readline().strip().lower()


def episode_name(episode_idx: int) -> str:
    return f"episode_{episode_idx:06d}"


def find_next_episode_index(root: Path) -> int:
    root = Path(root)
    if not root.exists():
        return 0
    max_idx = -1
    for child in root.iterdir():
        if not child.is_dir() or not child.name.startswith("episode_"):
            continue
        try:
            max_idx = max(max_idx, int(child.name.split("_", 1)[1]))
        except ValueError:
            continue
    return max_idx + 1


def latest_episode_dir(root: Path) -> Optional[Path]:
    root = Path(root)
    if not root.exists():
        return None
    candidates = sorted(
        child for child in root.iterdir()
        if child.is_dir() and child.name.startswith("episode_")
    )
    return candidates[-1] if candidates else None


def default_camera_map(camera_names: Iterable[str]) -> Dict[str, str]:
    return {str(name): str(name) for name in camera_names}


def _arm_qpos(status) -> np.ndarray:
    return np.asarray(status.joint_pos, dtype=np.float32).reshape(-1)[:7].copy()


def _arm_qvel(status) -> np.ndarray:
    return np.asarray(status.joint_vel, dtype=np.float32).reshape(-1)[:7].copy()


def _arm_effort(status) -> np.ndarray:
    return np.asarray(status.joint_cur, dtype=np.float32).reshape(-1)[:7].copy()


def _arm_eef(status) -> np.ndarray:
    end_pos = np.asarray(status.end_pos, dtype=np.float32).reshape(-1)[:6]
    gripper = np.asarray([status.joint_pos[6]], dtype=np.float32)
    return np.concatenate([end_pos, gripper], axis=0).astype(np.float32)


def _vr_arm_from_msg(msg) -> np.ndarray:
    return np.array(
        [
            float(getattr(msg, "x", 0.0)),
            float(getattr(msg, "y", 0.0)),
            float(getattr(msg, "z", 0.0)),
            float(getattr(msg, "roll", 0.0)),
            float(getattr(msg, "pitch", 0.0)),
            float(getattr(msg, "yaw", 0.0)),
            float(getattr(msg, "gripper", 0.0)),
        ],
        dtype=np.float32,
    )


def _vr_base_from_msg(msg) -> np.ndarray:
    return np.array(
        [
            float(getattr(msg, "chx", 0.0)),
            float(getattr(msg, "chy", 0.0)),
            float(getattr(msg, "chz", 0.0)),
            float(getattr(msg, "height", 0.0)),
        ],
        dtype=np.float32,
    )


def _base_wheels(base_status) -> np.ndarray:
    if base_status is None:
        return np.zeros((3,), dtype=np.float32)
    raw = list(getattr(base_status, "temp_float_data", []))
    wheels = raw[1:4] if len(raw) >= 4 else raw[:3]
    wheels = wheels + [0.0] * max(0, 3 - len(wheels))
    return np.asarray(wheels[:3], dtype=np.float32)


def _base_velocity_from_wheels(base_wheels: np.ndarray) -> np.ndarray:
    omega1, omega2, omega3 = np.asarray(base_wheels, dtype=np.float32)
    return np.array(
        [
            KIWI_WHEEL_RADIUS_M * (2.0 * omega1 - omega2 - omega3) / 3.0,
            KIWI_WHEEL_RADIUS_M * (omega2 - omega3) / SQRT3,
            KIWI_WHEEL_RADIUS_M * (omega1 + omega2 + omega3) / (3.0 * KIWI_BASE_RADIUS_M),
        ],
        dtype=np.float32,
    )


def _deadband_vector(deadband: float | Iterable[float], dim: int) -> np.ndarray:
    values = np.asarray(deadband, dtype=np.float32).reshape(-1)
    if values.shape[0] == 1:
        return np.full((dim,), float(values[0]), dtype=np.float32)
    if values.shape[0] != dim:
        raise ValueError(f"Deadband dim mismatch: expected 1 or {dim}, got {values.shape[0]}")
    return values.astype(np.float32, copy=True)


def _smooth_target(
    target: np.ndarray,
    previous: Optional[np.ndarray],
    alpha: float,
    deadband: float | Iterable[float],
) -> np.ndarray:
    target = np.asarray(target, dtype=np.float32).reshape(-1)
    if previous is None:
        return target.copy()
    alpha = float(alpha)
    if alpha <= 0.0 or alpha > 1.0:
        raise ValueError(f"alpha must be in (0, 1], got {alpha}")
    previous = np.asarray(previous, dtype=np.float32).reshape(-1)
    if previous.shape != target.shape:
        raise ValueError(f"Target shape mismatch: {previous.shape} vs {target.shape}")
    filtered = alpha * target + (1.0 - alpha) * previous
    thresholds = _deadband_vector(deadband, filtered.shape[0])
    delta = np.abs(filtered - previous)
    filtered[delta < thresholds] = previous[delta < thresholds]
    return filtered.astype(np.float32, copy=False)


@dataclass(frozen=True)
class SpaceMouseSample:
    translation: np.ndarray
    rotation: np.ndarray
    buttons: tuple[bool, ...]
    timestamp: float


def _value_from_state(raw: Any, *names: str, default: float = 0.0) -> float:
    for name in names:
        if isinstance(raw, dict) and name in raw:
            try:
                return float(raw[name])
            except Exception:
                continue
        if hasattr(raw, name):
            try:
                return float(getattr(raw, name))
            except Exception:
                continue
    return float(default)


def _buttons_from_state(raw: Any) -> tuple[bool, ...]:
    payload = None
    if isinstance(raw, dict):
        payload = raw.get("buttons")
    elif hasattr(raw, "buttons"):
        payload = getattr(raw, "buttons")
    if payload is not None:
        if isinstance(payload, int):
            return (bool(payload & 0x1), bool(payload & 0x2))
        if isinstance(payload, np.ndarray):
            payload = payload.reshape(-1).tolist()
        if isinstance(payload, (list, tuple)):
            return tuple(bool(value) for value in payload)
    return (
        bool(_value_from_state(raw, "button_0", "button0", "b0", "left_button", default=0.0)),
        bool(_value_from_state(raw, "button_1", "button1", "b1", "right_button", default=0.0)),
    )


def _normalize_axis_signs(signs: Iterable[float], dim: int, label: str) -> np.ndarray:
    values = np.asarray(tuple(signs), dtype=np.float32).reshape(-1)
    if values.shape[0] != dim:
        raise ValueError(f"{label} must have {dim} values, got {values.shape[0]}")
    return values.astype(np.float32, copy=True)


def _apply_axis_deadzone(values: np.ndarray, deadzone: float, exponent: float) -> np.ndarray:
    arr = np.asarray(values, dtype=np.float32).reshape(-1).copy()
    deadzone = max(float(deadzone), 0.0)
    if deadzone > 0.0:
        arr[np.abs(arr) < deadzone] = 0.0
    exponent = max(float(exponent), 1.0)
    if exponent != 1.0:
        arr = np.sign(arr) * np.power(np.abs(arr), exponent)
    return arr.astype(np.float32, copy=False)


def _clip_gripper(value: float) -> float:
    return float(np.clip(value, GRIPPER_OPEN_VALUE, GRIPPER_CLOSED_VALUE))


def _compose_eef_target(current: np.ndarray, delta_xyz: np.ndarray, delta_rpy: np.ndarray, gripper_delta: float) -> np.ndarray:
    current = np.asarray(current, dtype=np.float32).reshape(-1)
    if current.shape[0] < 7:
        raise ValueError(f"EEF payload must be 7D, got {current.shape[0]}")
    target_xyz = current[:3] + np.asarray(delta_xyz, dtype=np.float32).reshape(3)
    q_current = _quat_from_rpy(current[3:6])
    q_delta = _quat_from_rpy(np.asarray(delta_rpy, dtype=np.float32).reshape(3))
    q_target = _quat_multiply(q_delta, q_current)
    target_rpy = _rpy_from_quat(q_target)
    target_gripper = _clip_gripper(float(current[6]) + float(gripper_delta))
    return np.concatenate([target_xyz, target_rpy, [target_gripper]], axis=0).astype(np.float32)


def _effective_motion(delta_xyz: np.ndarray, delta_rpy: np.ndarray, gripper_delta: float) -> bool:
    if np.max(np.abs(np.asarray(delta_xyz, dtype=np.float32))) > 1e-6:
        return True
    if np.max(np.abs(np.asarray(delta_rpy, dtype=np.float32))) > 1e-6:
        return True
    return abs(float(gripper_delta)) > 1e-6


class SpaceMouseDevice:
    def __init__(self):
        try:
            self._backend = importlib.import_module("pyspacemouse")
        except ImportError as exc:  # pragma: no cover
            raise ImportError(
                "3D mouse collection requires the optional `pyspacemouse` package. "
                "Install it in the active Python environment before using collect_3dmouse_*."
            ) from exc
        opener = getattr(self._backend, "open", None)
        if opener is None:
            raise RuntimeError("`pyspacemouse` backend does not expose open().")
        try:
            opened = opener()
        except TypeError:
            opened = opener(dof_callback=None, button_callback=None)
        if opened is False:
            raise RuntimeError(
                "Failed to open SpaceMouse device. Ensure the device is connected and accessible."
            )

    def read(self) -> SpaceMouseSample:
        reader = getattr(self._backend, "read", None)
        if reader is None:
            raise RuntimeError("`pyspacemouse` backend does not expose read().")
        raw = reader()
        if raw is None:
            return SpaceMouseSample(
                translation=np.zeros((3,), dtype=np.float32),
                rotation=np.zeros((3,), dtype=np.float32),
                buttons=(False, False),
                timestamp=time.time(),
            )
        translation = np.array(
            [
                _value_from_state(raw, "x", default=0.0),
                _value_from_state(raw, "y", default=0.0),
                _value_from_state(raw, "z", default=0.0),
            ],
            dtype=np.float32,
        )
        rotation = np.array(
            [
                _value_from_state(raw, "roll", "rx", default=0.0),
                _value_from_state(raw, "pitch", "ry", default=0.0),
                _value_from_state(raw, "yaw", "rz", default=0.0),
            ],
            dtype=np.float32,
        )
        return SpaceMouseSample(
            translation=translation,
            rotation=rotation,
            buttons=_buttons_from_state(raw),
            timestamp=time.time(),
        )

    def close(self) -> None:
        closer = getattr(self._backend, "close", None)
        if closer is None:
            return
        try:
            closer()
        except Exception:
            pass


def _capture_camera_and_status(
    env: ARXRobotEnv,
    include_camera: bool,
    target_size: tuple[int, int],
) -> tuple[Dict[str, np.ndarray], Dict[str, Any]]:
    if include_camera:
        camera_frames, status = env.get_camera(
            save_dir=None,
            video=False,
            target_size=target_size,
            return_status=True,
        )
        return camera_frames, status
    return {}, env.get_robot_status()


def _extract_camera_frames(
    camera_frames: Dict[str, np.ndarray],
    camera_map: Dict[str, str],
    include_depth: bool,
) -> tuple[Dict[str, np.ndarray], Dict[str, np.ndarray]]:
    colors: Dict[str, np.ndarray] = {}
    depths: Dict[str, np.ndarray] = {}
    for physical_name, logical_name in camera_map.items():
        color_key = f"{physical_name}_color"
        depth_key = f"{physical_name}_aligned_depth_to_color"
        if color_key in camera_frames:
            colors[logical_name] = np.asarray(camera_frames[color_key])
        if include_depth and depth_key in camera_frames:
            depths[logical_name] = np.asarray(camera_frames[depth_key])
    return colors, depths


def _camera_ready(
    env: ARXRobotEnv,
    camera_names: Iterable[str],
    camera_type: str,
) -> list[str]:
    missing = []
    with env.node.cam_lock:
        image_keys = set(env.node.latest_images.keys())
    for camera_name in camera_names:
        color_key = f"{camera_name}_color"
        depth_key = f"{camera_name}_aligned_depth_to_color"
        if color_key not in image_keys:
            missing.append(color_key)
        if camera_type in {"all", "depth"} and depth_key not in image_keys:
            missing.append(depth_key)
    return missing


def _build_topic_stamps(
    env: ARXRobotEnv,
    left_status: Any = None,
    right_status: Any = None,
    base_status: Any = None,
    left_vr_stamp: Optional[float] = None,
    right_vr_stamp: Optional[float] = None,
) -> Dict[str, Optional[float]]:
    stamps: Dict[str, Optional[float]] = {}
    with env.node.cam_lock:
        for key, msg in env.node.latest_images.items():
            stamps[f"camera:{key}"] = _stamp_to_float(getattr(msg, "header", None))
    if left_status is not None:
        stamps["arm_status_left"] = _stamp_to_float(getattr(left_status, "header", None))
    if right_status is not None:
        stamps["arm_status_right"] = _stamp_to_float(getattr(right_status, "header", None))
    if base_status is not None:
        stamps["base_status"] = _stamp_to_float(getattr(base_status, "header", None))
    if left_vr_stamp is not None:
        stamps["vr_left"] = left_vr_stamp
    if right_vr_stamp is not None:
        stamps["vr_right"] = right_vr_stamp
    return stamps


def _set_gravity_mode(env: ARXRobotEnv, side: Literal["left", "right"]) -> None:
    success, error_message = env.set_special_mode(3, side=side)
    if not success:
        raise RuntimeError(error_message or f"Failed to enable gravity mode for {side}")


def _expected_dim(mode: Literal["dual", "single"]) -> int:
    return 14 if mode == "dual" else 7


def _split_dual(array: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    payload = np.asarray(array, dtype=np.float32).reshape(-1)
    if payload.shape[0] != 14:
        raise ValueError(f"Expected dual-arm payload dim 14, got {payload.shape[0]}")
    return payload[:7].copy(), payload[7:14].copy()


@dataclass
class EpisodeFrame:
    frame_idx: int
    timestamp: float
    qpos: np.ndarray
    qvel: np.ndarray
    effort: np.ndarray
    eef: np.ndarray
    action: np.ndarray
    images: Dict[str, np.ndarray] = field(default_factory=dict)
    images_depth: Dict[str, np.ndarray] = field(default_factory=dict)
    robot_base: Optional[np.ndarray] = None
    base_wheels: Optional[np.ndarray] = None
    base_velocity: Optional[np.ndarray] = None
    action_base: Optional[np.ndarray] = None
    topic_stamps: Dict[str, Optional[float]] = field(default_factory=dict)

    def to_manifest_dict(self) -> dict:
        return {
            "frame_idx": int(self.frame_idx),
            "timestamp": float(self.timestamp),
            "topic_stamps": _topic_stamps_to_serializable(self.topic_stamps),
        }


@dataclass
class EpisodeBuffer:
    episode_idx: int
    mode: Literal["dual", "single"]
    frame_rate: float
    action_kind: Literal["joint", "eef"]
    include_camera: bool
    include_base: bool
    camera_map: Dict[str, str]
    config: Dict[str, Any]
    side: Optional[Literal["left", "right"]] = None
    created_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    frames: list[EpisodeFrame] = field(default_factory=list)

    @property
    def frame_count(self) -> int:
        return len(self.frames)

    @property
    def dim(self) -> int:
        return _expected_dim(self.mode)

    def add_frame(self, frame: EpisodeFrame) -> None:
        dim = self.dim
        for name in ("qpos", "qvel", "effort", "eef", "action"):
            array = np.asarray(getattr(frame, name), dtype=np.float32).reshape(-1)
            if array.shape[0] != dim:
                raise ValueError(f"{name} dim mismatch: expected {dim}, got {array.shape[0]}")
        expected_cameras = set(self.camera_map.values())
        if self.include_camera and set(frame.images.keys()) != expected_cameras:
            raise ValueError(
                f"Camera keys mismatch: expected {sorted(expected_cameras)}, got {sorted(frame.images.keys())}"
            )
        if not self.include_base:
            frame.robot_base = None
            frame.base_wheels = None
            frame.base_velocity = None
            frame.action_base = None
        self.frames.append(frame)


def create_episode_buffer(
    episode_idx: int,
    mode: Literal["dual", "single"],
    frame_rate: float,
    action_kind: Literal["joint", "eef"],
    include_camera: bool,
    include_base: bool,
    camera_names: Iterable[str],
    config: Dict[str, Any],
    side: Optional[Literal["left", "right"]] = None,
) -> EpisodeBuffer:
    return EpisodeBuffer(
        episode_idx=int(episode_idx),
        mode=mode,
        frame_rate=float(frame_rate),
        action_kind=action_kind,
        include_camera=bool(include_camera),
        include_base=bool(include_base),
        camera_map=default_camera_map(camera_names if include_camera else ()),
        config=dict(config),
        side=side,
    )


def _normalize_action_kind(action_kind: str) -> Literal["joint", "eef"]:
    kind = str(action_kind).strip().lower()
    if kind == "vr":
        return "eef"
    if kind in {"joint", "eef"}:
        return kind
    raise ValueError(f"Unsupported action kind: {action_kind}")


def save_episode(episode: EpisodeBuffer, out_dir: Path) -> Path:
    if not episode.frames:
        raise ValueError("Episode has no frames.")

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    episode_dir = out_dir / episode_name(episode.episode_idx)
    episode_dir.mkdir(parents=True, exist_ok=False)

    low_dim_payload: Dict[str, np.ndarray] = {
        "timestamps": np.asarray([frame.timestamp for frame in episode.frames], dtype=np.float32),
        "qpos": np.stack([frame.qpos for frame in episode.frames], axis=0),
        "qvel": np.stack([frame.qvel for frame in episode.frames], axis=0),
        "effort": np.stack([frame.effort for frame in episode.frames], axis=0),
        "eef": np.stack([frame.eef for frame in episode.frames], axis=0),
        "action": np.stack([frame.action for frame in episode.frames], axis=0),
    }
    if episode.include_base:
        low_dim_payload["robot_base"] = np.stack([frame.robot_base for frame in episode.frames], axis=0)
        low_dim_payload["base_wheels"] = np.stack([frame.base_wheels for frame in episode.frames], axis=0)
        low_dim_payload["base_velocity"] = np.stack([frame.base_velocity for frame in episode.frames], axis=0)
        low_dim_payload["action_base"] = np.stack([frame.action_base for frame in episode.frames], axis=0)

    low_dim_path = episode_dir / "low_dim.npz"
    np.savez_compressed(low_dim_path, **low_dim_payload)

    image_files: Dict[str, str] = {}
    depth_files: Dict[str, str] = {}
    if episode.include_camera and episode.camera_map:
        image_root = episode_dir / "images"
        image_root.mkdir(parents=True, exist_ok=True)
        for logical_name in episode.camera_map.values():
            frames = np.stack([frame.images[logical_name] for frame in episode.frames], axis=0)
            file_path = image_root / f"{logical_name}.npz"
            np.savez_compressed(file_path, frames=frames)
            image_files[logical_name] = str(file_path.relative_to(episode_dir))

        first_depth_keys = set(episode.frames[0].images_depth.keys())
        if first_depth_keys:
            depth_root = episode_dir / "images_depth"
            depth_root.mkdir(parents=True, exist_ok=True)
            for logical_name in sorted(first_depth_keys):
                frames = np.stack([frame.images_depth[logical_name] for frame in episode.frames], axis=0)
                file_path = depth_root / f"{logical_name}.npz"
                np.savez_compressed(file_path, frames=frames)
                depth_files[logical_name] = str(file_path.relative_to(episode_dir))

    manifest = {
        "schema_version": EPISODE_SCHEMA_VERSION,
        "episode_idx": int(episode.episode_idx),
        "mode": episode.mode,
        "side": episode.side,
        "dim": int(episode.dim),
        "frame_rate": float(episode.frame_rate),
        "frame_count": int(episode.frame_count),
        "action_kind": episode.action_kind,
        "include_camera": bool(episode.include_camera),
        "include_base": bool(episode.include_base),
        "camera_map": dict(episode.camera_map),
        "created_at": episode.created_at,
        "config": dict(episode.config),
        "frames": [frame.to_manifest_dict() for frame in episode.frames],
        "files": {
            "low_dim": str(low_dim_path.relative_to(episode_dir)),
            "images": image_files,
            "images_depth": depth_files,
        },
    }
    (episode_dir / "episode.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return episode_dir


def load_episode(episode_dir: Path) -> EpisodeBuffer:
    episode_dir = Path(episode_dir)
    manifest = json.loads((episode_dir / "episode.json").read_text(encoding="utf-8"))
    low_dim = np.load(episode_dir / manifest["files"]["low_dim"])
    images = {
        logical_name: np.load(episode_dir / rel_path)["frames"]
        for logical_name, rel_path in manifest["files"].get("images", {}).items()
    }
    images_depth = {
        logical_name: np.load(episode_dir / rel_path)["frames"]
        for logical_name, rel_path in manifest["files"].get("images_depth", {}).items()
    }

    episode = EpisodeBuffer(
        episode_idx=int(manifest["episode_idx"]),
        mode=str(manifest["mode"]),
        frame_rate=float(manifest["frame_rate"]),
        action_kind=_normalize_action_kind(manifest["action_kind"]),
        include_camera=bool(manifest.get("include_camera", False)),
        include_base=bool(manifest.get("include_base", False)),
        camera_map=dict(manifest.get("camera_map", {})),
        config=dict(manifest.get("config", {})),
        side=manifest.get("side"),
        created_at=str(manifest.get("created_at", "")),
    )

    frame_count = int(manifest.get("frame_count", 0))
    for idx in range(frame_count):
        frame_meta = manifest["frames"][idx]
        episode.add_frame(
            EpisodeFrame(
                frame_idx=int(frame_meta["frame_idx"]),
                timestamp=float(frame_meta["timestamp"]),
                qpos=np.asarray(low_dim["qpos"][idx], dtype=np.float32),
                qvel=np.asarray(low_dim["qvel"][idx], dtype=np.float32),
                effort=np.asarray(low_dim["effort"][idx], dtype=np.float32),
                eef=np.asarray(low_dim["eef"][idx], dtype=np.float32),
                action=np.asarray(low_dim["action"][idx], dtype=np.float32),
                images={key: value[idx] for key, value in images.items()},
                images_depth={key: value[idx] for key, value in images_depth.items()},
                robot_base=np.asarray(low_dim["robot_base"][idx], dtype=np.float32) if "robot_base" in low_dim else None,
                base_wheels=np.asarray(low_dim["base_wheels"][idx], dtype=np.float32) if "base_wheels" in low_dim else None,
                base_velocity=np.asarray(low_dim["base_velocity"][idx], dtype=np.float32) if "base_velocity" in low_dim else None,
                action_base=np.asarray(low_dim["action_base"][idx], dtype=np.float32) if "action_base" in low_dim else None,
                topic_stamps=dict(frame_meta.get("topic_stamps", {})),
            )
        )
    return episode


def wait_until_status_ready(env: ARXRobotEnv, include_base: bool = False) -> None:
    last_report = 0.0
    while True:
        status = env.get_robot_status()
        missing = []
        if status.get("left") is None:
            missing.append("left_arm_status")
        if status.get("right") is None:
            missing.append("right_arm_status")
        if include_base and status.get("base") is None:
            missing.append("base_status")
        if not missing:
            return
        now = time.time()
        if now - last_report > 1.0:
            print(f"Waiting for status: {', '.join(missing)}")
            last_report = now
        time.sleep(0.2)


def record_episode_interactive(
    episode: EpisodeBuffer,
    capture_fn,
    frame_rate: float,
    max_frames: int = 0,
    prompt_start: bool = True,
) -> bool:
    period = 1.0 / max(float(frame_rate), 1e-6)
    quit_requested = False
    last_error_report = 0.0
    if prompt_start:
        print("Press Enter to start recording.")
        input()
    print("Recording started. Press Enter to stop, or type 'q' then Enter to stop and quit.")

    frame_idx = 0
    while True:
        if max_frames > 0 and frame_idx >= int(max_frames):
            print("Reached max frame count.")
            break
        loop_start = time.perf_counter()
        command = poll_stdin_line()
        if command == "":
            break
        if command == "q":
            quit_requested = True
            break

        frame, error = capture_fn(frame_idx)
        if frame is None:
            now = time.time()
            if error and now - last_error_report > 1.0:
                print(f"Skipping frame: {error}")
                last_error_report = now
        else:
            episode.add_frame(frame)
            frame_idx += 1
            if episode.frame_count % 20 == 0:
                print(f"Captured {episode.frame_count} frames")

        sleep_need = period - (time.perf_counter() - loop_start)
        if sleep_need > 0.0:
            time.sleep(sleep_need)
    return quit_requested


class VRCommandMirror:
    def __init__(self, left_topic: str = "/ARX_VR_L", right_topic: str = "/ARX_VR_R"):
        import rclpy
        from rclpy.executors import SingleThreadedExecutor
        from rclpy.node import Node

        if not rclpy.ok():
            raise RuntimeError("rclpy must be initialized before creating VRCommandMirror")

        self._lock = threading.Lock()
        self._latest_left = None
        self._latest_right = None
        self._latest_left_stamp = None
        self._latest_right_stamp = None

        class MirrorNode(Node):
            pass

        self.node = MirrorNode("collect_vr_mirror")
        self.node.create_subscription(PosCmd, left_topic, self._on_left, 10)
        self.node.create_subscription(PosCmd, right_topic, self._on_right, 10)
        self.executor = SingleThreadedExecutor()
        self.executor.add_node(self.node)
        self.thread = threading.Thread(target=self.executor.spin, daemon=True)
        self.thread.start()

    def _on_left(self, msg) -> None:
        with self._lock:
            self._latest_left = msg
            self._latest_left_stamp = time.time()

    def _on_right(self, msg) -> None:
        with self._lock:
            self._latest_right = msg
            self._latest_right_stamp = time.time()

    def snapshot(self) -> tuple[Any, Any, Optional[float], Optional[float]]:
        with self._lock:
            return (
                self._latest_left,
                self._latest_right,
                self._latest_left_stamp,
                self._latest_right_stamp,
            )

    def close(self) -> None:
        self.executor.shutdown()
        self.node.destroy_node()
        self.thread.join(timeout=2.0)


class DualVRCollector:
    def __init__(
        self,
        env: ARXRobotEnv,
        camera_names: Iterable[str] = ("camera_h",),
        include_camera: bool = True,
        include_base: bool = True,
        use_depth: bool = False,
        action_kind: Literal["joint", "eef"] = "joint",
        left_vr_topic: str = "/ARX_VR_L",
        right_vr_topic: str = "/ARX_VR_R",
        img_size: tuple[int, int] = (640, 480),
    ):
        self.env = env
        self.include_camera = bool(include_camera)
        self.include_base = bool(include_base)
        self.action_kind = _normalize_action_kind(action_kind)
        self.camera_names = [str(name) for name in camera_names] if self.include_camera else []
        self.camera_type = "all" if use_depth else "color"
        self.img_size = tuple(img_size)
        if self.include_camera:
            env_cameras = set(str(name) for name in getattr(self.env, "camera_view", ()))
            missing_cameras = [name for name in self.camera_names if name not in env_cameras]
            if missing_cameras:
                raise ValueError(
                    f"env.camera_view does not include requested cameras: {missing_cameras}"
                )
            if use_depth and getattr(self.env, "camera_type", "color") not in {"all", "depth"}:
                raise ValueError("collect requested depth frames, but env.camera_type does not subscribe depth")
        self.vr_mirror = VRCommandMirror(left_topic=left_vr_topic, right_topic=right_vr_topic)
        self._odom_pose = np.zeros((3,), dtype=np.float32)
        self._last_odom_timestamp: Optional[float] = None

    def readiness(self) -> tuple[bool, list[str]]:
        missing = []
        status = self.env.get_robot_status()
        if status.get("left") is None:
            missing.append("left_arm_status")
        if status.get("right") is None:
            missing.append("right_arm_status")
        if self.include_base and status.get("base") is None:
            missing.append("base_status")
        left_vr, right_vr, _, _ = self.vr_mirror.snapshot()
        if left_vr is None:
            missing.append("vr_left")
        if right_vr is None:
            missing.append("vr_right")
        if self.include_camera:
            missing.extend(_camera_ready(self.env, self.camera_names, self.camera_type))
        return len(missing) == 0, missing

    def wait_until_ready(self) -> None:
        last_report = 0.0
        while True:
            ready, missing = self.readiness()
            if ready:
                return
            now = time.time()
            if now - last_report > 1.0:
                print(f"Waiting for streams: {', '.join(missing)}")
                last_report = now
            time.sleep(0.2)

    def reset_episode(self) -> None:
        self._odom_pose = np.zeros((3,), dtype=np.float32)
        self._last_odom_timestamp = None

    def capture_frame(self, frame_idx: int) -> tuple[Optional[EpisodeFrame], Optional[str]]:
        frame_timestamp = time.time()
        camera_frames, status = _capture_camera_and_status(
            self.env,
            include_camera=self.include_camera,
            target_size=self.img_size,
        )
        left_status = status.get("left") if isinstance(status, dict) else None
        right_status = status.get("right") if isinstance(status, dict) else None
        base_status = status.get("base") if isinstance(status, dict) else None
        if left_status is None or right_status is None:
            return None, "arm status not ready"

        color_frames, depth_frames = _extract_camera_frames(
            camera_frames,
            camera_map=default_camera_map(self.camera_names),
            include_depth=self.camera_type == "all",
        )
        expected_cameras = set(default_camera_map(self.camera_names).values())
        if self.include_camera and set(color_frames.keys()) != expected_cameras:
            return None, "camera color frames not ready"
        if self.include_camera and self.camera_type == "all" and set(depth_frames.keys()) != expected_cameras:
            return None, "camera depth frames not ready"

        left_vr, right_vr, left_vr_stamp, right_vr_stamp = self.vr_mirror.snapshot()
        if left_vr is None or right_vr is None:
            return None, "vr topics not ready"

        left_qpos = _arm_qpos(left_status)
        right_qpos = _arm_qpos(right_status)
        left_qvel = _arm_qvel(left_status)
        right_qvel = _arm_qvel(right_status)
        left_effort = _arm_effort(left_status)
        right_effort = _arm_effort(right_status)
        left_eef = _arm_eef(left_status)
        right_eef = _arm_eef(right_status)
        vr_left_arm = _vr_arm_from_msg(left_vr)
        vr_right_arm = _vr_arm_from_msg(right_vr)
        action_base = _vr_base_from_msg(left_vr) if self.include_base else None

        qpos = np.concatenate([left_qpos, right_qpos], axis=0)
        qvel = np.concatenate([left_qvel, right_qvel], axis=0)
        effort = np.concatenate([left_effort, right_effort], axis=0)
        eef = np.concatenate([left_eef, right_eef], axis=0)

        if self.action_kind == "joint":
            action = qpos.copy()
        else:
            action = np.concatenate([vr_left_arm, vr_right_arm], axis=0)

        robot_base = None
        base_wheels = None
        base_velocity = None
        if self.include_base:
            base_wheels = _base_wheels(base_status)
            base_velocity = _base_velocity_from_wheels(base_wheels)
            if base_status is None:
                robot_base = np.zeros((4,), dtype=np.float32)
            else:
                robot_base = np.array(
                    [
                        0.0,
                        0.0,
                        0.0,
                        float(getattr(base_status, "height", 0.0)),
                    ],
                    dtype=np.float32,
                )

        return EpisodeFrame(
            frame_idx=int(frame_idx),
            timestamp=float(frame_timestamp),
            qpos=qpos,
            qvel=qvel,
            effort=effort,
            eef=eef,
            action=np.asarray(action, dtype=np.float32),
            images=color_frames if self.include_camera else {},
            images_depth=depth_frames if self.include_camera else {},
            robot_base=robot_base,
            base_wheels=base_wheels,
            base_velocity=base_velocity,
            action_base=action_base,
            topic_stamps=_build_topic_stamps(
                self.env,
                left_status=left_status,
                right_status=right_status,
                base_status=base_status,
                left_vr_stamp=left_vr_stamp,
                right_vr_stamp=right_vr_stamp,
            ),
        ), None

    def close(self) -> None:
        self.vr_mirror.close()


class DualArmGravityCollector:
    def __init__(
        self,
        env: ARXRobotEnv,
        camera_names: Iterable[str] = (),
        include_camera: bool = False,
        use_depth: bool = False,
        action_kind: Literal["joint", "eef"] = "joint",
        img_size: tuple[int, int] = (640, 480),
    ):
        self.env = env
        self.include_camera = bool(include_camera)
        self.action_kind = str(action_kind)
        self.camera_names = [str(name) for name in camera_names] if self.include_camera else []
        self.camera_type = "all" if use_depth else "color"
        self.img_size = tuple(img_size)
        if self.action_kind not in {"joint", "eef"}:
            raise ValueError("dual-arm gravity collect only supports action_kind='joint' or 'eef'")
        if self.include_camera:
            env_cameras = set(str(name) for name in getattr(self.env, "camera_view", ()))
            missing_cameras = [name for name in self.camera_names if name not in env_cameras]
            if missing_cameras:
                raise ValueError(
                    f"env.camera_view does not include requested cameras: {missing_cameras}"
                )
            if use_depth and getattr(self.env, "camera_type", "color") not in {"all", "depth"}:
                raise ValueError("collect requested depth frames, but env.camera_type does not subscribe depth")

    def wait_until_ready(self) -> None:
        last_report = 0.0
        while True:
            status = self.env.get_robot_status()
            missing = []
            if status.get("left") is None:
                missing.append("left_arm_status")
            if status.get("right") is None:
                missing.append("right_arm_status")
            if self.include_camera:
                missing.extend(_camera_ready(self.env, self.camera_names, self.camera_type))
            if not missing:
                return
            now = time.time()
            if now - last_report > 1.0:
                print(f"Waiting for streams: {', '.join(missing)}")
                last_report = now
            time.sleep(0.2)

    def prepare(self) -> None:
        success, error_message = self.env.set_special_mode(3, side="both")
        if not success:
            raise RuntimeError(error_message or "Failed to enable gravity mode for both arms")
        time.sleep(0.2)

    def capture_frame(self, frame_idx: int) -> tuple[Optional[EpisodeFrame], Optional[str]]:
        frame_timestamp = time.time()
        camera_frames, status = _capture_camera_and_status(
            self.env,
            include_camera=self.include_camera,
            target_size=self.img_size,
        )
        left_status = status.get("left") if isinstance(status, dict) else None
        right_status = status.get("right") if isinstance(status, dict) else None
        if left_status is None or right_status is None:
            return None, "arm status not ready"

        color_frames, depth_frames = _extract_camera_frames(
            camera_frames,
            camera_map=default_camera_map(self.camera_names),
            include_depth=self.camera_type == "all",
        )
        expected_cameras = set(default_camera_map(self.camera_names).values())
        if self.include_camera and set(color_frames.keys()) != expected_cameras:
            return None, "camera color frames not ready"
        if self.include_camera and self.camera_type == "all" and set(depth_frames.keys()) != expected_cameras:
            return None, "camera depth frames not ready"

        left_qpos = _arm_qpos(left_status)
        right_qpos = _arm_qpos(right_status)
        left_qvel = _arm_qvel(left_status)
        right_qvel = _arm_qvel(right_status)
        left_effort = _arm_effort(left_status)
        right_effort = _arm_effort(right_status)
        left_eef = _arm_eef(left_status)
        right_eef = _arm_eef(right_status)

        qpos = np.concatenate([left_qpos, right_qpos], axis=0)
        qvel = np.concatenate([left_qvel, right_qvel], axis=0)
        effort = np.concatenate([left_effort, right_effort], axis=0)
        eef = np.concatenate([left_eef, right_eef], axis=0)
        action = qpos.copy() if self.action_kind == "joint" else eef.copy()

        return EpisodeFrame(
            frame_idx=int(frame_idx),
            timestamp=float(frame_timestamp),
            qpos=qpos,
            qvel=qvel,
            effort=effort,
            eef=eef,
            action=action,
            images=color_frames if self.include_camera else {},
            images_depth=depth_frames if self.include_camera else {},
            topic_stamps=_build_topic_stamps(
                self.env,
                left_status=left_status,
                right_status=right_status,
            ),
        ), None

    def close(self) -> None:
        pass


class SingleArmMirrorCollector:
    def __init__(
        self,
        env: ARXRobotEnv,
        leader_side: Literal["left", "right"],
        camera_names: Iterable[str] = (),
        include_camera: bool = False,
        use_depth: bool = False,
        action_kind: Literal["joint", "eef"] = "joint",
        mirror: bool = True,
        img_size: tuple[int, int] = (640, 480),
        control_rate: float = 80.0,
        joint_lowpass_alpha: float = 0.2,
        joint_deadband: float | Iterable[float] = 0.004,
        eef_lowpass_alpha: float = 0.25,
        eef_deadband: float | Iterable[float] = (
            0.001,
            0.001,
            0.001,
            0.01,
            0.01,
            0.01,
            0.02,
        ),
    ):
        self.env = env
        self.leader_side = leader_side
        self.follow_side: Literal["left", "right"] = "right" if leader_side == "left" else "left"
        self.include_camera = bool(include_camera)
        self.action_kind = action_kind
        self.mirror = bool(mirror)
        self.record_side: Literal["left", "right"] = self.follow_side if self.mirror else self.leader_side
        self.camera_names = [str(name) for name in camera_names] if self.include_camera else []
        self.camera_type = "all" if use_depth else "color"
        self.img_size = tuple(img_size)
        self.control_rate = float(control_rate)
        self.joint_lowpass_alpha = float(joint_lowpass_alpha)
        self.joint_deadband = joint_deadband
        self.eef_lowpass_alpha = float(eef_lowpass_alpha)
        self.eef_deadband = eef_deadband
        self._last_joint_target: Optional[np.ndarray] = None
        self._last_eef_target: Optional[np.ndarray] = None
        self._latest_command: Optional[np.ndarray] = None
        self._control_lock = threading.Lock()
        self._control_thread: Optional[threading.Thread] = None
        self._control_stop = threading.Event()
        if not 0.0 < self.joint_lowpass_alpha <= 1.0:
            raise ValueError("joint_lowpass_alpha must be in (0, 1]")
        if not 0.0 < self.eef_lowpass_alpha <= 1.0:
            raise ValueError("eef_lowpass_alpha must be in (0, 1]")
        if self.mirror and self.control_rate <= 0.0:
            raise ValueError("control_rate must be > 0 when mirror=True")
        if self.include_camera:
            env_cameras = set(str(name) for name in getattr(self.env, "camera_view", ()))
            missing_cameras = [name for name in self.camera_names if name not in env_cameras]
            if missing_cameras:
                raise ValueError(
                    f"env.camera_view does not include requested cameras: {missing_cameras}"
                )
            if use_depth and getattr(self.env, "camera_type", "color") not in {"all", "depth"}:
                raise ValueError("collect requested depth frames, but env.camera_type does not subscribe depth")

    def wait_until_ready(self) -> None:
        last_report = 0.0
        while True:
            status = self.env.get_robot_status()
            missing = []
            if status.get("left") is None:
                missing.append("left_arm_status")
            if status.get("right") is None:
                missing.append("right_arm_status")
            if self.include_camera:
                missing.extend(_camera_ready(self.env, self.camera_names, self.camera_type))
            if not missing:
                return
            now = time.time()
            if now - last_report > 1.0:
                print(f"Waiting for streams: {', '.join(missing)}")
                last_report = now
            time.sleep(0.2)

    def prepare(self) -> None:
        _set_gravity_mode(self.env, self.leader_side)
        self._last_joint_target = None
        self._last_eef_target = None
        with self._control_lock:
            self._latest_command = None
        time.sleep(0.2)
        self.start_control()

    def start_control(self) -> None:
        if not self.mirror:
            return
        self.stop_control()
        self._control_stop.clear()
        self._control_once()
        self._control_thread = threading.Thread(target=self._control_loop, daemon=True)
        self._control_thread.start()

    def stop_control(self) -> None:
        self._control_stop.set()
        if self._control_thread is not None:
            self._control_thread.join(timeout=1.0)
            self._control_thread = None

    def _command_snapshot(self) -> Optional[np.ndarray]:
        with self._control_lock:
            if self._latest_command is None:
                return None
            return self._latest_command.copy()

    def _control_once(self) -> bool:
        status = self.env.get_robot_status()
        leader_status = status.get(self.leader_side) if isinstance(status, dict) else None
        follower_status = status.get(self.follow_side) if isinstance(status, dict) else None
        if leader_status is None or follower_status is None:
            return False
        if self.action_kind == "joint":
            target = _smooth_target(
                _arm_qpos(leader_status),
                self._last_joint_target,
                alpha=self.joint_lowpass_alpha,
                deadband=self.joint_deadband,
            )
            self._last_joint_target = target.copy()
            self.env.step_raw_joint({self.follow_side: target})
        else:
            target = _smooth_target(
                _arm_eef(leader_status),
                self._last_eef_target,
                alpha=self.eef_lowpass_alpha,
                deadband=self.eef_deadband,
            )
            self._last_eef_target = target.copy()
            self.env.step_raw_eef({self.follow_side: target})
        with self._control_lock:
            self._latest_command = target.copy()
        return True

    def _control_loop(self) -> None:
        period = 1.0 / max(self.control_rate, 1e-6)
        last_error_report = 0.0
        while not self._control_stop.is_set():
            loop_start = time.perf_counter()
            try:
                ok = self._control_once()
                if not ok:
                    now = time.time()
                    if now - last_error_report > 1.0:
                        print("Mirror control waiting for arm status.")
                        last_error_report = now
            except Exception as exc:
                now = time.time()
                if now - last_error_report > 1.0:
                    print(f"Mirror control error: {exc}")
                    last_error_report = now
            sleep_need = period - (time.perf_counter() - loop_start)
            if sleep_need > 0.0:
                time.sleep(sleep_need)

    def capture_frame(self, frame_idx: int) -> tuple[Optional[EpisodeFrame], Optional[str]]:
        frame_timestamp = time.time()
        camera_frames, status = _capture_camera_and_status(
            self.env,
            include_camera=self.include_camera,
            target_size=self.img_size,
        )
        leader_status = status.get(self.leader_side) if isinstance(status, dict) else None
        follower_status = status.get(self.follow_side) if isinstance(status, dict) else None
        if leader_status is None or follower_status is None:
            return None, "arm status not ready"
        record_status = follower_status if self.record_side == self.follow_side else leader_status

        color_frames, depth_frames = _extract_camera_frames(
            camera_frames,
            camera_map=default_camera_map(self.camera_names),
            include_depth=self.camera_type == "all",
        )
        expected_cameras = set(default_camera_map(self.camera_names).values())
        if self.include_camera and set(color_frames.keys()) != expected_cameras:
            return None, "camera color frames not ready"
        if self.include_camera and self.camera_type == "all" and set(depth_frames.keys()) != expected_cameras:
            return None, "camera depth frames not ready"

        qpos = _arm_qpos(record_status)
        qvel = _arm_qvel(record_status)
        effort = _arm_effort(record_status)
        eef = _arm_eef(record_status)
        action = qpos.copy() if self.action_kind == "joint" else eef.copy()

        if self.mirror:
            latest_command = self._command_snapshot()
            if latest_command is not None:
                action = latest_command

        return EpisodeFrame(
            frame_idx=int(frame_idx),
            timestamp=float(frame_timestamp),
            qpos=qpos,
            qvel=qvel,
            effort=effort,
            eef=eef,
            action=action,
            images=color_frames if self.include_camera else {},
            images_depth=depth_frames if self.include_camera else {},
            topic_stamps=_build_topic_stamps(
                self.env,
                left_status=status.get("left"),
                right_status=status.get("right"),
            ),
        ), None

    def close(self) -> None:
        self.stop_control()


class SingleArmSpaceMouseCollector:
    def __init__(
        self,
        env: ARXRobotEnv,
        side: Literal["left", "right"],
        camera_names: Iterable[str] = (),
        include_camera: bool = False,
        use_depth: bool = False,
        img_size: tuple[int, int] = (640, 480),
        control_rate: float = 60.0,
        translation_speed: float = 0.10,
        rotation_speed: float = 0.60,
        gripper_speed: float = 2.5,
        translation_deadzone: float = 0.05,
        rotation_deadzone: float = 0.05,
        response_exponent: float = 1.5,
        translation_axis_signs: Iterable[float] = (1.0, 1.0, 1.0),
        rotation_axis_signs: Iterable[float] = (1.0, 1.0, 1.0),
    ):
        if side not in {"left", "right"}:
            raise ValueError("side must be 'left' or 'right'")
        if control_rate <= 0.0:
            raise ValueError("control_rate must be > 0")
        self.env = env
        self.side: Literal["left", "right"] = side
        self.include_camera = bool(include_camera)
        self.camera_names = [str(name) for name in camera_names] if self.include_camera else []
        self.camera_type = "all" if use_depth else "color"
        self.img_size = tuple(img_size)
        self.control_rate = float(control_rate)
        self.translation_speed = float(translation_speed)
        self.rotation_speed = float(rotation_speed)
        self.gripper_speed = float(gripper_speed)
        self.translation_deadzone = float(translation_deadzone)
        self.rotation_deadzone = float(rotation_deadzone)
        self.response_exponent = float(response_exponent)
        self.translation_axis_signs = _normalize_axis_signs(translation_axis_signs, 3, "translation_axis_signs")
        self.rotation_axis_signs = _normalize_axis_signs(rotation_axis_signs, 3, "rotation_axis_signs")
        self.spacemouse = SpaceMouseDevice()
        self._latest_command: Optional[np.ndarray] = None
        self._control_lock = threading.Lock()
        self._control_thread: Optional[threading.Thread] = None
        self._control_stop = threading.Event()
        if self.include_camera:
            env_cameras = set(str(name) for name in getattr(self.env, "camera_view", ()))
            missing_cameras = [name for name in self.camera_names if name not in env_cameras]
            if missing_cameras:
                raise ValueError(f"env.camera_view does not include requested cameras: {missing_cameras}")
            if use_depth and getattr(self.env, "camera_type", "color") not in {"all", "depth"}:
                raise ValueError("collect requested depth frames, but env.camera_type does not subscribe depth")

    def wait_until_ready(self) -> None:
        last_report = 0.0
        while True:
            status = self.env.get_robot_status()
            missing = []
            if status.get(self.side) is None:
                missing.append(f"{self.side}_arm_status")
            if self.include_camera:
                missing.extend(_camera_ready(self.env, self.camera_names, self.camera_type))
            if not missing:
                return
            now = time.time()
            if now - last_report > 1.0:
                print(f"Waiting for streams: {', '.join(missing)}")
                last_report = now
            time.sleep(0.2)

    def prepare(self) -> None:
        with self._control_lock:
            self._latest_command = None
        time.sleep(0.1)
        self.start_control()

    def start_control(self) -> None:
        self.stop_control()
        self._control_stop.clear()
        self._control_thread = threading.Thread(target=self._control_loop, daemon=True)
        self._control_thread.start()

    def stop_control(self) -> None:
        self._control_stop.set()
        if self._control_thread is not None:
            self._control_thread.join(timeout=1.0)
            self._control_thread = None

    def _command_snapshot(self) -> Optional[np.ndarray]:
        with self._control_lock:
            if self._latest_command is None:
                return None
            return self._latest_command.copy()

    def _motion_from_sample(self, sample: SpaceMouseSample, dt: float) -> tuple[np.ndarray, np.ndarray, float]:
        translation = _apply_axis_deadzone(
            sample.translation * self.translation_axis_signs,
            self.translation_deadzone,
            self.response_exponent,
        ) * (self.translation_speed * float(dt))
        rotation = _apply_axis_deadzone(
            sample.rotation * self.rotation_axis_signs,
            self.rotation_deadzone,
            self.response_exponent,
        ) * (self.rotation_speed * float(dt))
        buttons = tuple(bool(v) for v in sample.buttons)
        close_pressed = len(buttons) > 0 and buttons[0]
        open_pressed = len(buttons) > 1 and buttons[1]
        if close_pressed and not open_pressed:
            gripper_delta = self.gripper_speed * float(dt)
        elif open_pressed and not close_pressed:
            gripper_delta = -self.gripper_speed * float(dt)
        else:
            gripper_delta = 0.0
        return translation.astype(np.float32), rotation.astype(np.float32), float(gripper_delta)

    def _control_once(self, dt: float) -> bool:
        status = self.env.get_robot_status()
        arm_status = status.get(self.side) if isinstance(status, dict) else None
        if arm_status is None:
            return False
        sample = self.spacemouse.read()
        delta_xyz, delta_rpy, gripper_delta = self._motion_from_sample(sample, dt)
        if not _effective_motion(delta_xyz, delta_rpy, gripper_delta):
            return True
        target = _compose_eef_target(_arm_eef(arm_status), delta_xyz, delta_rpy, gripper_delta)
        self.env.step_raw_eef({self.side: target})
        with self._control_lock:
            self._latest_command = target.copy()
        return True

    def _control_loop(self) -> None:
        period = 1.0 / max(self.control_rate, 1e-6)
        last_tick = time.perf_counter()
        last_error_report = 0.0
        while not self._control_stop.is_set():
            loop_start = time.perf_counter()
            now = loop_start
            dt = min(max(now - last_tick, period), 0.2)
            last_tick = now
            try:
                ok = self._control_once(dt)
                if not ok:
                    report_t = time.time()
                    if report_t - last_error_report > 1.0:
                        print(f"SpaceMouse control waiting for {self.side} arm status.")
                        last_error_report = report_t
            except Exception as exc:
                report_t = time.time()
                if report_t - last_error_report > 1.0:
                    print(f"SpaceMouse control error: {exc}")
                    last_error_report = report_t
            sleep_need = period - (time.perf_counter() - loop_start)
            if sleep_need > 0.0:
                time.sleep(sleep_need)

    def capture_frame(self, frame_idx: int) -> tuple[Optional[EpisodeFrame], Optional[str]]:
        frame_timestamp = time.time()
        camera_frames, status = _capture_camera_and_status(
            self.env,
            include_camera=self.include_camera,
            target_size=self.img_size,
        )
        arm_status = status.get(self.side) if isinstance(status, dict) else None
        if arm_status is None:
            return None, f"{self.side} arm status not ready"

        color_frames, depth_frames = _extract_camera_frames(
            camera_frames,
            camera_map=default_camera_map(self.camera_names),
            include_depth=self.camera_type == "all",
        )
        expected_cameras = set(default_camera_map(self.camera_names).values())
        if self.include_camera and set(color_frames.keys()) != expected_cameras:
            return None, "camera color frames not ready"
        if self.include_camera and self.camera_type == "all" and set(depth_frames.keys()) != expected_cameras:
            return None, "camera depth frames not ready"

        qpos = _arm_qpos(arm_status)
        qvel = _arm_qvel(arm_status)
        effort = _arm_effort(arm_status)
        eef = _arm_eef(arm_status)
        action = self._command_snapshot()
        if action is None:
            action = eef.copy()

        return EpisodeFrame(
            frame_idx=int(frame_idx),
            timestamp=float(frame_timestamp),
            qpos=qpos,
            qvel=qvel,
            effort=effort,
            eef=eef,
            action=np.asarray(action, dtype=np.float32),
            images=color_frames if self.include_camera else {},
            images_depth=depth_frames if self.include_camera else {},
            topic_stamps=_build_topic_stamps(
                self.env,
                left_status=status.get("left") if isinstance(status, dict) else None,
                right_status=status.get("right") if isinstance(status, dict) else None,
            ),
        ), None

    def close(self) -> None:
        self.stop_control()
        self.spacemouse.close()


class DualArmSpaceMouseCollector:
    def __init__(
        self,
        env: ARXRobotEnv,
        camera_names: Iterable[str] = (),
        include_camera: bool = False,
        use_depth: bool = False,
        img_size: tuple[int, int] = (640, 480),
        control_rate: float = 60.0,
        translation_speed: float = 0.10,
        rotation_speed: float = 0.60,
        gripper_speed: float = 2.5,
        translation_deadzone: float = 0.05,
        rotation_deadzone: float = 0.05,
        response_exponent: float = 1.5,
        translation_axis_signs: Iterable[float] = (1.0, 1.0, 1.0),
        rotation_axis_signs: Iterable[float] = (1.0, 1.0, 1.0),
        initial_active_side: Literal["left", "right"] = "left",
    ):
        if control_rate <= 0.0:
            raise ValueError("control_rate must be > 0")
        if initial_active_side not in {"left", "right"}:
            raise ValueError("initial_active_side must be 'left' or 'right'")
        self.env = env
        self.include_camera = bool(include_camera)
        self.camera_names = [str(name) for name in camera_names] if self.include_camera else []
        self.camera_type = "all" if use_depth else "color"
        self.img_size = tuple(img_size)
        self.control_rate = float(control_rate)
        self.translation_speed = float(translation_speed)
        self.rotation_speed = float(rotation_speed)
        self.gripper_speed = float(gripper_speed)
        self.translation_deadzone = float(translation_deadzone)
        self.rotation_deadzone = float(rotation_deadzone)
        self.response_exponent = float(response_exponent)
        self.translation_axis_signs = _normalize_axis_signs(translation_axis_signs, 3, "translation_axis_signs")
        self.rotation_axis_signs = _normalize_axis_signs(rotation_axis_signs, 3, "rotation_axis_signs")
        self.initial_active_side: Literal["left", "right"] = initial_active_side
        self._active_side: Literal["left", "right"] = initial_active_side
        self._last_buttons: tuple[bool, bool] = (False, False)
        self.spacemouse = SpaceMouseDevice()
        self._latest_commands: Dict[str, Optional[np.ndarray]] = {"left": None, "right": None}
        self._control_lock = threading.Lock()
        self._control_thread: Optional[threading.Thread] = None
        self._control_stop = threading.Event()
        if self.include_camera:
            env_cameras = set(str(name) for name in getattr(self.env, "camera_view", ()))
            missing_cameras = [name for name in self.camera_names if name not in env_cameras]
            if missing_cameras:
                raise ValueError(f"env.camera_view does not include requested cameras: {missing_cameras}")
            if use_depth and getattr(self.env, "camera_type", "color") not in {"all", "depth"}:
                raise ValueError("collect requested depth frames, but env.camera_type does not subscribe depth")

    @property
    def active_side(self) -> Literal["left", "right"]:
        return self._active_side

    def wait_until_ready(self) -> None:
        last_report = 0.0
        while True:
            status = self.env.get_robot_status()
            missing = []
            if status.get("left") is None:
                missing.append("left_arm_status")
            if status.get("right") is None:
                missing.append("right_arm_status")
            if self.include_camera:
                missing.extend(_camera_ready(self.env, self.camera_names, self.camera_type))
            if not missing:
                return
            now = time.time()
            if now - last_report > 1.0:
                print(f"Waiting for streams: {', '.join(missing)}")
                last_report = now
            time.sleep(0.2)

    def prepare(self) -> None:
        self._active_side = self.initial_active_side
        self._last_buttons = (False, False)
        with self._control_lock:
            self._latest_commands = {"left": None, "right": None}
        print(
            "Dual 3D mouse mapping: button0=close gripper, button1=open gripper, "
            "press both buttons together to switch active arm."
        )
        print(f"Active arm: {self._active_side}")
        time.sleep(0.1)
        self.start_control()

    def start_control(self) -> None:
        self.stop_control()
        self._control_stop.clear()
        self._control_thread = threading.Thread(target=self._control_loop, daemon=True)
        self._control_thread.start()

    def stop_control(self) -> None:
        self._control_stop.set()
        if self._control_thread is not None:
            self._control_thread.join(timeout=1.0)
            self._control_thread = None

    def _command_snapshot(self) -> Dict[str, Optional[np.ndarray]]:
        with self._control_lock:
            return {
                "left": None if self._latest_commands["left"] is None else self._latest_commands["left"].copy(),
                "right": None if self._latest_commands["right"] is None else self._latest_commands["right"].copy(),
            }

    def _motion_from_sample(self, sample: SpaceMouseSample, dt: float) -> tuple[np.ndarray, np.ndarray, float]:
        translation = _apply_axis_deadzone(
            sample.translation * self.translation_axis_signs,
            self.translation_deadzone,
            self.response_exponent,
        ) * (self.translation_speed * float(dt))
        rotation = _apply_axis_deadzone(
            sample.rotation * self.rotation_axis_signs,
            self.rotation_deadzone,
            self.response_exponent,
        ) * (self.rotation_speed * float(dt))
        buttons = tuple(bool(v) for v in sample.buttons)
        close_pressed = len(buttons) > 0 and buttons[0]
        open_pressed = len(buttons) > 1 and buttons[1]
        both_pressed = close_pressed and open_pressed
        if both_pressed and not (self._last_buttons[0] and self._last_buttons[1]):
            self._active_side = "right" if self._active_side == "left" else "left"
            print(f"Active arm switched to {self._active_side}")
        self._last_buttons = (close_pressed, open_pressed)
        if both_pressed:
            gripper_delta = 0.0
        elif close_pressed:
            gripper_delta = self.gripper_speed * float(dt)
        elif open_pressed:
            gripper_delta = -self.gripper_speed * float(dt)
        else:
            gripper_delta = 0.0
        return translation.astype(np.float32), rotation.astype(np.float32), float(gripper_delta)

    def _control_once(self, dt: float) -> bool:
        status = self.env.get_robot_status()
        left_status = status.get("left") if isinstance(status, dict) else None
        right_status = status.get("right") if isinstance(status, dict) else None
        if left_status is None or right_status is None:
            return False

        current_eef = {
            "left": _arm_eef(left_status),
            "right": _arm_eef(right_status),
        }
        with self._control_lock:
            for side in ("left", "right"):
                if self._latest_commands[side] is None:
                    self._latest_commands[side] = current_eef[side].copy()

        sample = self.spacemouse.read()
        delta_xyz, delta_rpy, gripper_delta = self._motion_from_sample(sample, dt)
        if not _effective_motion(delta_xyz, delta_rpy, gripper_delta):
            return True

        active_side = self._active_side
        target = _compose_eef_target(current_eef[active_side], delta_xyz, delta_rpy, gripper_delta)
        self.env.step_raw_eef({active_side: target})
        with self._control_lock:
            self._latest_commands[active_side] = target.copy()
        return True

    def _control_loop(self) -> None:
        period = 1.0 / max(self.control_rate, 1e-6)
        last_tick = time.perf_counter()
        last_error_report = 0.0
        while not self._control_stop.is_set():
            loop_start = time.perf_counter()
            now = loop_start
            dt = min(max(now - last_tick, period), 0.2)
            last_tick = now
            try:
                ok = self._control_once(dt)
                if not ok:
                    report_t = time.time()
                    if report_t - last_error_report > 1.0:
                        print("Dual 3D mouse control waiting for arm status.")
                        last_error_report = report_t
            except Exception as exc:
                report_t = time.time()
                if report_t - last_error_report > 1.0:
                    print(f"Dual 3D mouse control error: {exc}")
                    last_error_report = report_t
            sleep_need = period - (time.perf_counter() - loop_start)
            if sleep_need > 0.0:
                time.sleep(sleep_need)

    def capture_frame(self, frame_idx: int) -> tuple[Optional[EpisodeFrame], Optional[str]]:
        frame_timestamp = time.time()
        camera_frames, status = _capture_camera_and_status(
            self.env,
            include_camera=self.include_camera,
            target_size=self.img_size,
        )
        left_status = status.get("left") if isinstance(status, dict) else None
        right_status = status.get("right") if isinstance(status, dict) else None
        if left_status is None or right_status is None:
            return None, "arm status not ready"

        color_frames, depth_frames = _extract_camera_frames(
            camera_frames,
            camera_map=default_camera_map(self.camera_names),
            include_depth=self.camera_type == "all",
        )
        expected_cameras = set(default_camera_map(self.camera_names).values())
        if self.include_camera and set(color_frames.keys()) != expected_cameras:
            return None, "camera color frames not ready"
        if self.include_camera and self.camera_type == "all" and set(depth_frames.keys()) != expected_cameras:
            return None, "camera depth frames not ready"

        left_qpos = _arm_qpos(left_status)
        right_qpos = _arm_qpos(right_status)
        left_qvel = _arm_qvel(left_status)
        right_qvel = _arm_qvel(right_status)
        left_effort = _arm_effort(left_status)
        right_effort = _arm_effort(right_status)
        left_eef = _arm_eef(left_status)
        right_eef = _arm_eef(right_status)
        command_snapshot = self._command_snapshot()
        left_action = command_snapshot["left"] if command_snapshot["left"] is not None else left_eef.copy()
        right_action = command_snapshot["right"] if command_snapshot["right"] is not None else right_eef.copy()

        return EpisodeFrame(
            frame_idx=int(frame_idx),
            timestamp=float(frame_timestamp),
            qpos=np.concatenate([left_qpos, right_qpos], axis=0),
            qvel=np.concatenate([left_qvel, right_qvel], axis=0),
            effort=np.concatenate([left_effort, right_effort], axis=0),
            eef=np.concatenate([left_eef, right_eef], axis=0),
            action=np.concatenate([left_action, right_action], axis=0).astype(np.float32),
            images=color_frames if self.include_camera else {},
            images_depth=depth_frames if self.include_camera else {},
            topic_stamps=_build_topic_stamps(
                self.env,
                left_status=left_status,
                right_status=right_status,
            ),
        ), None

    def close(self) -> None:
        self.stop_control()
        self.spacemouse.close()
