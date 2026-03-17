from __future__ import annotations

import importlib
import json
import shutil
from dataclasses import dataclass
from pathlib import Path
import sys
from typing import Any, Optional

import numpy as np

ROOT_DIR = Path(__file__).resolve().parent.parent


@dataclass(frozen=True)
class EpisodeFrame:
    frame_idx: int
    timestamp: float
    qpos: np.ndarray
    qvel: np.ndarray
    effort: np.ndarray
    eef: np.ndarray
    action: np.ndarray
    images: dict[str, np.ndarray]
    images_depth: dict[str, np.ndarray]
    robot_base: Optional[np.ndarray] = None
    base_wheels: Optional[np.ndarray] = None
    base_velocity: Optional[np.ndarray] = None
    action_base: Optional[np.ndarray] = None
    topic_stamps: dict[str, Optional[float]] | None = None


@dataclass
class EpisodeBuffer:
    episode_idx: int
    mode: str
    frame_rate: float
    action_kind: str
    include_camera: bool
    include_base: bool
    camera_map: dict[str, str]
    config: dict[str, Any]
    side: Optional[str] = None
    created_at: str = ""
    frames: list[EpisodeFrame] = None

    def __post_init__(self) -> None:
        if self.frames is None:
            self.frames = []

    @property
    def frame_count(self) -> int:
        return len(self.frames)

    def add_frame(self, frame: EpisodeFrame) -> None:
        self.frames.append(frame)


def load_episode(episode_dir: Path | str) -> EpisodeBuffer:
    episode_dir = Path(episode_dir)
    manifest = json.loads(
        (episode_dir / "episode.json").read_text(encoding="utf-8"))
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
        action_kind=str(manifest["action_kind"]),
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
                images_depth={key: value[idx]
                              for key, value in images_depth.items()},
                robot_base=np.asarray(
                    low_dim["robot_base"][idx], dtype=np.float32) if "robot_base" in low_dim else None,
                base_wheels=np.asarray(
                    low_dim["base_wheels"][idx], dtype=np.float32) if "base_wheels" in low_dim else None,
                base_velocity=np.asarray(
                    low_dim["base_velocity"][idx], dtype=np.float32) if "base_velocity" in low_dim else None,
                action_base=np.asarray(
                    low_dim["action_base"][idx], dtype=np.float32) if "action_base" in low_dim else None,
                topic_stamps=dict(frame_meta.get("topic_stamps", {})),
            )
        )
    return episode


@dataclass(frozen=True)
class SourceSpec:
    mode: str
    dim: int
    action_kind: str
    include_camera: bool
    include_base: bool
    camera_keys: tuple[str, ...]
    depth_keys: tuple[str, ...]
    frame_rate: float


def _episode_dirs(episodes_root: Path | str) -> list[Path]:
    root = Path(episodes_root)
    if not root.exists():
        raise FileNotFoundError(f"Episodes root does not exist: {root}")
    return sorted(
        child for child in root.iterdir()
        if child.is_dir() and child.name.startswith("episode_")
    )


def _mode_prefixes(mode: str) -> list[str]:
    if mode == "single":
        return [""]
    if mode == "dual":
        return ["left_", "right_"]
    raise ValueError(f"Unsupported mode: {mode}")


def _joint_names(mode: str) -> list[str]:
    names = []
    for prefix in _mode_prefixes(mode):
        names.extend([f"{prefix}joint_{idx}" for idx in range(6)])
        names.append(f"{prefix}gripper")
    return names


def _eef_names(mode: str) -> list[str]:
    axes = ["x", "y", "z", "roll", "pitch", "yaw", "gripper"]
    names = []
    for prefix in _mode_prefixes(mode):
        names.extend([f"{prefix}{axis}" for axis in axes])
    return names


def _vector_feature(shape: tuple[int, ...], names: list[str]) -> dict[str, Any]:
    return {
        "dtype": "float32",
        "shape": shape,
        "names": names,
    }


def _image_feature(image: np.ndarray, dtype: str = "image") -> dict[str, Any]:
    arr = np.asarray(image)
    if arr.ndim == 2:
        shape = (int(arr.shape[0]), int(arr.shape[1]), 1)
    elif arr.ndim == 3:
        shape = tuple(int(v) for v in arr.shape)
    else:
        raise ValueError(f"Unsupported image shape: {arr.shape}")
    return {
        "dtype": dtype,
        "shape": shape,
        "names": ["height", "width", "channel"],
    }


def _normalize_image(image: np.ndarray) -> np.ndarray:
    arr = np.asarray(image)
    if arr.ndim == 2:
        return arr[..., None]
    return arr


def _episode_dim(episode: EpisodeBuffer) -> int:
    if not episode.frames:
        raise RuntimeError(f"Episode {episode.episode_idx} has no frames.")
    return int(np.asarray(episode.frames[0].action, dtype=np.float32).reshape(-1).shape[0])


def _infer_source_spec(episode: EpisodeBuffer) -> SourceSpec:
    first = episode.frames[0]
    return SourceSpec(
        mode=str(episode.mode),
        dim=_episode_dim(episode),
        action_kind=str(episode.action_kind),
        include_camera=bool(episode.include_camera),
        include_base=bool(episode.include_base),
        camera_keys=tuple(sorted(first.images.keys())),
        depth_keys=tuple(sorted(first.images_depth.keys())),
        frame_rate=float(episode.frame_rate),
    )


def _validate_episode_against_spec(episode: EpisodeBuffer, spec: SourceSpec, path: Path) -> None:
    current = _infer_source_spec(episode)
    if current.mode != spec.mode:
        raise ValueError(
            f"Mixed modes are not supported: {path} has {current.mode}, expected {spec.mode}")
    if current.dim != spec.dim:
        raise ValueError(
            f"Mixed action dims are not supported: {path} has {current.dim}, expected {spec.dim}")
    if current.action_kind != spec.action_kind:
        raise ValueError(
            f"Mixed action kinds are not supported: {path} has {current.action_kind}, expected {spec.action_kind}"
        )
    if current.include_camera != spec.include_camera:
        raise ValueError(f"Mixed camera settings are not supported: {path}")
    if current.include_base != spec.include_base:
        raise ValueError(f"Mixed base settings are not supported: {path}")
    if current.camera_keys != spec.camera_keys:
        raise ValueError(f"Mixed camera keys are not supported: {path}")
    if current.depth_keys != spec.depth_keys:
        raise ValueError(f"Mixed depth camera keys are not supported: {path}")


def _default_task(episode: EpisodeBuffer) -> str:
    task = str(episode.config.get("task", "")).strip()
    if task:
        return task
    collection_kind = str(episode.config.get("collection_kind", "")).strip()
    if collection_kind:
        return collection_kind
    return f"{episode.mode}_{episode.action_kind}"


def _build_features(
    episode: EpisodeBuffer,
    spec: SourceSpec,
    include_depth_images: bool,
) -> dict[str, Any]:
    first = episode.frames[0]
    features: dict[str, Any] = {
        "observation.state": _vector_feature((spec.dim,), _joint_names(spec.mode)),
        "observation.qvel": _vector_feature((spec.dim,), _joint_names(spec.mode)),
        "observation.effort": _vector_feature((spec.dim,), _joint_names(spec.mode)),
        "observation.eef": _vector_feature((spec.dim,), _eef_names(spec.mode)),
        "action": _vector_feature((spec.dim,), _eef_names(spec.mode) if spec.action_kind == "eef" else _joint_names(spec.mode)),
    }
    if spec.include_base:
        features["observation.base_state"] = _vector_feature(
            (4,), ["x", "y", "yaw", "height"])
        features["observation.base_wheels"] = _vector_feature(
            (3,), ["wheel_0", "wheel_1", "wheel_2"])
        features["observation.base_velocity"] = _vector_feature(
            (3,), ["vx", "vy", "wz"])
        features["action.base"] = _vector_feature(
            (4,), ["vx", "vy", "wz", "height"])
    for camera_key in spec.camera_keys:
        features[f"observation.images.{camera_key}"] = _image_feature(
            first.images[camera_key], dtype="video")
    if include_depth_images:
        for camera_key in spec.depth_keys:
            features[f"observation.images_depth.{camera_key}"] = _image_feature(
                first.images_depth[camera_key], dtype="video")
    return features


def _build_frame_dict(
    episode: EpisodeBuffer,
    frame,
    include_depth_images: bool,
    task_text: str,
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "observation.state": np.asarray(frame.qpos, dtype=np.float32),
        "observation.qvel": np.asarray(frame.qvel, dtype=np.float32),
        "observation.effort": np.asarray(frame.effort, dtype=np.float32),
        "observation.eef": np.asarray(frame.eef, dtype=np.float32),
        "action": np.asarray(frame.action, dtype=np.float32),
        "task": task_text,
    }
    if episode.include_base:
        payload["observation.base_state"] = np.asarray(
            frame.robot_base, dtype=np.float32)
        payload["observation.base_wheels"] = np.asarray(
            frame.base_wheels, dtype=np.float32)
        payload["observation.base_velocity"] = np.asarray(
            frame.base_velocity, dtype=np.float32)
        payload["action.base"] = np.asarray(
            frame.action_base, dtype=np.float32)
    for camera_key, image in frame.images.items():
        payload[f"observation.images.{camera_key}"] = _normalize_image(image)
    if include_depth_images:
        for camera_key, image in frame.images_depth.items():
            payload[f"observation.images_depth.{camera_key}"] = _normalize_image(
                image)
    return payload


def _import_lerobot_dataset():
    importlib_paths = [
        [],
        [str((ROOT_DIR / "lerobot" / "src").resolve())],
        [
            path for path in sys.path
            if Path(path or ".").resolve() != ROOT_DIR.resolve()
        ],
    ]
    import_attempts = (
        "lerobot.datasets.lerobot_dataset",
        "lerobot.common.datasets.lerobot_dataset",
    )

    original_sys_path = list(sys.path)
    original_lerobot_module = sys.modules.pop("lerobot", None)
    try:
        for extra_paths in importlib_paths:
            sys.path = extra_paths + \
                [path for path in original_sys_path if path not in extra_paths]
            sys.modules.pop("lerobot", None)
            for module_name in import_attempts:
                try:
                    module = importlib.import_module(module_name)
                    return module.LeRobotDataset
                except ImportError:
                    continue
        raise ImportError(
            "LeRobot v3 export requires a current official `lerobot` package. "
            "Expected module paths like `lerobot.datasets.lerobot_dataset` were not importable."
        )
    finally:
        sys.path = original_sys_path
        if original_lerobot_module is not None:
            sys.modules["lerobot"] = original_lerobot_module


def _import_pyarrow():
    try:
        import pyarrow as pa
        import pyarrow.parquet as pq
    except ImportError as exc:  # pragma: no cover
        raise ImportError(
            "LeRobot v2.1 export requires `pyarrow` in the Python environment used to run this script."
        ) from exc
    return pa, pq


def _import_cv2():
    try:
        import cv2
    except ImportError as exc:  # pragma: no cover
        raise ImportError(
            "LeRobot v2.1 video export requires `opencv-python`/`cv2` in the Python environment "
            "used to run this script."
        ) from exc
    return cv2


def _jsonl_write(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False))
            stream.write("\n")


def _scalar_feature(dtype: str) -> dict[str, Any]:
    return {
        "dtype": dtype,
        "shape": [1],
        "names": None,
    }


def _list_feature(dim: int, names: list[str]) -> dict[str, Any]:
    return {
        "dtype": "float32",
        "shape": [int(dim)],
        "names": names,
    }


def _build_v21_features(
    episode: EpisodeBuffer,
    spec: SourceSpec,
    include_depth_images: bool,
) -> dict[str, Any]:
    first = episode.frames[0]
    features: dict[str, Any] = {
        "action": _list_feature(spec.dim, _eef_names(spec.mode) if spec.action_kind == "eef" else _joint_names(spec.mode)),
        "observation.state": _list_feature(spec.dim, _joint_names(spec.mode)),
        "observation.qvel": _list_feature(spec.dim, _joint_names(spec.mode)),
        "observation.effort": _list_feature(spec.dim, _joint_names(spec.mode)),
        "observation.eef": _list_feature(spec.dim, _eef_names(spec.mode)),
        "timestamp": _scalar_feature("float32"),
        "frame_index": _scalar_feature("int64"),
        "episode_index": _scalar_feature("int64"),
        "index": _scalar_feature("int64"),
        "task_index": _scalar_feature("int64"),
    }
    if spec.include_base:
        features["observation.base_state"] = _list_feature(
            4, ["x", "y", "yaw", "height"])
        features["observation.base_wheels"] = _list_feature(
            3, ["wheel_0", "wheel_1", "wheel_2"])
        features["observation.base_velocity"] = _list_feature(
            3, ["vx", "vy", "wz"])
        features["action.base"] = _list_feature(
            4, ["vx", "vy", "wz", "height"])
    image_dtype = "video"
    for camera_key in spec.camera_keys:
        feature = _image_feature(first.images[camera_key])
        feature["dtype"] = image_dtype
        features[f"observation.images.{camera_key}"] = feature
    if include_depth_images:
        for camera_key in spec.depth_keys:
            feature = _image_feature(first.images_depth[camera_key])
            feature["dtype"] = image_dtype
            features[f"observation.images_depth.{camera_key}"] = feature
    return features


def _compute_episode_stats(episode: EpisodeBuffer) -> dict[str, Any]:
    arrays: dict[str, np.ndarray] = {
        "action": np.stack([np.asarray(frame.action, dtype=np.float32) for frame in episode.frames], axis=0),
        "observation.state": np.stack([np.asarray(frame.qpos, dtype=np.float32) for frame in episode.frames], axis=0),
        "observation.qvel": np.stack([np.asarray(frame.qvel, dtype=np.float32) for frame in episode.frames], axis=0),
        "observation.effort": np.stack([np.asarray(frame.effort, dtype=np.float32) for frame in episode.frames], axis=0),
        "observation.eef": np.stack([np.asarray(frame.eef, dtype=np.float32) for frame in episode.frames], axis=0),
    }
    if episode.include_base:
        arrays["observation.base_state"] = np.stack(
            [np.asarray(frame.robot_base, dtype=np.float32) for frame in episode.frames], axis=0
        )
        arrays["observation.base_wheels"] = np.stack(
            [np.asarray(frame.base_wheels, dtype=np.float32) for frame in episode.frames], axis=0
        )
        arrays["observation.base_velocity"] = np.stack(
            [np.asarray(frame.base_velocity, dtype=np.float32) for frame in episode.frames], axis=0
        )
        arrays["action.base"] = np.stack(
            [np.asarray(frame.action_base, dtype=np.float32) for frame in episode.frames], axis=0
        )
    stats: dict[str, Any] = {}
    for key, array in arrays.items():
        stats[key] = {
            "min": np.min(array, axis=0).astype(np.float32).tolist(),
            "max": np.max(array, axis=0).astype(np.float32).tolist(),
            "mean": np.mean(array, axis=0).astype(np.float32).tolist(),
            "std": np.std(array, axis=0).astype(np.float32).tolist(),
        }
    return stats


def _remove_dir_tree(root: Path) -> None:
    shutil.rmtree(root, ignore_errors=True)


def _v21_video_keys(spec: SourceSpec, include_depth_images: bool) -> list[str]:
    keys = [
        f"observation.images.{camera_key}" for camera_key in spec.camera_keys]
    if include_depth_images:
        keys.extend(
            [f"observation.images_depth.{camera_key}" for camera_key in spec.depth_keys])
    return keys


def _write_video_frames(path: Path, frames: list[np.ndarray], fps: int) -> None:
    cv2 = _import_cv2()
    if not frames:
        raise ValueError(f"No frames to write for video: {path}")
    first = np.asarray(frames[0])
    if first.ndim == 2 or (first.ndim == 3 and first.shape[2] == 1):
        rgb_frames = [
            cv2.cvtColor(np.asarray(frame).squeeze(-1) if np.asarray(frame).ndim ==
                         3 else np.asarray(frame), cv2.COLOR_GRAY2BGR)
            for frame in frames
        ]
    else:
        rgb_frames = [cv2.cvtColor(np.asarray(
            frame), cv2.COLOR_RGB2BGR) for frame in frames]
    height, width = rgb_frames[0].shape[:2]
    path.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(
        str(path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        float(fps),
        (int(width), int(height)),
    )
    if not writer.isOpened():
        raise RuntimeError(f"Failed to open video writer: {path}")
    try:
        for frame in rgb_frames:
            if frame.shape[:2] != (height, width):
                raise ValueError(
                    f"Inconsistent frame shape in video export: {frame.shape}")
            if frame.dtype != np.uint8:
                frame = np.clip(frame, 0, 255).astype(np.uint8)
            writer.write(frame)
    finally:
        writer.release()


def _write_v21_parquet(
    episode: EpisodeBuffer,
    episode_index: int,
    task_index: int,
    global_frame_start: int,
    parquet_path: Path,
) -> int:
    pa, pq = _import_pyarrow()
    frame_count = episode.frame_count
    columns: dict[str, Any] = {
        "action": [np.asarray(frame.action, dtype=np.float32).tolist() for frame in episode.frames],
        "observation.state": [np.asarray(frame.qpos, dtype=np.float32).tolist() for frame in episode.frames],
        "observation.qvel": [np.asarray(frame.qvel, dtype=np.float32).tolist() for frame in episode.frames],
        "observation.effort": [np.asarray(frame.effort, dtype=np.float32).tolist() for frame in episode.frames],
        "observation.eef": [np.asarray(frame.eef, dtype=np.float32).tolist() for frame in episode.frames],
        "timestamp": [float(frame.timestamp) for frame in episode.frames],
        "frame_index": list(range(frame_count)),
        "episode_index": [int(episode_index)] * frame_count,
        "index": list(range(global_frame_start, global_frame_start + frame_count)),
        "task_index": [int(task_index)] * frame_count,
    }
    if episode.include_base:
        columns["observation.base_state"] = [
            np.asarray(frame.robot_base, dtype=np.float32).tolist() for frame in episode.frames
        ]
        columns["observation.base_wheels"] = [
            np.asarray(frame.base_wheels, dtype=np.float32).tolist() for frame in episode.frames
        ]
        columns["observation.base_velocity"] = [
            np.asarray(frame.base_velocity, dtype=np.float32).tolist() for frame in episode.frames
        ]
        columns["action.base"] = [
            np.asarray(frame.action_base, dtype=np.float32).tolist() for frame in episode.frames
        ]

    parquet_path.parent.mkdir(parents=True, exist_ok=True)
    table = pa.table(columns)
    pq.write_table(table, parquet_path)
    return frame_count


def _export_collect_to_lerobot_v21(
    episode_dirs: list[Path],
    output_root: Path,
    spec: SourceSpec,
    fps: int,
    robot_type: str,
    task_override: Optional[str],
    include_depth_images: bool,
) -> Path:
    first_episode = load_episode(episode_dirs[0])
    info_features = _build_v21_features(
        first_episode,
        spec,
        include_depth_images=include_depth_images,
    )

    output_root.parent.mkdir(parents=True, exist_ok=True)
    if output_root.exists() and any(output_root.iterdir()):
        raise FileExistsError(f"Output directory is not empty: {output_root}")
    output_root.mkdir(parents=True, exist_ok=True)

    data_root = output_root / "data" / "chunk-000"
    meta_root = output_root / "meta"
    videos_root = output_root / "videos" / "chunk-000"
    data_root.mkdir(parents=True, exist_ok=True)
    meta_root.mkdir(parents=True, exist_ok=True)
    if spec.include_camera:
        videos_root.mkdir(parents=True, exist_ok=True)

    task_to_index: dict[str, int] = {}
    episodes_rows: list[dict[str, Any]] = []
    episodes_stats_rows: list[dict[str, Any]] = []
    total_frames = 0
    total_videos = 0
    video_keys = _v21_video_keys(
        spec, include_depth_images=include_depth_images)

    for episode_dir in episode_dirs:
        episode = load_episode(episode_dir)
        _validate_episode_against_spec(episode, spec, episode_dir)
        task_text = str(task_override).strip(
        ) if task_override else _default_task(episode)
        if task_text not in task_to_index:
            task_to_index[task_text] = len(task_to_index)
        task_index = task_to_index[task_text]

        parquet_path = data_root / \
            f"episode_{int(episode.episode_idx):06d}.parquet"
        written_frames = _write_v21_parquet(
            episode,
            episode_index=int(episode.episode_idx),
            task_index=task_index,
            global_frame_start=total_frames,
            parquet_path=parquet_path,
        )
        total_frames += written_frames

        videos: dict[str, str] = {}
        if spec.include_camera:
            for camera_key in spec.camera_keys:
                video_key = f"observation.images.{camera_key}"
                video_rel = Path("videos") / "chunk-000" / video_key / \
                    f"episode_{int(episode.episode_idx):06d}.mp4"
                video_path = output_root / video_rel
                _write_video_frames(
                    video_path, [frame.images[camera_key] for frame in episode.frames], fps=fps)
                videos[video_key] = str(video_rel)
                total_videos += 1
            if include_depth_images:
                for camera_key in spec.depth_keys:
                    video_key = f"observation.images_depth.{camera_key}"
                    video_rel = Path("videos") / "chunk-000" / video_key / \
                        f"episode_{int(episode.episode_idx):06d}.mp4"
                    video_path = output_root / video_rel
                    _write_video_frames(video_path, [
                                        frame.images_depth[camera_key] for frame in episode.frames], fps=fps)
                    videos[video_key] = str(video_rel)
                    total_videos += 1

        episodes_rows.append(
            {
                "episode_index": int(episode.episode_idx),
                "tasks": [task_text],
                "length": int(episode.frame_count),
                "videos": videos,
            }
        )
        episodes_stats_rows.append(
            {
                "episode_index": int(episode.episode_idx),
                "stats": _compute_episode_stats(episode),
            }
        )

    tasks_rows = [
        {"task_index": int(task_index), "task": task}
        for task, task_index in sorted(task_to_index.items(), key=lambda item: item[1])
    ]

    info: dict[str, Any] = {
        "codebase_version": "v2.1",
        "robot_type": robot_type,
        "total_episodes": len(episode_dirs),
        "total_frames": total_frames,
        "total_tasks": len(task_to_index),
        "total_videos": total_videos,
        "total_chunks": 1,
        "chunks_size": len(episode_dirs),
        "fps": int(fps),
        "splits": {"train": f"0:{len(episode_dirs)}"},
        "data_path": "data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet",
        "features": info_features,
    }
    if spec.include_camera:
        info["video_path"] = "videos/chunk-{episode_chunk:03d}/{video_key}/episode_{episode_index:06d}.mp4"

    source_manifest = {
        "target_version": "v2.1",
        "robot_type": robot_type,
        "fps": int(fps),
        "mode": spec.mode,
        "dim": spec.dim,
        "action_kind": spec.action_kind,
        "include_camera": spec.include_camera,
        "include_base": spec.include_base,
        "camera_keys": list(spec.camera_keys),
        "depth_keys": list(spec.depth_keys),
        "use_videos": True,
        "source_episodes": [str(path) for path in episode_dirs],
        "video_keys": video_keys,
    }

    (meta_root / "info.json").write_text(json.dumps(info, indent=2), encoding="utf-8")
    _jsonl_write(meta_root / "tasks.jsonl", tasks_rows)
    _jsonl_write(meta_root / "episodes.jsonl", episodes_rows)
    _jsonl_write(meta_root / "episodes_stats.jsonl", episodes_stats_rows)
    (output_root / "arx_collect_source.json").write_text(
        json.dumps(source_manifest, indent=2),
        encoding="utf-8",
    )
    _remove_dir_tree(output_root / "images")
    return output_root


def convert_collect_to_lerobot(
    episodes_root: Path | str = "episodes_raw",
    output_root: Path | str = "lerobot_v3",
    repo_id: str = "local/arx_collect",
    target_version: str = "v3.0",
    fps: Optional[int] = None,
    robot_type: str = "arx",
    task_override: Optional[str] = None,
    include_depth_images: bool = False,
    max_episodes: int = 0,
) -> Path:
    version = str(target_version).strip().lower()
    if version not in {"v2.1", "v3", "v3.0"}:
        raise NotImplementedError(
            "target_version must be one of: 'v2.1', 'v3.0'")

    episode_dirs = _episode_dirs(episodes_root)
    if max_episodes > 0:
        episode_dirs = episode_dirs[: int(max_episodes)]
    if not episode_dirs:
        raise FileNotFoundError(f"No episodes found under {episodes_root}")

    first_episode = load_episode(episode_dirs[0])
    spec = _infer_source_spec(first_episode)
    if fps is None:
        export_fps = int(round(spec.frame_rate))
    else:
        export_fps = int(fps)

    if version == "v2.1":
        return _export_collect_to_lerobot_v21(
            episode_dirs=episode_dirs,
            output_root=Path(output_root),
            spec=spec,
            fps=export_fps,
            robot_type=robot_type,
            task_override=task_override,
            include_depth_images=include_depth_images,
        )

    features = _build_features(
        first_episode, spec, include_depth_images=include_depth_images)
    output_root = Path(output_root)
    output_root.parent.mkdir(parents=True, exist_ok=True)
    if output_root.exists() and any(output_root.iterdir()):
        raise FileExistsError(f"Output directory is not empty: {output_root}")

    LeRobotDataset = _import_lerobot_dataset()
    dataset = LeRobotDataset.create(
        repo_id=repo_id,
        root=output_root,
        robot_type=robot_type,
        fps=export_fps,
        features=features,
        use_videos=True,
    )

    source_manifest: dict[str, Any] = {
        "target_version": "v3.0",
        "repo_id": repo_id,
        "robot_type": robot_type,
        "fps": export_fps,
        "mode": spec.mode,
        "dim": spec.dim,
        "action_kind": spec.action_kind,
        "include_camera": spec.include_camera,
        "include_base": spec.include_base,
        "camera_keys": list(spec.camera_keys),
        "depth_keys": list(spec.depth_keys),
        "source_episodes": [],
    }

    for episode_dir in episode_dirs:
        episode = load_episode(episode_dir)
        _validate_episode_against_spec(episode, spec, episode_dir)
        task_text = str(task_override).strip(
        ) if task_override else _default_task(episode)
        for frame in episode.frames:
            dataset.add_frame(
                _build_frame_dict(
                    episode,
                    frame,
                    include_depth_images=include_depth_images,
                    task_text=task_text,
                )
            )
        dataset.save_episode()
        source_manifest["source_episodes"].append(
            {
                "episode_dir": str(episode_dir),
                "episode_idx": int(episode.episode_idx),
                "frame_count": int(episode.frame_count),
                "task": task_text,
                "side": episode.side,
            }
        )

    (output_root / "arx_collect_source.json").write_text(
        json.dumps(source_manifest, indent=2),
        encoding="utf-8",
    )
    _remove_dir_tree(output_root / "images")
    return output_root


def main() -> None:
    convert_collect_to_lerobot(
        episodes_root="episodes_raw/gravity_single",
        output_root="lerobot_v3/gravity_single",
        repo_id="tjudrllab/gravity_single",
        target_version="v3.0",
        fps=None,
        robot_type="arx",
        task_override=None,
        include_depth_images=False,
        max_episodes=0,
    )


if __name__ == "__main__":
    main()
