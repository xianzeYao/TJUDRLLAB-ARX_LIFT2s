#!/usr/bin/env python3
"""Shared helpers for ARX deployment scripts."""

from __future__ import annotations

import json
import os
import sys
import time
from collections import deque
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
import torch

if TYPE_CHECKING:
    from arx_ros2_env import ARXRobotEnv

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
LEROBOT_SRC_ROOT = REPO_ROOT / "lerobot" / "src"
if str(LEROBOT_SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(LEROBOT_SRC_ROOT))
ROS2_ROOT = REPO_ROOT / "ARX_Realenv" / "ROS2"
if str(ROS2_ROOT) not in sys.path:
    sys.path.insert(0, str(ROS2_ROOT))


CONFIG_FILENAME = "config.json"


def create_default_env(
    camera_keys: tuple[str, ...] = ("camera_h", "camera_r"),
    save_dir: str = "testdata",
    video: bool = True,
    video_name: str = "test",
    image_width: int = 640,
    image_height: int = 480,
) -> "ARXRobotEnv":
    """Create ARXRobotEnv using the requested deployment defaults."""
    from arx_ros2_env import ARXRobotEnv

    return ARXRobotEnv(
        duration_per_step=1.0 / 20.0,
        min_steps=20,
        max_v_xyz=0.25,
        max_a_xyz=0.20,
        max_v_rpy=0.3,
        max_a_rpy=1.00,
        camera_type="all",
        camera_view=tuple(camera_keys),
        dir=save_dir,
        video=video,
        video_fps=30.0,
        video_name=video_name,
        img_size=(image_width, image_height),
    )


def bgr_to_rgb_if_needed(image: np.ndarray) -> np.ndarray:
    image = np.asarray(image)
    if image.ndim == 3 and image.shape[2] == 3:
        return image[:, :, ::-1].copy()
    return image


def _to_chw_float_image(image: np.ndarray, expected_shape: tuple[int, ...]) -> torch.Tensor:
    image = np.asarray(image)
    if image.ndim != 3:
        raise RuntimeError(
            f"Expected RGB image with 3 dims, got shape {image.shape}")

    # Align to policy feature shape. LeRobot policy configs use channel-first visual shapes.
    if image.shape == expected_shape:
        chw = image
    elif image.shape[::-1] != expected_shape and not (
        len(expected_shape) == 3
        and image.shape[2] == expected_shape[0]
        and image.shape[0] == expected_shape[1]
        and image.shape[1] == expected_shape[2]
    ):
        raise RuntimeError(
            f"RGB image shape {image.shape} does not match expected feature shape {expected_shape}"
        )
    else:
        chw = np.transpose(image, (2, 0, 1))

    chw = np.asarray(chw, dtype=np.float32)
    if chw.max(initial=0.0) > 1.0:
        chw = chw / 255.0
    return torch.from_numpy(np.ascontiguousarray(chw))


def _to_depth_tensor(depth: np.ndarray, expected_shape: tuple[int, ...]) -> torch.Tensor:
    depth = np.asarray(depth, dtype=np.float32)

    if not expected_shape:
        return torch.from_numpy(np.ascontiguousarray(depth))

    if depth.shape == expected_shape:
        aligned = depth
    elif depth.ndim == 2 and len(expected_shape) == 3 and expected_shape[0] == 1:
        if depth.shape != expected_shape[1:]:
            raise RuntimeError(
                f"Depth image shape {depth.shape} does not match expected feature shape {expected_shape}"
            )
        aligned = depth[None, ...]
    else:
        raise RuntimeError(
            f"Depth image shape {depth.shape} does not match expected feature shape {expected_shape}"
        )

    return torch.from_numpy(np.ascontiguousarray(aligned))


def extract_expected_keys(policy: Any) -> set[str]:
    return set(get_input_feature_specs(policy).keys())


def get_input_feature_specs(policy: Any) -> dict[str, Any]:
    input_features = getattr(policy.config, "input_features", None)
    if hasattr(input_features, "items"):
        return dict(input_features.items())
    if isinstance(input_features, dict):
        return dict(input_features)
    return {}


def get_output_feature_specs(policy: Any) -> dict[str, Any]:
    output_features = getattr(policy.config, "output_features", None)
    if hasattr(output_features, "items"):
        return dict(output_features.items())
    if isinstance(output_features, dict):
        return dict(output_features)
    return {}


def get_feature_shape(spec: Any) -> tuple[int, ...]:
    if isinstance(spec, dict):
        shape = spec.get("shape", ())
    else:
        shape = getattr(spec, "shape", ())
    if shape is None:
        return ()
    return tuple(int(x) for x in shape)


def infer_visual_feature_keys(policy: Any) -> tuple[list[str], list[str]]:
    rgb_keys: list[str] = []
    depth_keys: list[str] = []
    for key in get_input_feature_specs(policy).keys():
        if key.startswith("observation.images_depth."):
            depth_keys.append(key.removeprefix("observation.images_depth."))
        elif key.startswith("observation.images."):
            rgb_keys.append(key.removeprefix("observation.images."))
    return rgb_keys, depth_keys


def infer_action_dim(policy: Any) -> int:
    action_spec = get_output_feature_specs(policy).get("action")
    shape = get_feature_shape(action_spec)
    if not shape:
        raise RuntimeError(
            "Policy output_features.action.shape is unavailable")
    return int(shape[-1])


def infer_chunk_length(policy: Any) -> int:
    n_action_steps = int(getattr(policy.config, "n_action_steps", 1) or 1)
    chunk_size = int(getattr(policy.config, "chunk_size",
                     n_action_steps) or n_action_steps)
    return max(1, min(n_action_steps, chunk_size))


def resolve_chunk_length(model_chunk_length: int, chunk_size: int | None) -> int:
    if chunk_size is None:
        return model_chunk_length
    requested = int(chunk_size)
    if requested <= 0:
        raise ValueError(f"chunk_size must be positive, got {requested}")
    if requested > model_chunk_length:
        raise ValueError(
            f"chunk_size ({requested}) cannot be greater than model chunk length ({model_chunk_length})"
        )
    return requested


def normalize_replan_settings(
    chunk_length: int,
    replan_interval: int,
    chunk_method: str,
) -> tuple[int, str | None]:
    if replan_interval <= 0:
        return chunk_length, None
    if replan_interval > chunk_length:
        raise ValueError(
            f"replan_interval ({replan_interval}) cannot be greater than chunk_length ({chunk_length})"
        )
    if replan_interval == chunk_length:
        return replan_interval, None
    return replan_interval, chunk_method


def get_arm_status(status: dict[str, Any], arm_side: str):
    arm_status = status.get(arm_side)
    if arm_status is None:
        raise RuntimeError(f"{arm_side} arm status is unavailable")
    return arm_status


def _status_field(status: Any, field_name: str) -> Any:
    if isinstance(status, dict):
        return status.get(field_name)
    return getattr(status, field_name, None)


def _joint_like_vector(status: dict[str, Any], side: str, field_name: str) -> np.ndarray:
    arm_status = get_arm_status(status, side)
    value = _status_field(arm_status, field_name)
    if value is None:
        raise RuntimeError(f"{side} {field_name} is unavailable")
    value = np.asarray(value, dtype=np.float32).reshape(-1)
    if value.shape[0] < 7:
        raise RuntimeError(f"{side} {field_name} shape invalid: {value.shape}")
    return value[:7]


def _eef_like_vector(status: dict[str, Any], side: str) -> np.ndarray:
    arm_status = get_arm_status(status, side)
    end_pos = _status_field(arm_status, "end_pos")
    joint_pos = _status_field(arm_status, "joint_pos")
    if end_pos is None or joint_pos is None:
        raise RuntimeError(f"{side} end_pos/joint_pos is unavailable")
    end_pos = np.asarray(end_pos, dtype=np.float32).reshape(-1)
    joint_pos = np.asarray(joint_pos, dtype=np.float32).reshape(-1)
    if end_pos.shape[0] < 6 or joint_pos.shape[0] < 7:
        raise RuntimeError(
            f"{side} end_pos/joint_pos shape invalid: {end_pos.shape}, {joint_pos.shape}")
    return np.concatenate([end_pos[:6], joint_pos[6:7]], axis=0).astype(np.float32)


def _pick_single_or_dual(
    status: dict[str, Any],
    arm_side: str,
    dim: int,
    value_kind: str,
) -> np.ndarray:
    if value_kind == "eef":
        left = _eef_like_vector(status, "left")
        right = _eef_like_vector(status, "right")
        single = _eef_like_vector(status, arm_side)
    elif value_kind == "state":
        left = _joint_like_vector(status, "left", "joint_pos")
        right = _joint_like_vector(status, "right", "joint_pos")
        single = _joint_like_vector(status, arm_side, "joint_pos")
    elif value_kind == "qvel":
        left = _joint_like_vector(status, "left", "joint_vel")
        right = _joint_like_vector(status, "right", "joint_vel")
        single = _joint_like_vector(status, arm_side, "joint_vel")
    elif value_kind == "effort":
        left = _joint_like_vector(status, "left", "joint_cur")
        right = _joint_like_vector(status, "right", "joint_cur")
        single = _joint_like_vector(status, arm_side, "joint_cur")
    else:
        raise ValueError(f"Unsupported value_kind: {value_kind}")

    dual = np.concatenate([left, right], axis=0).astype(np.float32)
    if dim <= single.shape[0]:
        return single[:dim]
    if dim <= dual.shape[0]:
        return dual[:dim]
    raise RuntimeError(
        f"Requested dim {dim} for {value_kind}, but only {single.shape[0]} or {dual.shape[0]} are available")


def _to_float_tensor(array: np.ndarray) -> torch.Tensor:
    return torch.from_numpy(np.ascontiguousarray(np.asarray(array, dtype=np.float32)))


def build_policy_observation(
    policy: Any,
    frames: dict[str, np.ndarray],
    status: dict[str, Any],
    arm_side: str,
    task_text: str | None = None,
) -> dict[str, Any]:
    payload: dict[str, Any] = {}
    input_specs = get_input_feature_specs(policy)

    for key, spec in input_specs.items():
        shape = get_feature_shape(spec)
        dim = int(shape[-1]) if shape else 0
        if key == "observation.state":
            payload[key] = _to_float_tensor(
                _pick_single_or_dual(status, arm_side, dim, "state")
            )
        elif key == "observation.qvel":
            payload[key] = _to_float_tensor(
                _pick_single_or_dual(status, arm_side, dim, "qvel")
            )
        elif key == "observation.effort":
            payload[key] = _to_float_tensor(
                _pick_single_or_dual(status, arm_side, dim, "effort")
            )
        elif key == "observation.eef":
            payload[key] = _to_float_tensor(
                _pick_single_or_dual(status, arm_side, dim, "eef")
            )
        elif key.startswith("observation.images."):
            camera_key = key.removeprefix("observation.images.")
            color_key = f"{camera_key}_color"
            if color_key not in frames:
                raise RuntimeError(f"Missing camera frame: {color_key}")
            payload[key] = _to_chw_float_image(
                bgr_to_rgb_if_needed(np.asarray(frames[color_key])),
                shape,
            )
        elif key.startswith("observation.images_depth."):
            camera_key = key.removeprefix("observation.images_depth.")
            depth_key = f"{camera_key}_aligned_depth_to_color"
            if depth_key not in frames:
                raise RuntimeError(f"Missing depth frame: {depth_key}")
            payload[key] = _to_depth_tensor(frames[depth_key], shape)

    if task_text:
        payload["task"] = task_text

    return payload


def unwrap_action_sequence(pred_action: Any, action_dim: int, max_action_steps: int | None = None) -> list[np.ndarray]:
    if isinstance(pred_action, dict):
        if "action" in pred_action:
            pred_action = pred_action["action"]
        else:
            first_key = next(iter(pred_action))
            pred_action = pred_action[first_key]

    if torch.is_tensor(pred_action):
        action = pred_action.detach().float().cpu().numpy()
    else:
        action = np.asarray(pred_action, dtype=np.float32)

    while action.ndim > 2:
        action = action[0]
    if action.ndim == 1:
        action = action.reshape(1, -1)
    elif action.ndim != 2:
        raise RuntimeError(
            f"Predicted action has unsupported shape: {action.shape}")

    if action.shape[-1] < action_dim:
        raise RuntimeError(
            f"Predicted action has invalid last dim: {action.shape}")
    if max_action_steps is not None:
        action = action[:max_action_steps]
    return [np.asarray(step[:action_dim], dtype=np.float32) for step in action]


def merge_action_chunks(
    pending_actions: list[np.ndarray],
    new_actions: list[np.ndarray],
    chunk_method: str = "replace",
) -> list[np.ndarray]:
    old = [np.asarray(action, dtype=np.float32) for action in pending_actions]
    new = [np.asarray(action, dtype=np.float32) for action in new_actions]

    if not old:
        return new

    if chunk_method == "replace":
        return new

    if chunk_method == "blend":
        overlap = min(len(old), len(new))
        blended = [
            0.5 * old[idx] + 0.5 * new[idx]
            for idx in range(overlap)
        ]
        return blended + new[overlap:]

    raise ValueError(
        f"Unsupported chunk_method={chunk_method!r}. "
        "Expected one of: replace, blend."
    )


def build_control_payload(action: np.ndarray, arm_side: str) -> dict[str, np.ndarray]:
    action = np.asarray(action, dtype=np.float32).reshape(-1)
    if action.shape[0] <= 7:
        return {arm_side: action[:7]}
    if action.shape[0] <= 14:
        if action.shape[0] < 14:
            raise RuntimeError(
                f"Dual-arm action must have 14 dims, got {action.shape}")
        return {"left": action[:7], "right": action[7:14]}
    raise RuntimeError(
        f"Unsupported action dim for control payload: {action.shape}")


def timestamp_string() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def build_dry_run_path(output_dir: str | Path, model_path: str) -> Path:
    output_root = Path(output_dir)
    output_root.mkdir(parents=True, exist_ok=True)
    model_name = Path(str(model_path).rstrip("/")).name or "model"
    return output_root / f"{model_name}_{timestamp_string()}.json"


def save_dry_run_records(
    output_dir: str | Path,
    model_path: str,
    records: list[dict[str, Any]],
) -> Path:
    path = build_dry_run_path(output_dir, model_path)
    serializable_records: list[dict[str, Any]] = []
    for record in records:
        item = dict(record)
        action = item.get("action")
        if isinstance(action, np.ndarray):
            item["action"] = action.astype(np.float32).tolist()
        serializable_records.append(item)
    path.write_text(json.dumps(serializable_records,
                    ensure_ascii=False, indent=2))
    return path


def load_dry_run_records(path: str | Path) -> list[dict[str, Any]]:
    data = json.loads(Path(path).read_text())
    if not isinstance(data, list):
        raise ValueError(f"Dry-run file must be a JSON list: {path}")
    return data


def infer_default_task_text(model_path: str) -> str | None:
    model_root = Path(model_path)
    candidates = [
        model_root / "meta" / "tasks.jsonl",
        model_root / "meta" / "tasks.parquet",
        model_root / "arx_collect_source.json",
    ]

    for path in candidates:
        if not path.exists():
            continue
        if path.name == "arx_collect_source.json":
            try:
                payload = json.loads(path.read_text())
                source_episodes = payload.get("source_episodes", [])
                if source_episodes:
                    task = str(source_episodes[0].get("task", "")).strip()
                    if task:
                        return task
            except Exception:
                continue
    return None


@dataclass(frozen=True)
class PolicyBundle:
    policy: Any
    preprocess: Any
    postprocess: Any
    expected_keys: set[str]
    rgb_camera_keys: list[str]
    depth_camera_keys: list[str]
    action_dim: int
    model_chunk_length: int
    device: torch.device


def _import_lerobot_dataset():
    try:
        from lerobot.datasets.lerobot_dataset import LeRobotDataset
    except ImportError as exc:
        raise ImportError(
            "Failed to import LeRobotDataset. Make sure the current Python environment "
            "contains the local `lerobot` dependencies such as `datasets`, `pyarrow`, and video backends."
        ) from exc
    return LeRobotDataset


def _import_act_policy():
    try:
        from lerobot.policies.act.modeling_act import ACTPolicy
    except ImportError:
        from lerobot.policies.act import ACTPolicy
    return ACTPolicy


def _import_diffusion_policy():
    from lerobot.policies.diffusion.modeling_diffusion import DiffusionPolicy

    return DiffusionPolicy


def resolve_pretrained_model_path(model_path: str | Path) -> str:
    raw_path = Path(model_path).expanduser()
    candidates: list[Path] = []

    if raw_path.is_absolute():
        candidates.append(raw_path)
    else:
        candidates.extend(
            [
                Path.cwd() / raw_path,
                REPO_ROOT / raw_path,
            ]
        )

    for candidate in candidates:
        if candidate.is_dir() and (candidate / CONFIG_FILENAME).is_file():
            return str(candidate.resolve())

    return str(model_path)


def _load_policy_bundle(policy_type: str, model_path: str, device: str) -> PolicyBundle:
    from lerobot.policies.factory import make_pre_post_processors

    torch_device = torch.device(device if torch.cuda.is_available() else "cpu")
    resolved_model_path = resolve_pretrained_model_path(model_path)

    if policy_type == "act":
        policy_cls = _import_act_policy()
    elif policy_type == "diffusion":
        policy_cls = _import_diffusion_policy()
    elif policy_type == "pi05":
        from lerobot.policies.pi05 import PI05Policy

        policy_cls = PI05Policy
    else:
        raise ValueError(
            f"Unsupported policy_type={policy_type!r}. Expected one of: act, diffusion, pi05.")

    policy = policy_cls.from_pretrained(
        resolved_model_path).to(torch_device).eval()
    preprocess, postprocess = make_pre_post_processors(
        policy.config,
        resolved_model_path,
        preprocessor_overrides={
            "device_processor": {"device": str(torch_device)}},
    )
    expected_keys = extract_expected_keys(policy)
    rgb_camera_keys, depth_camera_keys = infer_visual_feature_keys(policy)
    action_dim = infer_action_dim(policy)
    model_chunk_length = infer_chunk_length(policy)
    return PolicyBundle(
        policy=policy,
        preprocess=preprocess,
        postprocess=postprocess,
        expected_keys=expected_keys,
        rgb_camera_keys=rgb_camera_keys,
        depth_camera_keys=depth_camera_keys,
        action_dim=action_dim,
        model_chunk_length=model_chunk_length,
        device=torch_device,
    )


def load_policy_bundle(policy_type: str, model_path: str, device: str = "cuda") -> PolicyBundle:
    return _load_policy_bundle(policy_type=policy_type, model_path=model_path, device=device)


def _load_repo_id(dataset_root: Path) -> str:
    source_path = dataset_root / "arx_collect_source.json"
    if source_path.is_file():
        try:
            payload = json.loads(source_path.read_text(encoding="utf-8"))
            repo_id = str(payload.get("repo_id", "")).strip()
            if repo_id:
                return repo_id
        except Exception:
            pass
    return f"local/{dataset_root.name}"


def _load_action_names(dataset_root: Path) -> list[str]:
    info_path = dataset_root / "meta" / "info.json"
    if not info_path.is_file():
        return [f"action_{idx}" for idx in range(7)]

    try:
        info = json.loads(info_path.read_text(encoding="utf-8"))
        names = info.get("features", {}).get("action", {}).get("names", [])
        if isinstance(names, list) and names:
            return [str(name) for name in names]
    except Exception:
        pass
    return [f"action_{idx}" for idx in range(7)]


def _get_policy_action_queue_state(policy: Any) -> tuple[int | None, int | None]:
    action_queue = getattr(policy, "_action_queue", None)
    if action_queue is not None:
        maxlen = getattr(action_queue, "maxlen", None)
        return len(action_queue), int(maxlen) if maxlen is not None else len(action_queue)

    queues = getattr(policy, "_queues", None)
    if isinstance(queues, dict):
        from lerobot.utils.constants import ACTION

        action_queue = queues.get(ACTION)
        if action_queue is not None:
            maxlen = getattr(action_queue, "maxlen", None)
            return len(action_queue), int(maxlen) if maxlen is not None else len(action_queue)

    return None, None


def _scalar(value: Any) -> float:
    if torch.is_tensor(value):
        return float(value.detach().cpu().reshape(-1)[0].item())
    array = np.asarray(value)
    return float(array.reshape(-1)[0])


def _vector(value: Any, dtype=np.float32) -> np.ndarray:
    if torch.is_tensor(value):
        return value.detach().cpu().numpy().astype(dtype, copy=False).reshape(-1)
    return np.asarray(value, dtype=dtype).reshape(-1)


def _prepare_value(key: str, value: Any) -> Any:
    if key == "task":
        return str(value)

    if torch.is_tensor(value):
        tensor = value.detach().cpu()
    elif isinstance(value, np.ndarray):
        tensor = torch.from_numpy(np.ascontiguousarray(value))
    else:
        tensor = torch.as_tensor(value)

    if tensor.ndim == 0:
        tensor = tensor.reshape(1)

    if key.startswith("observation.images."):
        if not tensor.is_floating_point():
            tensor = tensor.float()
        if tensor.max().item() > 1.0:
            tensor = tensor / 255.0
        return tensor

    if key.startswith("observation.images_depth."):
        return tensor.float()

    if tensor.is_floating_point():
        return tensor.float()
    return tensor


def _build_policy_input(
    item: dict[str, Any],
    expected_keys: set[str],
    policy_type: str,
    task_text: str | None,
) -> dict[str, Any]:
    payload: dict[str, Any] = {}
    for key in sorted(expected_keys):
        if key not in item:
            raise KeyError(
                f"Dataset item is missing required key {key!r}. Available keys: {sorted(item.keys())}")
        payload[key] = _prepare_value(key, item[key])

    if policy_type == "pi05":
        resolved_task = task_text or str(item.get("task", "")).strip()
        if not resolved_task:
            raise RuntimeError(
                "PI05 open-loop test requires task text, but neither `task` nor dataset item task is available.")
        payload["task"] = resolved_task

    return payload


def _to_jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _to_jsonable(val) for key, val in value.items()}
    if isinstance(value, (list, tuple)):
        return [_to_jsonable(item) for item in value]
    if isinstance(value, np.ndarray):
        return value.astype(np.float32).tolist()
    if torch.is_tensor(value):
        return value.detach().cpu().numpy().astype(np.float32).tolist()
    if isinstance(value, (np.floating, np.integer)):
        return value.item()
    return value


def _save_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(_to_jsonable(payload), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def _load_matplotlib_pyplot(prefer_headless: bool):
    try:
        import matplotlib

        if prefer_headless and "matplotlib.pyplot" not in sys.modules:
            matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise ImportError(
            "matplotlib is required for plotting. Install it in the current Python environment.") from exc
    return plt


class LiveAlignmentPlotter:
    def __init__(self, action_names: list[str], title: str):
        plt = _load_matplotlib_pyplot(prefer_headless=False)
        plt.ion()

        self._plt = plt
        self._dim = len(action_names)
        self._fig, axes = plt.subplots(
            self._dim,
            1,
            figsize=(14, max(2.5 * self._dim, 10)),
            sharex=True,
            constrained_layout=True,
        )
        if self._dim == 1:
            axes = [axes]

        self._axes = list(axes)
        self._gt_lines = []
        self._pred_lines = []

        for idx, axis in enumerate(self._axes):
            label = action_names[idx]
            gt_line, = axis.plot([], [], label="gt",
                                 linewidth=1.8, color="#1f77b4")
            pred_line, = axis.plot([], [], label="pred",
                                   linewidth=1.4, color="#d62728", alpha=0.9)
            axis.set_ylabel(label)
            axis.grid(True, alpha=0.35)
            if idx == 0:
                axis.legend(loc="upper right")
            self._gt_lines.append(gt_line)
            self._pred_lines.append(pred_line)

        self._axes[-1].set_xlabel("time (s)")
        self._fig.suptitle(title)
        self._fig.canvas.draw_idle()
        self._fig.show()
        self._plt.pause(0.001)

    def update(self, time_s: np.ndarray, gt_actions: np.ndarray, pred_actions: np.ndarray) -> None:
        if time_s.size == 0:
            return

        x_min = float(time_s[0])
        x_max = float(time_s[-1])
        if x_max <= x_min:
            x_max = x_min + 1e-3

        for idx, axis in enumerate(self._axes):
            gt_vals = gt_actions[:, idx]
            pred_vals = pred_actions[:, idx]
            self._gt_lines[idx].set_data(time_s, gt_vals)
            self._pred_lines[idx].set_data(time_s, pred_vals)
            axis.set_xlim(x_min, x_max)

            y_vals = np.concatenate([gt_vals, pred_vals], axis=0)
            y_min = float(np.min(y_vals))
            y_max = float(np.max(y_vals))
            y_span = y_max - y_min
            pad = max(y_span * 0.08, 1e-4)
            if y_span < 1e-6:
                pad = max(abs(y_min) * 0.1, 1e-3)
            axis.set_ylim(y_min - pad, y_max + pad)

        self._fig.canvas.draw_idle()
        self._plt.pause(0.001)

    def show(self) -> None:
        self._plt.ioff()
        self._plt.show()


def _plot_alignment(
    figure_path: Path,
    per_dim_dir: Path,
    time_s: np.ndarray,
    gt_actions: np.ndarray,
    pred_actions: np.ndarray,
    action_names: list[str],
    title: str,
) -> None:
    plt = _load_matplotlib_pyplot(prefer_headless=True)

    figure_path.parent.mkdir(parents=True, exist_ok=True)
    per_dim_dir.mkdir(parents=True, exist_ok=True)

    dim = gt_actions.shape[1]
    fig, axes = plt.subplots(
        dim,
        1,
        figsize=(14, max(2.5 * dim, 10)),
        sharex=True,
        constrained_layout=True,
    )
    if dim == 1:
        axes = [axes]

    for idx, axis in enumerate(axes):
        label = action_names[idx] if idx < len(
            action_names) else f"action_{idx}"
        axis.plot(time_s, gt_actions[:, idx],
                  label="gt", linewidth=1.8, color="#1f77b4")
        axis.plot(time_s, pred_actions[:, idx], label="pred",
                  linewidth=1.4, color="#d62728", alpha=0.9)
        axis.set_ylabel(label)
        axis.grid(True, alpha=0.35)
        if idx == 0:
            axis.legend(loc="upper right")

        per_dim_fig, per_dim_axis = plt.subplots(
            figsize=(12, 4.5), constrained_layout=True)
        per_dim_axis.plot(
            time_s, gt_actions[:, idx], label="gt", linewidth=1.8, color="#1f77b4")
        per_dim_axis.plot(
            time_s, pred_actions[:, idx], label="pred", linewidth=1.4, color="#d62728", alpha=0.9)
        per_dim_axis.set_title(label)
        per_dim_axis.set_xlabel("time (s)")
        per_dim_axis.set_ylabel("action value")
        per_dim_axis.grid(True, alpha=0.35)
        per_dim_axis.legend(loc="upper right")
        per_dim_path = per_dim_dir / f"{idx:02d}_{label.replace('/', '_')}.png"
        per_dim_fig.savefig(per_dim_path, dpi=200)
        plt.close(per_dim_fig)

    axes[-1].set_xlabel("time (s)")
    fig.suptitle(title)
    fig.savefig(figure_path, dpi=200)
    plt.close(fig)


def run_open_loop_dryrun(
    model_path: str,
    dataset_root: str | Path,
    episode_index: int = 0,
    policy_type: str = "act",
    repo_id: str | None = None,
    chunk_size: int | None = None,
    replan_interval: int = 0,
    chunk_method: str = "replace",
    max_steps: int = 0,
    device: str = "cuda",
    output_dir: str | Path = REPO_ROOT / "dryrun_records" / "open_loop",
    task: str | None = None,
    video_backend: str = "pyav",
    show_plot: bool = False,
    rollout_mode: str = "official",
) -> dict[str, Path]:
    os.environ.setdefault("HF_HUB_OFFLINE", "1")

    if rollout_mode not in {"chunked", "official"}:
        raise ValueError(
            f"Unsupported rollout_mode={rollout_mode!r}. Expected one of: chunked, official."
        )

    dataset_root = Path(dataset_root).resolve()
    if not dataset_root.is_dir():
        raise FileNotFoundError(f"Dataset root not found: {dataset_root}")

    repo_id = repo_id or _load_repo_id(dataset_root)
    LeRobotDataset = _import_lerobot_dataset()
    dataset = LeRobotDataset(
        repo_id=repo_id,
        root=dataset_root,
        episodes=[int(episode_index)],
        download_videos=False,
        video_backend=video_backend,
    )

    bundle = _load_policy_bundle(policy_type, model_path, device=device)
    bundle.policy.reset()
    chunk_length = bundle.model_chunk_length
    effective_chunk_method: str | None = None
    summary_replan_interval: int | None = None
    summary_chunk_method: str | None = None
    if rollout_mode == "chunked":
        chunk_length = resolve_chunk_length(
            bundle.model_chunk_length, chunk_size)
        replan_interval, effective_chunk_method = normalize_replan_settings(
            chunk_length=chunk_length,
            replan_interval=replan_interval,
            chunk_method=chunk_method,
        )
        summary_replan_interval = int(replan_interval)
        summary_chunk_method = effective_chunk_method or "disabled"

    total_steps = len(dataset)
    if total_steps <= 0:
        raise RuntimeError(
            f"Episode {episode_index} is empty in dataset {dataset_root}")
    if max_steps > 0:
        total_steps = min(total_steps, int(max_steps))

    sample0 = dataset[0]
    gt_action0 = _vector(sample0["action"])
    if gt_action0.shape[0] != bundle.action_dim:
        raise RuntimeError(
            f"Policy action_dim={bundle.action_dim} does not match dataset action dim={gt_action0.shape[0]}")
    action_names = _load_action_names(dataset_root)
    if len(action_names) < gt_action0.shape[0]:
        action_names.extend(
            [f"action_{idx}" for idx in range(len(action_names), gt_action0.shape[0])])
    action_names = action_names[:gt_action0.shape[0]]

    task_text = task
    if policy_type == "pi05":
        task_text = task_text or str(sample0.get("task", "")).strip(
        ) or infer_default_task_text(model_path)

    output_root = Path(output_dir).resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    stamp = timestamp_string()
    model_name = Path(str(model_path).rstrip("/")).name or "model"
    dataset_name = dataset_root.name
    stem = f"{model_name}_{dataset_name}_ep{episode_index:03d}_{stamp}"

    per_dim_dir = output_root / f"{stem}_dry_run"
    records_path = per_dim_dir / f"{stem}.json"
    summary_path = per_dim_dir / f"{stem}_summary.json"
    figure_path = per_dim_dir / f"{stem}.png"

    print(
        f"Open-loop dry-run started: policy_type={policy_type}, rollout_mode={rollout_mode}, "
        f"model={model_path}, "
        f"dataset={dataset_root}, repo_id={repo_id}, episode={episode_index}, "
        f"steps={total_steps}, action_dim={bundle.action_dim}, "
        f"chunk_length={chunk_length}, model_chunk_length={bundle.model_chunk_length}, "
        f"replan_interval={summary_replan_interval}, "
        f"chunk_method={summary_chunk_method}, "
        f"rgb_cameras={bundle.rgb_camera_keys}, depth_cameras={bundle.depth_camera_keys}, "
        f"device={bundle.device}"
    )
    if task_text:
        print(f"Task text: {task_text!r}")

    records: list[dict[str, Any]] = []
    pending_actions: deque[np.ndarray] = deque()
    steps_since_replan = 0
    plan_origin_step = 0
    official_plan_origin_step = 0
    episode_start_ts = _scalar(sample0["timestamp"])
    live_plotter = None
    live_time_s: list[float] = []
    live_gt_actions: list[np.ndarray] = []
    live_pred_actions: list[np.ndarray] = []

    if show_plot:
        live_plotter = LiveAlignmentPlotter(
            action_names=action_names,
            title=(
                f"Open-loop pred vs gt | {model_name} | "
                f"{dataset_name} | episode {episode_index}"
            ),
        )

    for step_idx in range(total_steps):
        item = sample0 if step_idx == 0 else dataset[step_idx]

        payload = _build_policy_input(
            item=item,
            expected_keys=bundle.expected_keys,
            policy_type=policy_type,
            task_text=task_text,
        )
        batch = bundle.preprocess(dict(payload))

        if rollout_mode == "official":
            queue_len_before, _ = _get_policy_action_queue_state(bundle.policy)
            t0 = time.perf_counter()
            with torch.inference_mode():
                pred = bundle.policy.select_action(batch)
                pred = bundle.postprocess(pred)
            latency_ms = (time.perf_counter() - t0) * 1000.0
            pred_action = np.asarray(
                unwrap_action_sequence(
                    pred,
                    action_dim=bundle.action_dim,
                    max_action_steps=1,
                )[0],
                dtype=np.float32,
            ).reshape(-1)
            queue_len_after, queue_capacity = _get_policy_action_queue_state(
                bundle.policy)
            if queue_capacity is None or queue_len_after is None:
                plan_origin_step = step_idx
                chunk_offset = 0
            else:
                if queue_len_before == 0:
                    official_plan_origin_step = step_idx
                plan_origin_step = official_plan_origin_step
                chunk_offset = max(0, int(queue_capacity) -
                                   int(queue_len_after) - 1)
        else:
            latency_ms = 0.0
            should_replan = not pending_actions
            if not should_replan and replan_interval > 0 and steps_since_replan >= replan_interval:
                should_replan = True

            if should_replan:
                plan_origin_step = step_idx
                t0 = time.perf_counter()
                if policy_type == "diffusion":
                    from lerobot.policies.utils import populate_queues
                    from lerobot.utils.constants import ACTION
                    from lerobot.utils.constants import OBS_IMAGES

                    batch = dict(batch)
                    batch.pop(ACTION, None)
                    if bundle.policy.config.image_features:
                        batch[OBS_IMAGES] = torch.stack(
                            [batch[key]
                                for key in bundle.policy.config.image_features],
                            dim=-4,
                        )

                    bundle.policy._queues = populate_queues(
                        bundle.policy._queues, batch)
                with torch.inference_mode():
                    pred = bundle.policy.predict_action_chunk(batch)
                    pred = bundle.postprocess(pred)
                new_actions = unwrap_action_sequence(
                    pred,
                    action_dim=bundle.action_dim,
                    max_action_steps=chunk_length,
                )
                merged_actions = merge_action_chunks(
                    list(pending_actions),
                    new_actions,
                    chunk_method=effective_chunk_method if (
                        pending_actions and effective_chunk_method) else "replace",
                )
                pending_actions = deque(merged_actions)
                steps_since_replan = 0
                latency_ms = (time.perf_counter() - t0) * 1000.0

            if not pending_actions:
                raise RuntimeError(
                    f"No predicted action available at step {step_idx}")

            pred_action = np.asarray(
                pending_actions.popleft(), dtype=np.float32).reshape(-1)
            chunk_offset = step_idx - plan_origin_step

        gt_action = _vector(item["action"])
        rel_time_s = _scalar(item["timestamp"]) - episode_start_ts
        abs_error = np.abs(pred_action - gt_action)

        records.append(
            {
                "step": step_idx,
                "episode_index": int(_scalar(item["episode_index"])),
                "frame_index": int(_scalar(item["frame_index"])),
                "time_s": rel_time_s,
                "dataset_timestamp_s": _scalar(item["timestamp"]),
                "plan_origin_step": plan_origin_step,
                "chunk_offset": chunk_offset,
                "latency_ms": latency_ms,
                "task": str(item.get("task", task_text or "")),
                "gt_action": gt_action,
                "pred_action": pred_action,
                "abs_error": abs_error,
            }
        )

        if live_plotter is not None:
            live_time_s.append(rel_time_s)
            live_gt_actions.append(gt_action.copy())
            live_pred_actions.append(pred_action.copy())
            live_plotter.update(
                np.asarray(live_time_s, dtype=np.float32),
                np.asarray(live_gt_actions, dtype=np.float32),
                np.asarray(live_pred_actions, dtype=np.float32),
            )

        if step_idx % 20 == 0 or step_idx == total_steps - 1:
            print(
                f"[step {step_idx:05d}] t={rel_time_s:8.3f}s "
                f"latency_ms={latency_ms:7.2f} "
                f"gt={np.round(gt_action, 4).tolist()} "
                f"pred={np.round(pred_action, 4).tolist()}"
            )

        if rollout_mode == "chunked":
            steps_since_replan += 1

    time_s = np.asarray([record["time_s"]
                        for record in records], dtype=np.float32)
    gt_actions = np.asarray([record["gt_action"]
                            for record in records], dtype=np.float32)
    pred_actions = np.asarray([record["pred_action"]
                              for record in records], dtype=np.float32)
    abs_error = np.abs(pred_actions - gt_actions)
    sq_error = (pred_actions - gt_actions) ** 2

    summary = {
        "policy_type": policy_type,
        "rollout_mode": rollout_mode,
        "model_path": str(model_path),
        "dataset_root": str(dataset_root),
        "repo_id": repo_id,
        "episode_index": int(episode_index),
        "total_steps": int(total_steps),
        "chunk_length": int(chunk_length),
        "model_chunk_length": int(bundle.model_chunk_length),
        "replan_interval": summary_replan_interval,
        "chunk_method": summary_chunk_method,
        "device": str(bundle.device),
        "task": task_text or str(sample0.get("task", "")).strip(),
        "action_names": action_names,
        "mae_per_dim": abs_error.mean(axis=0),
        "rmse_per_dim": np.sqrt(sq_error.mean(axis=0)),
        "mae_mean": float(abs_error.mean()),
        "rmse_mean": float(np.sqrt(sq_error.mean())),
        "records_path": str(records_path),
        "figure_path": str(figure_path),
        "per_dim_dir": str(per_dim_dir),
    }

    _save_json(records_path, records)
    _save_json(summary_path, summary)
    _plot_alignment(
        figure_path=figure_path,
        per_dim_dir=per_dim_dir,
        time_s=time_s,
        gt_actions=gt_actions,
        pred_actions=pred_actions,
        action_names=action_names,
        title=(
            f"Open-loop pred vs gt | {model_name} | "
            f"{dataset_name} | episode {episode_index}"
        ),
    )

    if live_plotter is not None:
        live_plotter.show()

    print(f"Aligned records saved to: {records_path}")
    print(f"Summary saved to: {summary_path}")
    print(f"Combined figure saved to: {figure_path}")
    print(f"Per-dimension figures saved to: {per_dim_dir}")
    print(
        f"MAE mean={summary['mae_mean']:.6f}, "
        f"RMSE mean={summary['rmse_mean']:.6f}, "
        f"MAE per dim={np.round(np.asarray(summary['mae_per_dim']), 6).tolist()}"
    )

    return {
        "records_path": records_path,
        "summary_path": summary_path,
        "figure_path": figure_path,
        "per_dim_dir": per_dim_dir,
    }
