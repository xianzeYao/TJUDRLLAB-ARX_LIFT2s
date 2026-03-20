#!/usr/bin/env python3
"""Minimal deployment helpers used by inference servers."""

from __future__ import annotations

import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch


CONFIG_FILENAME = "config.json"

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
LEROBOT_SRC_ROOT = REPO_ROOT / "lerobot" / "src"
if str(LEROBOT_SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(LEROBOT_SRC_ROOT))


def bgr_to_rgb_if_needed(image: np.ndarray) -> np.ndarray:
    image = np.asarray(image)
    if image.ndim == 3 and image.shape[2] == 3:
        return image[:, :, ::-1].copy()
    return image


def _to_chw_float_image(image: np.ndarray, expected_shape: tuple[int, ...]) -> torch.Tensor:
    image = np.asarray(image)
    if image.ndim != 3:
        raise RuntimeError(f"Expected RGB image with 3 dims, got shape {image.shape}")

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


def extract_expected_keys(policy: Any) -> set[str]:
    return set(get_input_feature_specs(policy).keys())


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
        raise RuntimeError("Policy output_features.action.shape is unavailable")
    return int(shape[-1])


def infer_chunk_length(policy: Any) -> int:
    n_action_steps = int(getattr(policy.config, "n_action_steps", 1) or 1)
    chunk_size = int(getattr(policy.config, "chunk_size", n_action_steps) or n_action_steps)
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


def _status_field(status: Any, field_name: str) -> Any:
    if isinstance(status, dict):
        return status.get(field_name)
    return getattr(status, field_name, None)


def get_arm_status(status: dict[str, Any], arm_side: str) -> Any:
    arm_status = status.get(arm_side)
    if arm_status is None:
        raise RuntimeError(f"{arm_side} arm status is unavailable")
    return arm_status


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
            f"{side} end_pos/joint_pos shape invalid: {end_pos.shape}, {joint_pos.shape}"
        )
    return np.concatenate([end_pos[:6], joint_pos[6:7]], axis=0).astype(np.float32)


def _pick_single_or_dual(status: dict[str, Any], arm_side: str, dim: int, value_kind: str) -> np.ndarray:
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
        f"Requested dim {dim} for {value_kind}, but only {single.shape[0]} or {dual.shape[0]} are available"
    )


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
            payload[key] = _to_float_tensor(_pick_single_or_dual(status, arm_side, dim, "state"))
        elif key == "observation.qvel":
            payload[key] = _to_float_tensor(_pick_single_or_dual(status, arm_side, dim, "qvel"))
        elif key == "observation.effort":
            payload[key] = _to_float_tensor(_pick_single_or_dual(status, arm_side, dim, "effort"))
        elif key == "observation.eef":
            payload[key] = _to_float_tensor(_pick_single_or_dual(status, arm_side, dim, "eef"))
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
        raise RuntimeError(f"Predicted action has unsupported shape: {action.shape}")

    if action.shape[-1] < action_dim:
        raise RuntimeError(f"Predicted action has invalid last dim: {action.shape}")
    if max_action_steps is not None:
        action = action[:max_action_steps]
    return [np.asarray(step[:action_dim], dtype=np.float32) for step in action]


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


def resolve_pretrained_model_path(model_path: str | Path) -> str:
    raw_path = Path(model_path).expanduser()
    candidates: list[Path] = []

    if raw_path.is_absolute():
        candidates.append(raw_path)
    else:
        candidates.extend([Path.cwd() / raw_path, REPO_ROOT / raw_path])

    for candidate in candidates:
        if candidate.is_dir() and (candidate / CONFIG_FILENAME).is_file():
            return str(candidate.resolve())

    return str(model_path)


def load_policy_bundle(policy_type: str, model_path: str, device: str = "cuda") -> PolicyBundle:
    from lerobot.policies.factory import make_pre_post_processors

    torch_device = torch.device(device if torch.cuda.is_available() else "cpu")
    resolved_model_path = resolve_pretrained_model_path(model_path)

    if policy_type == "pi05":
        from lerobot.policies.pi05 import PI05Policy

        policy_cls = PI05Policy
    else:
        raise ValueError(f"Unsupported policy_type={policy_type!r}. Expected one of: pi05.")

    policy = policy_cls.from_pretrained(resolved_model_path).to(torch_device).eval()
    preprocess, postprocess = make_pre_post_processors(
        policy.config,
        resolved_model_path,
        preprocessor_overrides={"device_processor": {"device": str(torch_device)}},
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
