#!/usr/bin/env python3
"""Chunked diffusion deployment helper for ARX."""

from __future__ import annotations

import sys
import time
from collections import deque
from pathlib import Path

import numpy as np
import torch


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
DEPLOYMENT_ROOT = Path(__file__).resolve().parent
if str(DEPLOYMENT_ROOT) not in sys.path:
    sys.path.insert(0, str(DEPLOYMENT_ROOT))

from ARX_Realenv.ROS2.arx_ros2_env import ARXRobotEnv  # noqa: E402
from deployment_utils import (  # noqa: E402
    build_control_payload,
    build_policy_observation,
    extract_expected_keys,
    infer_action_dim,
    infer_chunk_length,
    infer_visual_feature_keys,
    merge_action_chunks,
    normalize_replan_settings,
    resolve_chunk_length,
    resolve_pretrained_model_path,
    run_open_loop_dryrun,
    save_dry_run_records,
    unwrap_action_sequence,
)


def load_diffusion_policy(model_path: str, device: str = "cuda"):
    from lerobot.policies.diffusion.modeling_diffusion import DiffusionPolicy
    from lerobot.policies.factory import make_pre_post_processors

    torch_device = torch.device(device if torch.cuda.is_available() else "cpu")
    resolved_model_path = resolve_pretrained_model_path(model_path)
    policy = DiffusionPolicy.from_pretrained(
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
    chunk_length = infer_chunk_length(policy)
    return (
        policy,
        preprocess,
        postprocess,
        expected_keys,
        rgb_camera_keys,
        depth_camera_keys,
        action_dim,
        chunk_length,
    )


def _predict_diffusion_chunk(
    policy,
    preprocess,
    postprocess,
    payload: dict,
    action_dim: int,
    chunk_length: int,
):
    from lerobot.policies.utils import populate_queues
    from lerobot.utils.constants import ACTION
    from lerobot.utils.constants import OBS_IMAGES

    batch = preprocess(dict(payload))
    batch = dict(batch)
    batch.pop(ACTION, None)
    if policy.config.image_features:
        batch[OBS_IMAGES] = torch.stack(
            [batch[key] for key in policy.config.image_features],
            dim=-4,
        )
    policy._queues = populate_queues(policy._queues, batch)
    with torch.inference_mode():
        pred_action = policy.predict_action_chunk(batch)
        pred_action = postprocess(pred_action)
    return unwrap_action_sequence(
        pred_action,
        action_dim=action_dim,
        max_action_steps=chunk_length,
    )


def run_diffusion(
    arx: ARXRobotEnv | None,
    model_path: str,
    arm_side: str = "right",
    hz: float = 20.0,
    chunk_size: int | None = None,
    replan_interval: int = 0,
    chunk_method: str = "replace",
    max_steps: int = 0,
    device: str = "cuda",
    dry_run: bool = False,
    dry_run_dir: str = "dryrun_records/push_away_diffuion_chunked",
    dataset_root: str | None = None,
    episode_index: int = 0,
    show_plot: bool = False,
):
    """Run diffusion with deployment-side chunk/replan logic."""
    if dataset_root is not None and not dry_run:
        raise ValueError(
            "`dataset_root` is only supported when `dry_run=True`.")

    if dry_run and dataset_root:
        return run_open_loop_dryrun(
            model_path=model_path,
            dataset_root=dataset_root,
            episode_index=episode_index,
            policy_type="diffusion",
            chunk_size=chunk_size,
            replan_interval=replan_interval,
            chunk_method=chunk_method,
            max_steps=max_steps,
            device=device,
            output_dir=dry_run_dir,
            show_plot=show_plot,
            rollout_mode="chunked",
        )

    if arx is None:
        raise ValueError(
            "`arx` must be provided when `dataset_root` is not set.")

    (
        policy,
        preprocess,
        postprocess,
        expected_keys,
        rgb_camera_keys,
        depth_camera_keys,
        action_dim,
        model_chunk_length,
    ) = load_diffusion_policy(model_path, device=device)
    del expected_keys

    policy.reset()
    chunk_length = resolve_chunk_length(model_chunk_length, chunk_size)
    replan_interval, effective_chunk_method = normalize_replan_settings(
        chunk_length=chunk_length,
        replan_interval=replan_interval,
        chunk_method=chunk_method,
    )
    if effective_chunk_method == "replace":
        print(
            "Warning: partial replanning with chunk_method='replace' may cause visible joint jumps. "
            "Prefer chunk_method='blend' or set replan_interval=chunk_size for smoother control."
        )

    dry_run_records = []
    pending_actions: deque[np.ndarray] = deque()

    warmup_frames, warmup_status = arx.get_camera(
        save_dir=None,
        video=False,
        target_size=arx.img_size,
        return_status=True,
    )
    warmup_payload = build_policy_observation(
        policy,
        warmup_frames,
        warmup_status,
        arm_side=arm_side,
    )
    pending_actions.extend(
        _predict_diffusion_chunk(
            policy,
            preprocess,
            postprocess,
            warmup_payload,
            action_dim=action_dim,
            chunk_length=chunk_length,
        )
    )

    print(
        f"Diffusion chunked deployment started: side={arm_side}, cameras={rgb_camera_keys}, "
        f"depth_cameras={depth_camera_keys}, action_dim={action_dim}, chunk_length={chunk_length}, "
        f"model_chunk_length={model_chunk_length}, hz={hz:.2f}, "
        f"replan_interval={replan_interval}, "
        f"chunk_method={effective_chunk_method or 'disabled'}, dry_run={dry_run}"
    )
    step_idx = 0
    steps_since_replan = 0
    period = 1.0 / max(hz, 1e-6)

    try:
        while max_steps <= 0 or step_idx < max_steps:
            t0 = time.perf_counter()
            latency_ms = 0.0
            should_replan = not pending_actions
            if not should_replan and replan_interval > 0 and steps_since_replan >= replan_interval:
                should_replan = True

            if should_replan:
                frames, status = arx.get_camera(
                    save_dir=None,
                    video=False,
                    target_size=arx.img_size,
                    return_status=True,
                )
                payload = build_policy_observation(
                    policy,
                    frames,
                    status,
                    arm_side=arm_side,
                )
                new_actions = _predict_diffusion_chunk(
                    policy,
                    preprocess,
                    postprocess,
                    payload,
                    action_dim=action_dim,
                    chunk_length=chunk_length,
                )
                merged_actions = merge_action_chunks(
                    list(pending_actions),
                    new_actions,
                    chunk_method=effective_chunk_method if (
                        pending_actions and effective_chunk_method
                    ) else "replace",
                )
                pending_actions = deque(merged_actions)
                steps_since_replan = 0
                latency_ms = (time.perf_counter() - t0) * 1000.0

            if not pending_actions:
                raise RuntimeError(
                    f"No predicted action available at step {step_idx}")

            action = pending_actions.popleft()
            control_payload = build_control_payload(action, arm_side)

            if dry_run:
                dry_run_records.append(
                    {
                        "step": step_idx,
                        "wall_time": time.time(),
                        "latency_ms": latency_ms,
                        "arm_side": arm_side if action_dim <= 7 else "both",
                        "action_type": "joint",
                        "action": action.tolist(),
                    }
                )
                print(
                    f"[step {step_idx:05d}] action={action.round(4)} latency_ms={latency_ms:.1f}"
                )
            else:
                arx.step_smooth_joint(control_payload)

            elapsed = time.perf_counter() - t0
            time.sleep(max(0.0, period - elapsed))
            step_idx += 1
            steps_since_replan += 1
    finally:
        if dry_run and dry_run_records:
            record_path = save_dry_run_records(
                dry_run_dir, model_path, dry_run_records)
            print(f"Dry-run actions saved to: {record_path}")


def main() -> None:
    dry_run = False
    dataset_root = None
    # dataset_root = "/home/arx/Arx_Lift2s/Collect/lerobot_v3/push_away_cube_v2"

    arx = None
    if not (dry_run and dataset_root):
        arx = ARXRobotEnv(
            duration_per_step=1.0 / 20.0,
            min_steps=20,
            max_v_xyz=0.25,
            max_a_xyz=0.20,
            max_v_rpy=0.3,
            max_a_rpy=1.00,
            camera_type="all",
            camera_view=("camera_h", "camera_r"),
            img_size=(640, 480),
        )
    try:
        if arx is not None:
            arx.reset()
            arx.step_lift(12.5)
        run_diffusion(
            arx=arx,
            model_path="/home/arx/Arx_Lift2s/Deployment/models/push_away_diffusion",
            arm_side="right",
            hz=20.0,
            chunk_size=12,
            replan_interval=12,
            chunk_method="replace",
            max_steps=150,
            dry_run=dry_run,
            dataset_root=dataset_root,
            episode_index=0,
            dry_run_dir="dryrun_records/push_away_diffuion_chunked",
            show_plot=False,
        )
    finally:
        if arx is not None:
            arx.close()


if __name__ == "__main__":
    main()
