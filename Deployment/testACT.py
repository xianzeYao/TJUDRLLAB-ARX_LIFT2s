#!/usr/bin/env python3
"""Chunked ACT deployment helper for ARX."""

from __future__ import annotations

from concurrent.futures import Future, ThreadPoolExecutor
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
ROS2_ROOT = REPO_ROOT / "ARX_Realenv" / "ROS2"
if str(ROS2_ROOT) not in sys.path:
    sys.path.insert(0, str(ROS2_ROOT))

from arx_ros2_env import ARXRobotEnv  # noqa: E402
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


def _import_act_policy():
    try:
        from lerobot.policies.act.modeling_act import ACTPolicy
    except ImportError:
        from lerobot.policies.act import ACTPolicy
    return ACTPolicy


def _import_act_temporal_ensembler():
    from lerobot.policies.act.modeling_act import ACTTemporalEnsembler

    return ACTTemporalEnsembler


def load_act_policy(model_path: str, device: str = "cuda"):
    from lerobot.policies.factory import make_pre_post_processors

    torch_device = torch.device(device if torch.cuda.is_available() else "cpu")
    resolved_model_path = resolve_pretrained_model_path(model_path)
    ACTPolicy = _import_act_policy()
    policy = ACTPolicy.from_pretrained(
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


def _predict_act_chunk_tensor(
    arx: ARXRobotEnv,
    policy,
    preprocess,
    postprocess,
    arm_side: str,
    chunk_length: int | None = None,
) -> tuple[torch.Tensor, float]:
    t0 = time.perf_counter()
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
    batch = preprocess(dict(payload))
    with torch.inference_mode():
        pred_action = policy.predict_action_chunk(batch)
        pred_action = postprocess(pred_action)
    if not torch.is_tensor(pred_action):
        pred_action = torch.as_tensor(pred_action, dtype=torch.float32)
    while pred_action.ndim > 3:
        pred_action = pred_action[0]
    if pred_action.ndim == 2:
        pred_action = pred_action.unsqueeze(0)
    if chunk_length is not None:
        pred_action = pred_action[:, :chunk_length]
    latency_ms = (time.perf_counter() - t0) * 1000.0
    return pred_action.detach(), latency_ms


def _plan_act_chunk(
    arx: ARXRobotEnv,
    policy,
    preprocess,
    postprocess,
    arm_side: str,
    action_dim: int,
    chunk_length: int,
) -> tuple[list[np.ndarray], float]:
    pred_action, latency_ms = _predict_act_chunk_tensor(
        arx,
        policy,
        preprocess,
        postprocess,
        arm_side,
        chunk_length=chunk_length,
    )
    new_actions = unwrap_action_sequence(
        pred_action,
        action_dim=action_dim,
        max_action_steps=chunk_length,
    )
    return new_actions, latency_ms


def run_act(
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
    dry_run_dir: str = "dryrun_records/push_away/act_chunked",
    dataset_root: str | None = None,
    episode_index: int = 0,
    show_plot: bool = False,
    async_prefetch: bool = True,
    prefetch_lead_steps: int = 4,
    temporal_ensemble_coeff: float | None = None,
):
    """Run ACT with deployment-side chunk/replan logic."""
    if dataset_root is not None and not dry_run:
        raise ValueError(
            "`dataset_root` is only supported when `dry_run=True`.")

    if dry_run and dataset_root:
        return run_open_loop_dryrun(
            model_path=model_path,
            dataset_root=dataset_root,
            episode_index=episode_index,
            policy_type="act",
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
    ) = load_act_policy(model_path, device=device)
    del expected_keys

    chunk_length = resolve_chunk_length(model_chunk_length, chunk_size)
    if temporal_ensemble_coeff is not None:
        ACTTemporalEnsembler = _import_act_temporal_ensembler()
        temporal_ensembler = ACTTemporalEnsembler(
            temporal_ensemble_coeff=float(temporal_ensemble_coeff),
            chunk_size=chunk_length,
        )
        warmup_pred, warmup_latency_ms = _predict_act_chunk_tensor(
            arx,
            policy,
            preprocess,
            postprocess,
            arm_side,
            chunk_length=chunk_length,
        )
        current_action = unwrap_action_sequence(
            temporal_ensembler.update(warmup_pred),
            action_dim=action_dim,
            max_action_steps=1,
        )[0]
        print(
            f"ACT temporal-ensemble deployment started: side={arm_side}, cameras={rgb_camera_keys}, "
            f"depth_cameras={depth_camera_keys}, action_dim={action_dim}, chunk_length={chunk_length}, "
            f"model_chunk_length={model_chunk_length}, hz={hz:.2f}, dry_run={dry_run}, "
            f"temporal_ensemble_coeff={temporal_ensemble_coeff:.4f}, "
            f"warmup_latency_ms={warmup_latency_ms:.1f}"
        )

        dry_run_records = []
        step_idx = 0
        period = 1.0 / max(hz, 1e-6)
        try:
            while max_steps <= 0 or step_idx < max_steps:
                t0 = time.perf_counter()
                control_payload = build_control_payload(current_action, arm_side)
                if dry_run:
                    dry_run_records.append(
                        {
                            "step": step_idx,
                            "wall_time": time.time(),
                            "latency_ms": 0.0,
                            "plan_source": "temporal_ensemble",
                            "prefetch_target_step": None,
                            "arm_side": arm_side if action_dim <= 7 else "both",
                            "action_type": "joint",
                            "action": current_action.tolist(),
                        }
                    )
                    print(
                        f"[step {step_idx:05d}] action={current_action.round(4)} "
                        "latency_ms=0.0 plan_source=temporal_ensemble"
                    )
                else:
                    arx.step_smooth_joint(control_payload)

                next_pred, latency_ms = _predict_act_chunk_tensor(
                    arx,
                    policy,
                    preprocess,
                    postprocess,
                    arm_side,
                    chunk_length=chunk_length,
                )
                current_action = unwrap_action_sequence(
                    temporal_ensembler.update(next_pred),
                    action_dim=action_dim,
                    max_action_steps=1,
                )[0]
                print(
                    f"[step {step_idx:05d}] replanned via temporal_ensemble "
                    f"latency_ms={latency_ms:.1f}"
                )

                elapsed = time.perf_counter() - t0
                time.sleep(max(0.0, period - elapsed))
                step_idx += 1
        finally:
            if dry_run and dry_run_records:
                record_path = save_dry_run_records(
                    dry_run_dir, model_path, dry_run_records)
                print(f"Dry-run actions saved to: {record_path}")
        return

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
    prefetch_lead_steps = max(1, int(prefetch_lead_steps))
    executor = ThreadPoolExecutor(
        max_workers=1,
        thread_name_prefix="act_prefetch",
    ) if async_prefetch else None
    prefetch_future: Future | None = None
    prefetch_target_step: int | None = None

    def submit_prefetch(target_step: int) -> None:
        nonlocal prefetch_future, prefetch_target_step
        if executor is None or prefetch_future is not None:
            return
        prefetch_target_step = int(target_step)
        prefetch_future = executor.submit(
            _plan_act_chunk,
            arx,
            policy,
            preprocess,
            postprocess,
            arm_side,
            action_dim,
            chunk_length,
        )

    def consume_prefetch(force_wait: bool) -> tuple[list[np.ndarray], float, int | None] | None:
        nonlocal prefetch_future, prefetch_target_step
        if prefetch_future is None:
            return None
        if not force_wait and not prefetch_future.done():
            return None
        result = prefetch_future.result()
        target_step = prefetch_target_step
        prefetch_future = None
        prefetch_target_step = None
        return result[0], result[1], target_step

    warmup_actions, warmup_latency_ms = _plan_act_chunk(
        arx,
        policy,
        preprocess,
        postprocess,
        arm_side,
        action_dim,
        chunk_length,
    )
    pending_actions.extend(warmup_actions)

    print(
        f"ACT chunked deployment started: side={arm_side}, cameras={rgb_camera_keys}, "
        f"depth_cameras={depth_camera_keys}, action_dim={action_dim}, chunk_length={chunk_length}, "
        f"model_chunk_length={model_chunk_length}, hz={hz:.2f}, "
        f"replan_interval={replan_interval}, "
        f"chunk_method={effective_chunk_method or 'disabled'}, dry_run={dry_run}, "
        f"async_prefetch={async_prefetch}, prefetch_lead_steps={prefetch_lead_steps}, "
        f"warmup_latency_ms={warmup_latency_ms:.1f}"
    )
    step_idx = 0
    steps_since_replan = 0
    period = 1.0 / max(hz, 1e-6)

    try:
        while max_steps <= 0 or step_idx < max_steps:
            t0 = time.perf_counter()
            latency_ms = 0.0
            plan_source = "queue"
            replanned_target_step: int | None = None
            should_replan = not pending_actions
            if not should_replan and replan_interval > 0 and steps_since_replan >= replan_interval:
                should_replan = True

            if (
                executor is not None
                and prefetch_future is None
                and pending_actions
                and replan_interval > 0
            ):
                steps_until_replan = replan_interval - steps_since_replan
                if 0 < steps_until_replan <= prefetch_lead_steps:
                    submit_prefetch(step_idx + steps_until_replan)

            if should_replan:
                prefetched = consume_prefetch(force_wait=True)
                if prefetched is not None:
                    new_actions, latency_ms, replanned_target_step = prefetched
                    plan_source = "prefetch"
                else:
                    new_actions, latency_ms = _plan_act_chunk(
                        arx,
                        policy,
                        preprocess,
                        postprocess,
                        arm_side,
                        action_dim,
                        chunk_length,
                    )
                    plan_source = "sync"

                merged_actions = merge_action_chunks(
                    list(pending_actions),
                    new_actions,
                    chunk_method=effective_chunk_method if (
                        pending_actions and effective_chunk_method
                    ) else "replace",
                )
                pending_actions = deque(merged_actions)
                steps_since_replan = 0

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
                        "plan_source": plan_source,
                        "prefetch_target_step": replanned_target_step,
                        "arm_side": arm_side if action_dim <= 7 else "both",
                        "action_type": "joint",
                        "action": action.tolist(),
                    }
                )
                print(
                    f"[step {step_idx:05d}] action={action.round(4)} "
                    f"latency_ms={latency_ms:.1f} plan_source={plan_source}"
                )
            else:
                arx.step_smooth_joint(control_payload)

            if plan_source != "queue":
                print(
                    f"[step {step_idx:05d}] replanned via {plan_source} "
                    f"latency_ms={latency_ms:.1f} target_step={replanned_target_step}"
                )

            elapsed = time.perf_counter() - t0
            time.sleep(max(0.0, period - elapsed))
            step_idx += 1
            steps_since_replan += 1
    finally:
        if prefetch_future is not None:
            prefetch_future.cancel()
        if executor is not None:
            executor.shutdown(wait=False)
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
            dir="testdata",
            video=True,
            video_fps=30.0,
            video_name="testACT",
            img_size=(640, 480),
        )
    try:
        if arx is not None:
            arx.reset()
            arx.step_lift(12.5)
        run_act(
            arx=arx,
            model_path="/home/arx/Arx_Lift2s/Deployment/models/push_away_act",
            arm_side="right",
            hz=20.0,
            chunk_size=30,
            replan_interval=30,
            chunk_method="replace",
            max_steps=150,
            dry_run=dry_run,
            dataset_root=dataset_root,
            episode_index=0,
            dry_run_dir="dryrun_records/push_away/act_chunked",
            show_plot=False,
            async_prefetch=True,
            prefetch_lead_steps=4,
            temporal_ensemble_coeff=0.01,
        )
    finally:
        if arx is not None:
            arx.close()


if __name__ == "__main__":
    main()
