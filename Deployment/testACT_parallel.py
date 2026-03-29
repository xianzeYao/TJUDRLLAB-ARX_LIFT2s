#!/usr/bin/env python3
"""
Optimized ACT deployment with parallel control + inference.

Key optimizations:
1. Async camera capture in background thread
2. Parallel execution of control command and next inference
"""

from __future__ import annotations

import sys
import time
import threading
from collections import deque
from concurrent.futures import Future, ThreadPoolExecutor
from pathlib import Path
from typing import Any

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


# =============================================================================
# Timing Statistics
# =============================================================================
# Async Camera Capture
# =============================================================================


class AsyncCameraCapture:
    """
    Continuously captures camera frames in a background thread.
    Provides the latest frame on demand with minimal latency.
    """

    def __init__(self, arx: ARXRobotEnv, target_size: tuple[int, int]):
        self._arx = arx
        self._target_size = target_size
        self._latest_frame: dict[str, np.ndarray] | None = None
        self._latest_status: dict[str, Any] | None = None
        self._lock = threading.Lock()
        self._running = False
        self._thread: threading.Thread | None = None
        self._capture_count = 0

    def start(self):
        if self._running:
            return
        self._running = True
        self._thread = threading.Thread(target=self._capture_loop, daemon=True)
        self._thread.start()

    def stop(self):
        self._running = False
        if self._thread is not None:
            self._thread.join(timeout=1.0)
            self._thread = None

    def _capture_loop(self):
        while self._running:
            try:
                frames, status = self._arx.get_camera(
                    save_dir=None,
                    video=False,
                    target_size=self._target_size,
                    return_status=True,
                )
                with self._lock:
                    self._latest_frame = frames
                    self._latest_status = status
                    self._capture_count += 1
            except Exception as e:
                print(f"[AsyncCamera] Capture error: {e}")
                time.sleep(0.01)

    def get_latest(self) -> tuple[dict[str, np.ndarray], dict[str, Any], float]:
        """Returns (frames, status, wait_time_ms)"""
        t0 = time.perf_counter()
        # Wait until first frame is available
        while True:
            with self._lock:
                if self._latest_frame is not None:
                    frames = self._latest_frame
                    status = self._latest_status
                    break
            time.sleep(0.001)
        wait_ms = (time.perf_counter() - t0) * 1000.0
        return frames, status, wait_ms

    def get_sync(self) -> tuple[dict[str, np.ndarray], dict[str, Any], float]:
        """Synchronous capture (fallback mode)"""
        t0 = time.perf_counter()
        frames, status = self._arx.get_camera(
            save_dir=None,
            video=False,
            target_size=self._target_size,
            return_status=True,
        )
        elapsed_ms = (time.perf_counter() - t0) * 1000.0
        return frames, status, elapsed_ms


# =============================================================================
# Policy Loading with Optimization Options
# =============================================================================


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
    """Load ACT policy."""
    from lerobot.policies.factory import make_pre_post_processors

    torch_device = torch.device(device if torch.cuda.is_available() else "cpu")
    resolved_model_path = resolve_pretrained_model_path(model_path)
    ACTPolicy = _import_act_policy()
    policy = ACTPolicy.from_pretrained(resolved_model_path).to(torch_device).eval()

    preprocess, postprocess = make_pre_post_processors(
        policy.config,
        resolved_model_path,
        preprocessor_overrides={"device_processor": {"device": str(torch_device)}},
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
        torch_device,
    )


# =============================================================================
# Core Prediction Functions with Timing
# =============================================================================


def _predict_act_chunk_tensor(
    arx: ARXRobotEnv,
    policy,
    preprocess,
    postprocess,
    arm_side: str,
    chunk_length: int | None = None,
    async_camera: AsyncCameraCapture | None = None,
    stats: TimingStats | None = None,
) -> tuple[torch.Tensor, float, dict[str, float]]:
    """Predict action chunk with timing breakdown."""
    timings = {}
    t_start = time.perf_counter()

    # Stage 1: Camera capture
    t0 = time.perf_counter()
    if async_camera is not None:
        frames, status, camera_wait_ms = async_camera.get_latest()
        timings["camera_ms"] = camera_wait_ms
    else:
        frames, status = arx.get_camera(
            save_dir=None,
            video=False,
            target_size=arx.img_size,
            return_status=True,
        )
        timings["camera_ms"] = (time.perf_counter() - t0) * 1000.0

    # Stage 2: Preprocess
    t0 = time.perf_counter()
    payload = build_policy_observation(policy, frames, status, arm_side=arm_side)
    batch = preprocess(dict(payload))
    timings["preprocess_ms"] = (time.perf_counter() - t0) * 1000.0

    # Stage 3: Inference
    t0 = time.perf_counter()
    with torch.inference_mode():
        pred_action = policy.predict_action_chunk(batch)
    timings["inference_ms"] = (time.perf_counter() - t0) * 1000.0

    # Stage 4: Postprocess
    t0 = time.perf_counter()
    pred_action = postprocess(pred_action)
    if not torch.is_tensor(pred_action):
        pred_action = torch.as_tensor(pred_action, dtype=torch.float32)
    while pred_action.ndim > 3:
        pred_action = pred_action[0]
    if pred_action.ndim == 2:
        pred_action = pred_action.unsqueeze(0)
    if chunk_length is not None:
        pred_action = pred_action[:, :chunk_length]
    timings["postprocess_ms"] = (time.perf_counter() - t0) * 1000.0

    total_latency_ms = (time.perf_counter() - t_start) * 1000.0
    if stats is not None:
        stats.record(**timings)

    return pred_action.detach().float(), total_latency_ms, timings


def _plan_act_chunk(
    arx: ARXRobotEnv,
    policy,
    preprocess,
    postprocess,
    arm_side: str,
    action_dim: int,
    chunk_length: int,
    async_camera: AsyncCameraCapture | None = None,
    stats: TimingStats | None = None,
) -> tuple[list[np.ndarray], float, dict[str, float]]:
    """Plan a full action chunk."""
    pred_action, latency_ms, timings = _predict_act_chunk_tensor(
        arx,
        policy,
        preprocess,
        postprocess,
        arm_side,
        chunk_length=chunk_length,
        async_camera=async_camera,
        stats=stats,
    )
    new_actions = unwrap_action_sequence(
        pred_action,
        action_dim=action_dim,
        max_action_steps=chunk_length,
    )
    return new_actions, latency_ms, timings


# =============================================================================
# Main Run Function with Optimizations
# =============================================================================


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
    temporal_ensemble_coeff: float = 0.01,
):
    """Optimized ACT deployment with parallel control + inference.

    Args:
        temporal_ensemble_coeff: 时序集成的指数衰减系数。
            控制历史预测的衰减速度: weight = exp(-coeff * age)
            - 值越小 (如 0.01): 历史预测权重衰减慢，动作更平滑但响应慢
            - 值越大 (如 0.1): 历史预测权重衰减快，响应快但可能抖动
            推荐范围: 0.01 ~ 0.1
    """
    if dataset_root is not None and not dry_run:
        raise ValueError("`dataset_root` is only supported when `dry_run=True`.")

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
        raise ValueError("`arx` must be provided when `dataset_root` is not set.")

    # Load policy
    (
        policy,
        preprocess,
        postprocess,
        expected_keys,
        rgb_camera_keys,
        depth_camera_keys,
        action_dim,
        model_chunk_length,
        torch_device,
    ) = load_act_policy(model_path, device=device)
    del expected_keys

    chunk_length = resolve_chunk_length(model_chunk_length, chunk_size)

    # Initialize async camera capture
    async_camera = AsyncCameraCapture(arx, arx.img_size)
    async_camera.start()

    # Warmup inference
    warmup_start = time.perf_counter()
    for _ in range(3):
        _predict_act_chunk_tensor(
            arx,
            policy,
            preprocess,
            postprocess,
            arm_side,
            chunk_length=chunk_length,
            async_camera=async_camera,
        )
    warmup_time = (time.perf_counter() - warmup_start) * 1000.0
    print(f"[Warmup] {warmup_time:.1f}ms")

    # =========================================================================
    # Temporal Ensemble Mode (with PARALLEL control + inference)
    # =========================================================================
    if temporal_ensemble_coeff is not None:
        ACTTemporalEnsembler = _import_act_temporal_ensembler()
        temporal_ensembler = ACTTemporalEnsembler(
            temporal_ensemble_coeff=float(temporal_ensemble_coeff),
            chunk_size=chunk_length,
        )

        # Initial prediction
        warmup_pred, warmup_latency_ms, _ = _predict_act_chunk_tensor(
            arx,
            policy,
            preprocess,
            postprocess,
            arm_side,
            chunk_length=chunk_length,
            async_camera=async_camera,
        )
        current_action = unwrap_action_sequence(
            temporal_ensembler.update(warmup_pred),
            action_dim=action_dim,
            max_action_steps=1,
        )[0]

        # Pre-compute next action for pipeline startup
        next_pred, _, _ = _predict_act_chunk_tensor(
            arx,
            policy,
            preprocess,
            postprocess,
            arm_side,
            chunk_length=chunk_length,
            async_camera=async_camera,
        )
        next_action = unwrap_action_sequence(
            temporal_ensembler.update(next_pred),
            action_dim=action_dim,
            max_action_steps=1,
        )[0]

        print(
            f"ACT temporal-ensemble (parallel): side={arm_side}, "
            f"chunk={chunk_length}, hz={hz:.1f}, coeff={temporal_ensemble_coeff:.4f}"
        )

        # Thread pools for parallel execution
        control_executor = ThreadPoolExecutor(
            max_workers=1, thread_name_prefix="control"
        )
        inference_executor = ThreadPoolExecutor(
            max_workers=1, thread_name_prefix="inference"
        )

        dry_run_records = []
        step_idx = 0
        period = 1.0 / max(hz, 1e-6)

        def execute_control_task(action_to_execute, arm_side_arg, is_dry_run):
            """Execute control command in separate thread."""
            t0 = time.perf_counter()
            payload = build_control_payload(action_to_execute, arm_side_arg)
            if not is_dry_run:
                arx.step_raw_joint(payload)
            return (time.perf_counter() - t0) * 1000.0

        def predict_next_task():
            """Predict next action in separate thread."""
            pred, lat_ms, tims = _predict_act_chunk_tensor(
                arx,
                policy,
                preprocess,
                postprocess,
                arm_side,
                chunk_length=chunk_length,
                async_camera=async_camera,
            )
            act = unwrap_action_sequence(
                temporal_ensembler.update(pred),
                action_dim=action_dim,
                max_action_steps=1,
            )[0]
            return act, lat_ms, tims

        try:
            while max_steps <= 0 or step_idx < max_steps:
                t_step_start = time.perf_counter()

                # === PARALLEL EXECUTION ===
                # Execute current action AND predict next action simultaneously
                control_future = control_executor.submit(
                    execute_control_task, current_action, arm_side, dry_run
                )
                inference_future = inference_executor.submit(predict_next_task)

                # Record dry run data (non-blocking)
                if dry_run:
                    dry_run_records.append(
                        {
                            "step": step_idx,
                            "wall_time": time.time(),
                            "latency_ms": 0.0,
                            "plan_source": "temporal_ensemble_parallel",
                            "prefetch_target_step": None,
                            "arm_side": arm_side if action_dim <= 7 else "both",
                            "action_type": "joint",
                            "action": current_action.tolist(),
                        }
                    )

                # Wait for both to complete (they run in parallel!)
                control_ms = control_future.result()
                next_action, latency_ms, timings = inference_future.result()

                # Pipeline advance: next becomes current
                current_action = next_action

                elapsed = time.perf_counter() - t_step_start
                time.sleep(max(0.0, period - elapsed))
                step_idx += 1

        finally:
            control_executor.shutdown(wait=False)
            inference_executor.shutdown(wait=False)
            if async_camera is not None:
                async_camera.stop()
            if dry_run and dry_run_records:
                record_path = save_dry_run_records(
                    dry_run_dir, model_path, dry_run_records
                )
                print(f"Dry-run actions saved to: {record_path}")
        return

    # =========================================================================
    # Chunked Mode
    # =========================================================================
    replan_interval, effective_chunk_method = normalize_replan_settings(
        chunk_length=chunk_length,
        replan_interval=replan_interval,
        chunk_method=chunk_method,
    )

    dry_run_records = []
    pending_actions: deque[np.ndarray] = deque()
    prefetch_lead_steps = max(1, int(prefetch_lead_steps))

    executor = (
        ThreadPoolExecutor(max_workers=1, thread_name_prefix="act_prefetch")
        if async_prefetch
        else None
    )
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
            async_camera,
            None,
        )

    def consume_prefetch(
        force_wait: bool,
    ) -> tuple[list[np.ndarray], float, int | None] | None:
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

    # Initial warmup chunk
    warmup_actions, warmup_latency_ms, _ = _plan_act_chunk(
        arx,
        policy,
        preprocess,
        postprocess,
        arm_side,
        action_dim,
        chunk_length,
        async_camera,
    )
    pending_actions.extend(warmup_actions)

    # Start prefetching next chunk
    if executor is not None:
        submit_prefetch(chunk_length)

    print(
        f"ACT chunked: side={arm_side}, chunk={chunk_length}, hz={hz:.1f}, "
        f"replan={replan_interval}"
    )

    step_idx = 0
    steps_since_replan = 0
    period = 1.0 / max(hz, 1e-6)

    try:
        while max_steps <= 0 or step_idx < max_steps:
            t_step_start = time.perf_counter()
            latency_ms = 0.0
            plan_source = "queue"
            replanned_target_step: int | None = None

            # Check if we need to replan
            should_replan = not pending_actions
            if (
                not should_replan
                and replan_interval > 0
                and steps_since_replan >= replan_interval
            ):
                should_replan = True

            # Proactive prefetch scheduling
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
                    new_actions, latency_ms, _ = _plan_act_chunk(
                        arx,
                        policy,
                        preprocess,
                        postprocess,
                        arm_side,
                        action_dim,
                        chunk_length,
                        async_camera,
                    )
                    plan_source = "sync"

                merged_actions = merge_action_chunks(
                    list(pending_actions),
                    new_actions,
                    chunk_method=effective_chunk_method
                    if (pending_actions and effective_chunk_method)
                    else "replace",
                )
                pending_actions = deque(merged_actions)
                steps_since_replan = 0

                # Start next prefetch after replan
                if executor is not None and prefetch_future is None:
                    submit_prefetch(step_idx + len(pending_actions))

            if not pending_actions:
                raise RuntimeError(f"No predicted action available at step {step_idx}")

            action = pending_actions.popleft()

            # Execute control
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
            else:
                arx.step_raw_joint(control_payload)

            elapsed = time.perf_counter() - t_step_start
            time.sleep(max(0.0, period - elapsed))
            step_idx += 1
            steps_since_replan += 1

    finally:
        if prefetch_future is not None:
            prefetch_future.cancel()
        if executor is not None:
            executor.shutdown(wait=False)
        if async_camera is not None:
            async_camera.stop()
        if dry_run and dry_run_records:
            record_path = save_dry_run_records(dry_run_dir, model_path, dry_run_records)
            print(f"Dry-run actions saved to: {record_path}")


# =============================================================================
# Main Entry Point
# =============================================================================


def main() -> None:
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
        arx.reset()
        arx.step_lift(12.5)

        run_act(
            arx=arx,
            model_path="/home/arx/Arx_Lift2s/Deployment/models/push_away_act",
            arm_side="right",
            hz=20.0,
            chunk_size=30,
            max_steps=150,
            temporal_ensemble_coeff=0.01,
        )

    finally:
        arx.close()


if __name__ == "__main__":
    main()
