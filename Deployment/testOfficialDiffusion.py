#!/usr/bin/env python3
"""Official-style Diffusion deployment helper for ARX."""

from __future__ import annotations

import sys
import time
from pathlib import Path

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
    infer_action_dim,
    infer_visual_feature_keys,
    resolve_pretrained_model_path,
    run_open_loop_dryrun,
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
    rgb_camera_keys, depth_camera_keys = infer_visual_feature_keys(policy)
    action_dim = infer_action_dim(policy)
    return policy, preprocess, postprocess, rgb_camera_keys, depth_camera_keys, action_dim


def run_official_diffusion(
    arx: ARXRobotEnv | None,
    model_path: str,
    arm_side: str = "right",
    hz: float = 10.0,
    max_steps: int = 0,
    device: str = "cuda",
    dry_run: bool = False,
    dry_run_dir: str = "dryrun_records",
    dataset_root: str | None = None,
    episode_index: int = 0,
    show_plot: bool = False,
):
    """Run diffusion with the official single-step `select_action()` logic."""
    if dataset_root is not None and not dry_run:
        raise ValueError(
            "`dataset_root` is only supported when `dry_run=True`.")

    if dry_run and dataset_root:
        return run_open_loop_dryrun(
            model_path=model_path,
            dataset_root=dataset_root,
            episode_index=episode_index,
            policy_type="diffusion",
            max_steps=max_steps,
            device=device,
            output_dir=dry_run_dir,
            show_plot=show_plot,
            rollout_mode="official",
        )

    if arx is None:
        raise ValueError(
            "`arx` must be provided when `dataset_root` is not set.")

    policy, preprocess, postprocess, rgb_camera_keys, depth_camera_keys, action_dim = load_diffusion_policy(
        model_path,
        device=device,
    )
    policy.reset()

    print(
        f"Official diffusion deployment started: side={arm_side}, "
        f"rgb_cameras={rgb_camera_keys}, depth_cameras={depth_camera_keys}, "
        f"action_dim={action_dim}, n_obs_steps={policy.config.n_obs_steps}, "
        f"n_action_steps={policy.config.n_action_steps}, hz={hz:.2f}, dry_run={dry_run}"
    )

    step_idx = 0
    period = 1.0 / max(hz, 1e-6)

    while max_steps <= 0 or step_idx < max_steps:
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
            action = policy.select_action(batch)
            action = postprocess(action)

        step_action = unwrap_action_sequence(
            action,
            action_dim=action_dim,
            max_action_steps=1,
        )[0]
        control_payload = build_control_payload(step_action, arm_side)

        if dry_run:
            print(f"[step {step_idx:05d}] action={step_action.round(4)}")
        else:
            arx.step_smooth_joint(control_payload)

        elapsed = time.perf_counter() - t0
        time.sleep(max(0.0, period - elapsed))
        step_idx += 1


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
        run_official_diffusion(
            arx=arx,
            model_path="/home/arx/Arx_Lift2s/Deployment/models/push_away_diffusion",
            arm_side="right",
            hz=20.0,
            max_steps=150,
            dry_run=dry_run,
            dataset_root=dataset_root,
            episode_index=0,
            dry_run_dir="dryrun_records/push_away_diffuion",
            show_plot=False,
        )
    finally:
        if arx is not None:
            arx.close()


if __name__ == "__main__":
    main()
