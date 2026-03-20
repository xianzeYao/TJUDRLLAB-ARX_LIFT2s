#!/usr/bin/env python3
"""PI05 deployment client for ARX robot."""

from __future__ import annotations

import sys
import time
import urllib.error
import urllib.request
from collections import deque
from pathlib import Path
from typing import Any

import numpy as np


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
    merge_action_chunks,
    normalize_replan_settings,
    resolve_chunk_length,
)
from pi05_protocol import (  # noqa: E402
    build_infer_request,
    decode_action_response,
    loads_json,
    dumps_json,
)


DEFAULT_SERVER_URL = "http://172.28.102.11:8005"


def _http_get_json(url: str, timeout_s: float) -> dict[str, Any]:
    request = urllib.request.Request(url=url, method="GET")
    with urllib.request.urlopen(request, timeout=timeout_s) as response:
        return loads_json(response.read())


def _http_post_json(url: str, payload: dict[str, Any], timeout_s: float) -> dict[str, Any]:
    data = dumps_json(payload)
    request = urllib.request.Request(
        url=url,
        data=data,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=timeout_s) as response:
        return loads_json(response.read())


def fetch_server_meta(server_url: str, timeout_s: float = 10.0) -> dict[str, Any]:
    return _http_get_json(f"{server_url.rstrip('/')}/health", timeout_s=timeout_s)


def request_server_actions(
    server_url: str,
    *,
    frames: dict[str, np.ndarray],
    status: dict[str, Any],
    arm_side: str,
    task: str | None,
    rgb_camera_keys: list[str],
    depth_camera_keys: list[str],
    max_action_steps: int,
    rgb_codec: str,
    timeout_s: float,
) -> tuple[list[np.ndarray], dict[str, Any]]:
    request = build_infer_request(
        frames=frames,
        status=status,
        arm_side=arm_side,
        task=task,
        rgb_camera_keys=rgb_camera_keys,
        depth_camera_keys=depth_camera_keys,
        max_action_steps=max_action_steps,
        rgb_codec=rgb_codec,
    )
    response = _http_post_json(
        f"{server_url.rstrip('/')}/infer",
        request,
        timeout_s=timeout_s,
    )
    return decode_action_response(response), response


def run_pi05_client(
    *,
    arx,
    server_url: str,
    arm_side: str = "right",
    task: str | None = None,
    hz: float = 4.0,
    chunk_size: int | None = None,
    replan_interval: int = 0,
    chunk_method: str = "replace",
    max_steps: int = 0,
    request_timeout_s: float = 10.0,
    rgb_codec: str = "jpg",
    dry_run: bool = False,
) -> None:
    server_meta = fetch_server_meta(server_url, timeout_s=request_timeout_s)
    rgb_camera_keys = list(server_meta.get("rgb_camera_keys", []))
    depth_camera_keys = list(server_meta.get("depth_camera_keys", []))
    action_dim = int(server_meta["action_dim"])
    model_chunk_length = int(server_meta["model_chunk_length"])
    task_text = task or str(server_meta.get(
        "default_task") or "gravity_single4IL")

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

    pending_actions: deque[np.ndarray] = deque()
    step_idx = 0
    steps_since_replan = 0
    period = 1.0 / max(hz, 1e-6)

    print(
        f"PI05 client deployment started: server={server_url}, side={arm_side}, "
        f"task={task_text!r}, rgb={rgb_camera_keys}, depth={depth_camera_keys}, "
        f"action_dim={action_dim}, chunk_length={chunk_length}, "
        f"model_chunk_length={model_chunk_length}, hz={hz:.2f}, "
        f"replan_interval={replan_interval}, chunk_method={effective_chunk_method or 'disabled'}, "
        f"dry_run={dry_run}"
    )

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
            new_actions, response = request_server_actions(
                server_url,
                frames=frames,
                status=status,
                arm_side=arm_side,
                task=task_text,
                rgb_camera_keys=rgb_camera_keys,
                depth_camera_keys=depth_camera_keys,
                max_action_steps=chunk_length,
                rgb_codec=rgb_codec,
                timeout_s=request_timeout_s,
            )
            merged_actions = merge_action_chunks(
                list(pending_actions),
                new_actions,
                chunk_method=effective_chunk_method if (
                    pending_actions and effective_chunk_method) else "replace",
            )
            pending_actions = deque(merged_actions)
            steps_since_replan = 0
            latency_ms = float(response.get("latency_ms", 0.0))

        if not pending_actions:
            raise RuntimeError(
                f"No server action available at step {step_idx}")

        action = pending_actions.popleft()
        control_payload = build_control_payload(action, arm_side)

        if dry_run:
            print(
                f"[step {step_idx:05d}] action={action.round(4)} latency_ms={latency_ms:.1f}"
            )
        else:
            arx.step_smooth_joint(control_payload)

        elapsed = time.perf_counter() - t0
        time.sleep(max(0.0, period - elapsed))
        step_idx += 1
        steps_since_replan += 1


def infer_camera_keys_from_server(
    server_url: str,
    request_timeout_s: float,
) -> tuple[tuple[str, ...], dict[str, Any]]:
    server_meta = fetch_server_meta(server_url, timeout_s=request_timeout_s)
    camera_keys = tuple(
        dict.fromkeys(
            list(server_meta.get("rgb_camera_keys", [])) +
            list(server_meta.get("depth_camera_keys", []))
        )
    )
    if not camera_keys:
        raise RuntimeError(
            f"Server {server_url} did not provide any camera keys.")
    return camera_keys, server_meta


def main() -> None:
    try:
        camera_keys, _server_meta = infer_camera_keys_from_server(
            server_url=DEFAULT_SERVER_URL,
            request_timeout_s=10.0,
        )
        
        arx = ARXRobotEnv(
            duration_per_step=1.0 / 20.0,
            min_steps=20,
            max_v_xyz=0.25,
            max_a_xyz=0.20,
            max_v_rpy=0.3,
            max_a_rpy=1.00,
            camera_type="all",
            camera_view=camera_keys,
            img_size=(640, 480),
        )
        arx.reset()
        arx.step_lift(14.5)
        run_pi05_client(
            arx=arx,
            server_url=DEFAULT_SERVER_URL,
            arm_side="right",
            task="stack the two paper cups on top of the paper cup closest to the shelf one by one and place the stacked cups on the shelf",
            hz=20,
            chunk_size=50,
            replan_interval=30,
            chunk_method="replace",
            max_steps=500,
            request_timeout_s=10,
            rgb_codec="jpg",
            dry_run=False,
        )
    finally:
        arx.close()
if __name__ == "__main__":
    main()
