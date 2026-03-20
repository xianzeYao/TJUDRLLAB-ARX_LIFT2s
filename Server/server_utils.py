#!/usr/bin/env python3
"""Shared helpers for deployment-side inference servers."""

from __future__ import annotations

import time
from typing import Any

import torch

from deployment_utils import (
    build_policy_observation,
    infer_default_task_text,
    load_policy_bundle,
    unwrap_action_sequence,
)
from pi05_protocol import (
    decode_frames,
    serialize_action_response,
)


DEFAULT_PI05_MODEL_PATH = "/data/yxz/arx/train/arx_pi05_action_expert_joint_30k/checkpoints/last/pretrained_model"
# DEFAULT_PI05_MODEL_PATH = "/data/yxz/arx/train/arx_pi05_joint/checkpoints/last/pretrained_model"


class PI05InferenceService:
    def __init__(
        self,
        model_path: str,
        device: str = "cuda",
    ) -> None:
        self.bundle = load_policy_bundle(
            policy_type="pi05",
            model_path=model_path,
            device=device,
        )
        self.model_path = str(model_path)
        self.chunk_length = int(self.bundle.model_chunk_length)
        self.default_task = infer_default_task_text(
            model_path) or "gravity_single4IL"

    def metadata(self) -> dict[str, Any]:
        return {
            "policy_type": "pi05",
            "model_path": self.model_path,
            "device": str(self.bundle.device),
            "action_dim": int(self.bundle.action_dim),
            "model_chunk_length": int(self.bundle.model_chunk_length),
            "server_chunk_length": int(self.chunk_length),
            "rgb_camera_keys": list(self.bundle.rgb_camera_keys),
            "depth_camera_keys": list(self.bundle.depth_camera_keys),
            "default_task": self.default_task,
        }

    def infer(self, request: dict[str, Any]) -> dict[str, Any]:
        request_id = str(request.get("request_id", ""))
        arm_side = str(request.get("arm_side", "right"))
        task_text = str(request.get("task") or self.default_task)
        max_action_steps = request.get("max_action_steps")
        if max_action_steps is None:
            max_action_steps = self.chunk_length
        max_action_steps = max(1, min(int(max_action_steps), self.chunk_length))

        frames = decode_frames(request["frames"])
        status = dict(request["status"])
        payload = build_policy_observation(
            self.bundle.policy,
            frames,
            status,
            arm_side=arm_side,
            task_text=task_text,
        )
        batch = self.bundle.preprocess(dict(payload))

        t0 = time.perf_counter()
        with torch.inference_mode():
            pred_action = self.bundle.policy.predict_action_chunk(batch)
            pred_action = self.bundle.postprocess(pred_action)
        latency_ms = (time.perf_counter() - t0) * 1000.0

        actions = unwrap_action_sequence(
            pred_action,
            action_dim=self.bundle.action_dim,
            max_action_steps=max_action_steps,
        )
        response = serialize_action_response(
            request_id=request_id,
            actions=actions,
            action_dim=self.bundle.action_dim,
            latency_ms=latency_ms,
            model_chunk_length=self.bundle.model_chunk_length,
        )
        response["task"] = task_text
        return response


def create_pi05_inference_service(
    model_path: str = DEFAULT_PI05_MODEL_PATH,
    device: str = "cuda",
) -> PI05InferenceService:
    return PI05InferenceService(
        model_path=model_path,
        device=device,
    )
