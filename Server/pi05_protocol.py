#!/usr/bin/env python3
"""Serialization helpers for ARX PI05 client/server deployment."""

from __future__ import annotations

import base64
import json
from io import BytesIO
from typing import Any

import numpy as np


def _load_cv2():
    try:
        import cv2
    except ImportError:
        return None
    return cv2


def _load_pil():
    try:
        from PIL import Image
    except ImportError:
        return None
    return Image


def _b64encode(data: bytes) -> str:
    return base64.b64encode(data).decode("ascii")


def _b64decode(data: str) -> bytes:
    return base64.b64decode(data.encode("ascii"))


def _encode_color_with_pil(image_bgr: np.ndarray, codec: str) -> bytes:
    Image = _load_pil()
    if Image is None:
        raise ImportError("Need either `cv2` or `Pillow` to encode color images.")
    rgb = np.asarray(image_bgr, dtype=np.uint8)[..., ::-1]
    pil_image = Image.fromarray(rgb, mode="RGB")
    buffer = BytesIO()
    fmt = "JPEG" if codec == "jpg" else "PNG"
    save_kwargs = {"quality": 95} if fmt == "JPEG" else {}
    pil_image.save(buffer, format=fmt, **save_kwargs)
    return buffer.getvalue()


def _decode_color_with_pil(payload: bytes) -> np.ndarray:
    Image = _load_pil()
    if Image is None:
        raise ImportError("Need either `cv2` or `Pillow` to decode color images.")
    buffer = BytesIO(payload)
    rgb = np.asarray(Image.open(buffer).convert("RGB"), dtype=np.uint8)
    return rgb[..., ::-1].copy()


def _encode_depth_png(depth: np.ndarray) -> bytes:
    depth = np.asarray(depth)
    if depth.ndim != 2:
        raise ValueError(f"Depth image must be 2D, got shape {depth.shape}")

    cv2 = _load_cv2()
    if cv2 is not None:
        ok, encoded = cv2.imencode(".png", depth)
        if not ok:
            raise RuntimeError("cv2 failed to encode depth PNG")
        return encoded.tobytes()

    Image = _load_pil()
    if Image is None:
        raise ImportError("Need either `cv2` or `Pillow` to encode depth images.")
    buffer = BytesIO()
    Image.fromarray(depth).save(buffer, format="PNG")
    return buffer.getvalue()


def _decode_depth_png(payload: bytes, dtype: str) -> np.ndarray:
    cv2 = _load_cv2()
    if cv2 is not None:
        array = np.frombuffer(payload, dtype=np.uint8)
        image = cv2.imdecode(array, cv2.IMREAD_UNCHANGED)
        if image is None:
            raise RuntimeError("cv2 failed to decode depth PNG")
        return np.asarray(image, dtype=np.dtype(dtype))

    Image = _load_pil()
    if Image is None:
        raise ImportError("Need either `cv2` or `Pillow` to decode depth images.")
    buffer = BytesIO(payload)
    return np.asarray(Image.open(buffer), dtype=np.dtype(dtype))


def decode_color_image(payload: dict[str, Any]) -> np.ndarray:
    codec = str(payload.get("codec", "jpg")).lower()
    if codec not in {"jpg", "png"}:
        raise ValueError(f"Unsupported color codec: {codec}")

    raw = _b64decode(str(payload["data"]))
    cv2 = _load_cv2()
    if cv2 is not None:
        array = np.frombuffer(raw, dtype=np.uint8)
        image = cv2.imdecode(array, cv2.IMREAD_COLOR)
        if image is None:
            raise RuntimeError("cv2 failed to decode color image")
        return image
    return _decode_color_with_pil(raw)


def decode_depth_image(payload: dict[str, Any]) -> np.ndarray:
    codec = str(payload.get("codec", "png")).lower()
    if codec != "png":
        raise ValueError(f"Unsupported depth codec: {codec}")
    return _decode_depth_png(_b64decode(str(payload["data"])), str(payload["dtype"]))


def decode_frames(payload: dict[str, Any]) -> dict[str, np.ndarray]:
    frames: dict[str, np.ndarray] = {}
    for key, item in payload.items():
        kind = str(item.get("kind", ""))
        if kind == "color":
            frames[key] = decode_color_image(item)
        elif kind == "depth":
            frames[key] = decode_depth_image(item)
        else:
            raise ValueError(f"Unknown frame payload kind for {key!r}: {kind!r}")
    return frames


def serialize_action_response(
    *,
    request_id: str,
    actions: list[np.ndarray],
    action_dim: int,
    latency_ms: float,
    model_chunk_length: int,
) -> dict[str, Any]:
    return {
        "request_id": request_id,
        "action_dim": int(action_dim),
        "model_chunk_length": int(model_chunk_length),
        "latency_ms": float(latency_ms),
        "actions": [np.asarray(action, dtype=np.float32).reshape(-1).tolist() for action in actions],
    }


def dumps_json(payload: dict[str, Any]) -> bytes:
    return json.dumps(payload, ensure_ascii=False).encode("utf-8")


def loads_json(payload: bytes) -> dict[str, Any]:
    return json.loads(payload.decode("utf-8"))
