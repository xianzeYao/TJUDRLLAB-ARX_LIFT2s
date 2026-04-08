from __future__ import annotations

import ast
import json
import re
import time
from dataclasses import dataclass
from typing import Any, Literal, Mapping, Optional

import numpy as np
from arx_pointing import (
    predict_multi_points_from_multi_image,
    predict_multi_points_from_rgb,
)


DEFAULT_HAND_CAMERA_BY_ARM = {
    "left": "camera_l",
    "right": "camera_r",
}


@dataclass
class TaskCheckResult:
    status: str
    description: str
    third_description: Optional[str] = None
    wrist_description: Optional[str] = None
    third_status: Optional[str] = None
    wrist_status: Optional[str] = None


def _print_decision(title: str, parts: list[tuple[str, Any]]) -> None:
    chunks = []
    for key, value in parts:
        if value is None:
            continue
        chunks.append(f"{key}={value}")
    print(f"{title} " + " | ".join(chunks))


def _preprocess_text(text: str) -> str:
    text = re.sub(r"```(?:json|python|html)?\n?(.*?)\n?```",
                  r"\1", text, flags=re.DOTALL)
    match = re.search(r"<answer>(.*?)</answer>", text,
                      flags=re.DOTALL | re.IGNORECASE)
    if match:
        text = match.group(1)
    return text.strip()


def _decode_json_result(text: str) -> dict[str, Any]:
    cleaned = _preprocess_text(text)
    match = re.search(r"(\{.*\})", cleaned, re.DOTALL)
    raw_json = match.group(1) if match else cleaned

    parsed: Optional[dict[str, Any]] = None
    try:
        obj = json.loads(raw_json)
        if isinstance(obj, dict):
            parsed = obj
    except (json.JSONDecodeError, TypeError):
        try:
            obj = ast.literal_eval(raw_json)
            if isinstance(obj, dict):
                parsed = obj
        except (SyntaxError, ValueError):
            parsed = None

    if parsed is not None:
        normalized = {str(k).lower(): v for k, v in parsed.items()}
        status = str(normalized.get("status", "fail")).lower()
        if status == "failed":
            status = "fail"
        if status not in {"success", "fail"}:
            status = "fail"
        return {
            "status": status,
            "description": str(normalized.get("description", cleaned)),
        }

    lower = cleaned.lower()
    status = "fail"
    if "success" in lower:
        status = "success"
    elif "fail" in lower or "failure" in lower or "error" in lower:
        status = "fail"
    return {
        "status": status,
        "description": cleaned,
    }


def _resolve_hand_camera_key(
    arm: str,
    hand_camera_by_arm: Optional[Mapping[str, str]] = None,
) -> str:
    camera_map = dict(DEFAULT_HAND_CAMERA_BY_ARM)
    if hand_camera_by_arm:
        camera_map.update(hand_camera_by_arm)
    hand_camera_view = camera_map.get(arm)
    if not hand_camera_view:
        raise ValueError(f"no hand camera configured for arm={arm!r}")
    return f"{hand_camera_view}_color"


def capture_hand_check_frame(
    arx,
    arm: str,
    *,
    hand_camera_by_arm: Optional[Mapping[str, str]] = None,
    target_size: tuple[int, int] = (640, 480),
    settle_s: float = 1.0,
    max_retries: int = 1,
    retry_sleep_s: float = 0.2,
) -> tuple[np.ndarray, str]:
    hand_camera_key = _resolve_hand_camera_key(
        arm,
        hand_camera_by_arm=hand_camera_by_arm,
    )

    if settle_s > 0:
        time.sleep(settle_s)

    frames = None
    for _ in range(max(1, int(max_retries))):
        frames = arx.get_camera(target_size=target_size, return_status=False)
        hand_image = frames.get(hand_camera_key) if frames else None
        if hand_image is not None:
            return hand_image, hand_camera_key
        time.sleep(max(0.0, float(retry_sleep_s)))

    available = sorted(frames.keys()) if frames else []
    raise RuntimeError(
        "failed to fetch hand check frame, "
        f"need {hand_camera_key}, got {available}"
    )


def capture_third_check_frame(
    arx,
    *,
    third_camera_view: str = "camera_h",
    target_size: tuple[int, int] = (640, 480),
    settle_s: float = 1.0,
    max_retries: int = 1,
    retry_sleep_s: float = 0.2,
) -> tuple[np.ndarray, str]:
    third_camera_key = f"{third_camera_view}_color"

    if settle_s > 0:
        time.sleep(settle_s)

    frames = None
    for _ in range(max(1, int(max_retries))):
        frames = arx.get_camera(target_size=target_size, return_status=False)
        third_image = frames.get(third_camera_key) if frames else None
        if third_image is not None:
            return third_image, third_camera_key
        time.sleep(max(0.0, float(retry_sleep_s)))

    available = sorted(frames.keys()) if frames else []
    raise RuntimeError(
        "failed to fetch third-person check frame, "
        f"need {third_camera_key}, got {available}"
    )


def capture_task_check_frames(
    arx,
    arm: str,
    *,
    third_camera_view: str = "camera_h",
    hand_camera_by_arm: Optional[Mapping[str, str]] = None,
    target_size: tuple[int, int] = (640, 480),
    settle_s: float = 1.0,
    max_retries: int = 1,
    retry_sleep_s: float = 0.2,
) -> tuple[np.ndarray, np.ndarray, str, str]:
    hand_camera_key = _resolve_hand_camera_key(
        arm,
        hand_camera_by_arm=hand_camera_by_arm,
    )
    third_camera_key = f"{third_camera_view}_color"

    if settle_s > 0:
        time.sleep(settle_s)

    for _ in range(max(1, int(max_retries))):
        frames = arx.get_camera(target_size=target_size, return_status=False)
        hand_image = frames.get(hand_camera_key) if frames else None
        third_image = frames.get(third_camera_key) if frames else None
        if hand_image is not None and third_image is not None:
            return hand_image, third_image, hand_camera_key, third_camera_key
        time.sleep(max(0.0, float(retry_sleep_s)))

    available = sorted(frames.keys()) if frames else []
    raise RuntimeError(
        "failed to fetch task check frames, "
        f"need {hand_camera_key} and {third_camera_key}, got {available}"
    )


def _build_multi_image_task_check_prompt(
    *,
    pick_prompt: str,
) -> str:
    lines = [
        "You are checking whether a robot has successfully finished pick.",
        "Image 1 is the active arm camera. Image 2 is the third-person camera.",
        f"Pick target description: {pick_prompt}.",
        "Return only valid JSON with keys:",
        '- "status": one of "success", "fail"',
        '- "description": short evidence-based explanation',
        "Use status=success only when the task is visibly complete. Otherwise use status=fail.",
    ]
    return "\n".join(lines)


def _build_third_person_check_prompt(
    *,
    pick_prompt: str,
) -> str:
    lines = [
        f"Return status=success only if {pick_prompt} is no longer in view.",
    ]
    lines.extend([
        "Return only valid JSON with keys:",
        '- "status": one of "success", "fail"',
        '- "description": short evidence-based explanation',
        "Use status=success only when the image clearly supports success. Otherwise use status=fail.",
    ])
    return "\n".join(lines)


def _build_third_person_delta_check_prompt(
    *,
    pick_prompt: str,
) -> str:
    lines = [
        "Image 1 is before pick. Image 2 is after pick.",
        f"Is one {pick_prompt} missing in image 2 compared with image 1?",
        "Return only valid JSON with keys:",
        '- "status": one of "success", "fail"',
        '- "description": short evidence-based explanation',
        f'Use status=success only when image 2 clearly has one fewer {pick_prompt} than image 1.',
    ]
    return "\n".join(lines)


def _build_wrist_check_prompt(
    *,
    pick_prompt: str,
) -> str:
    lines = [
        f"Return status=success only if the {pick_prompt} object is being held securely by the gripper.",
    ]
    lines.extend([
        "Return only valid JSON with keys:",
        '- "status": one of "success", "fail"',
        '- "description": short evidence-based explanation',
        "Use status=success only when the image clearly supports success. Otherwise use status=fail.",
    ])
    return "\n".join(lines)


def _infer_single_image_status(
    *,
    image: np.ndarray,
    prompt: str,
    base_url: str,
    model_name: str,
    api_key: str,
    temperature: float,
    top_p: float,
    seed: int,
    max_tokens: int,
) -> TaskCheckResult:
    _, raw = predict_multi_points_from_rgb(
        image=image,
        text_prompt="",
        all_prompt=prompt,
        base_url=base_url,
        model_name=model_name,
        api_key=api_key,
        assume_bgr=False,
        temperature=temperature,
        top_p=top_p,
        seed=seed,
        max_tokens=max_tokens,
        return_raw=True,
    )
    raw = raw or ""
    parsed = _decode_json_result(raw)
    return TaskCheckResult(
        status=str(parsed["status"]),
        description=str(parsed["description"]),
    )


def _infer_third_person_status(
    *,
    image: np.ndarray,
    pick_prompt: str,
    base_url: str,
    model_name: str,
    api_key: str,
    temperature: float,
    top_p: float,
    seed: int,
    max_tokens: int,
) -> TaskCheckResult:
    return _infer_single_image_status(
        image=image,
        prompt=_build_third_person_check_prompt(
            pick_prompt=pick_prompt,
        ),
        base_url=base_url,
        model_name=model_name,
        api_key=api_key,
        temperature=temperature,
        top_p=top_p,
        seed=seed,
        max_tokens=max_tokens,
    )


def predict_third_person_target_check(
    *,
    third_image: np.ndarray,
    pick_prompt: str,
    base_url: str = "http://172.28.102.11:22002/v1",
    model_name: str = "Embodied-R1.5-SFT-0128",
    api_key: str = "EMPTY",
    temperature: float = 0.0,
    top_p: float = 0.8,
    seed: int = 3407,
    max_tokens: int = 256,
) -> TaskCheckResult:
    result = _infer_third_person_status(
        image=third_image,
        pick_prompt=pick_prompt,
        base_url=base_url,
        model_name=model_name,
        api_key=api_key,
        temperature=temperature,
        top_p=top_p,
        seed=seed,
        max_tokens=max_tokens,
    )
    _print_decision(
        "[task_check][第三视角]",
        [
            ("target", repr(pick_prompt)),
            ("result", result.status),
            ("rule", "不在第三视角里才算success"),
            ("desc", result.description),
        ],
    )
    return result


def predict_third_person_delta_check(
    *,
    before_image: np.ndarray,
    after_image: np.ndarray,
    pick_prompt: str,
    base_url: str = "http://172.28.102.11:22002/v1",
    model_name: str = "Embodied-R1.5-SFT-0128",
    api_key: str = "EMPTY",
    temperature: float = 0.0,
    top_p: float = 0.8,
    seed: int = 3407,
    max_tokens: int = 256,
) -> TaskCheckResult:
    prompt = _build_third_person_delta_check_prompt(
        pick_prompt=pick_prompt,
    )
    _, raw = predict_multi_points_from_multi_image(
        images=[before_image, after_image],
        text_prompt="",
        all_prompt=prompt,
        base_url=base_url,
        model_name=model_name,
        api_key=api_key,
        assume_bgr=(False, False),
        temperature=temperature,
        top_p=top_p,
        seed=seed,
        max_tokens=max_tokens,
        return_raw=True,
    )
    raw = raw or ""
    parsed = _decode_json_result(raw)
    result = TaskCheckResult(
        status=str(parsed["status"]),
        description=str(parsed["description"]),
    )
    _print_decision(
        "[task_check][第三视角对比]",
        [
            ("target", repr(pick_prompt)),
            ("result", result.status),
            ("rule", "第二张比第一张少一个目标才算success"),
            ("desc", result.description),
        ],
    )
    return result


def predict_wrist_target_check(
    *,
    hand_image: np.ndarray,
    pick_prompt: str,
    base_url: str = "http://172.28.102.11:22002/v1",
    model_name: str = "Embodied-R1.5-SFT-0128",
    api_key: str = "EMPTY",
    temperature: float = 0.0,
    top_p: float = 0.8,
    seed: int = 3407,
    max_tokens: int = 256,
) -> TaskCheckResult:
    result = _infer_single_image_status(
        image=hand_image,
        prompt=_build_wrist_check_prompt(
            pick_prompt=pick_prompt,
        ),
        base_url=base_url,
        model_name=model_name,
        api_key=api_key,
        temperature=temperature,
        top_p=top_p,
        seed=seed,
        max_tokens=max_tokens,
    )
    _print_decision(
        "[task_check][腕部]",
        [
            ("target", repr(pick_prompt)),
            ("result", result.status),
            ("rule", "被夹爪稳定抓住才算success"),
            ("desc", result.description),
        ],
    )
    return result


def predict_task_completion(
    *,
    hand_image: np.ndarray,
    third_image: np.ndarray,
    pick_prompt: str,
    item_type: str,
    mode: Literal["single_image", "multi_image"] = "single_image",
    base_url: str = "http://172.28.102.11:22002/v1",
    model_name: str = "Embodied-R1.5-SFT-0128",
    api_key: str = "EMPTY",
    temperature: float = 0.0,
    top_p: float = 0.8,
    seed: int = 3407,
    max_tokens: int = 256,
) -> TaskCheckResult:
    if mode == "multi_image":
        prompt = _build_multi_image_task_check_prompt(
            pick_prompt=pick_prompt,
        )
        _, raw = predict_multi_points_from_multi_image(
            images=[hand_image, third_image],
            text_prompt="",
            all_prompt=prompt,
            base_url=base_url,
            model_name=model_name,
            api_key=api_key,
            assume_bgr=(False, False),
            temperature=temperature,
            top_p=top_p,
            seed=seed,
            max_tokens=max_tokens,
            return_raw=True,
        )
        raw = raw or ""
        parsed = _decode_json_result(raw)
        result = TaskCheckResult(
            status=str(parsed["status"]),
            description=str(parsed["description"]),
        )
        _print_decision(
            "[task_check][multi_image]",
            [
                ("target", repr(pick_prompt)),
                ("result", result.status),
                ("rule", "双图联合直接输出success/fail"),
                ("desc", result.description),
            ],
        )
        return result

    third_result = _infer_third_person_status(
        image=third_image,
        pick_prompt=pick_prompt,
        base_url=base_url,
        model_name=model_name,
        api_key=api_key,
        temperature=temperature,
        top_p=top_p,
        seed=seed,
        max_tokens=max_tokens,
    )
    wrist_result = _infer_single_image_status(
        image=hand_image,
        prompt=_build_wrist_check_prompt(
            pick_prompt=pick_prompt,
        ),
        base_url=base_url,
        model_name=model_name,
        api_key=api_key,
        temperature=temperature,
        top_p=top_p,
        seed=seed,
        max_tokens=max_tokens,
    )
    final_status = (
        "success"
        if third_result.status == "success" or wrist_result.status == "success"
        else "fail"
    )
    combined_description = (
        f"third={third_result.description}; wrist={wrist_result.description}"
    )
    _print_decision(
        "[task_check][single_image]",
        [
            ("target", repr(pick_prompt)),
            ("third", third_result.status),
            ("wrist", wrist_result.status),
            ("final", final_status),
            ("rule", "third OR wrist"),
        ],
    )
    return TaskCheckResult(
        status=final_status,
        description=combined_description,
        third_description=third_result.description,
        wrist_description=wrist_result.description,
        third_status=third_result.status,
        wrist_status=wrist_result.status,
    )


def run_task_completion_check(
    arx,
    arm: str,
    *,
    pick_prompt: str,
    item_type: str,
    mode: Literal["single_image", "multi_image"] = "single_image",
    third_camera_view: str = "camera_h",
    hand_camera_by_arm: Optional[Mapping[str, str]] = None,
    target_size: tuple[int, int] = (640, 480),
    settle_s: float = 1.0,
    max_retries: int = 1,
    retry_sleep_s: float = 0.2,
    base_url: str = "http://172.28.102.11:22002/v1",
    model_name: str = "Embodied-R1.5-SFT-0128",
    api_key: str = "EMPTY",
    temperature: float = 0.0,
    top_p: float = 0.8,
    seed: int = 3407,
    max_tokens: int = 256,
) -> TaskCheckResult:
    hand_image, third_image, hand_camera_key, third_camera_key = capture_task_check_frames(
        arx,
        arm,
        third_camera_view=third_camera_view,
        hand_camera_by_arm=hand_camera_by_arm,
        target_size=target_size,
        settle_s=settle_s,
        max_retries=max_retries,
        retry_sleep_s=retry_sleep_s,
    )
    return predict_task_completion(
        hand_image=hand_image,
        third_image=third_image,
        pick_prompt=pick_prompt,
        item_type=item_type,
        mode=mode,
        base_url=base_url,
        model_name=model_name,
        api_key=api_key,
        temperature=temperature,
        top_p=top_p,
        seed=seed,
        max_tokens=max_tokens,
    )
