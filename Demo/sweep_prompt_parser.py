from __future__ import annotations

import ast
import json
import re
from dataclasses import dataclass
from typing import Any, Optional

try:
    from openai import OpenAI
except ImportError:  # pragma: no cover - runtime dependency may be absent on dev hosts
    OpenAI = None  # type: ignore[assignment]


DEFAULT_SWEEP_GOAL_PROMPT = "paper cup or paper ball or bottle on the floor"


@dataclass(frozen=True)
class SweepPromptPlan:
    raw_request: str
    goal_prompt: str
    intent: str
    reason: str
    matched_alias: Optional[str] = None
    parser_source: str = "model"
    parser_raw: Optional[str] = None
    target_objects: tuple[str, ...] = ()
    used_default_goal_prompt: bool = False


def _normalize_text(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def _preprocess_text(text: str) -> str:
    text = re.sub(r"```(?:json|python|text)?\n?(.*?)\n?```",
                  r"\1", text, flags=re.DOTALL)
    match = re.search(r"<answer>(.*?)</answer>", text,
                      flags=re.DOTALL | re.IGNORECASE)
    if match:
        text = match.group(1)
    return text.strip()


def _decode_json_object(raw: Optional[str]) -> Optional[dict[str, Any]]:
    cleaned = _preprocess_text(raw or "")
    if not cleaned:
        return None

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

    if parsed is None:
        return None
    return {str(k).strip().lower(): v for k, v in parsed.items()}


def _clean_text(value: Any) -> str:
    if value is None:
        return ""
    text = str(value).strip().strip("\"'`")
    text = re.sub(r"\s+", " ", text)
    return text.strip(" ,.;:[]{}()")


def _build_parser_prompt(request: str) -> str:
    return (
        "You convert a human floor-sweeping request into robot grounding targets.\n"
        "Return only one JSON object with this key:\n"
        "{\n"
        '  "sweep_target": "paper cup or bottle on the floor"\n'
        "}\n"
        "Rules:\n"
        '- If the request means generally sweep or clean the floor, set "sweep_target" to "".\n'
        '- If the request names one or more specific objects to sweep, convert them into one short phrase in "sweep_target".\n'
        '- Combine multiple objects into a single prompt with "or".\n'
        "- Output valid JSON only.\n"
        f'Request: "{request}"'
    )


def _parse_with_model(
    request: str,
    *,
    base_url: str,
    model_name: str,
    api_key: str,
) -> SweepPromptPlan:
    if OpenAI is None:
        raise RuntimeError("openai package is not installed")

    client = OpenAI(base_url=base_url, api_key=api_key)
    resp = client.chat.completions.create(
        model=model_name,
        messages=[{
            "role": "user",
            "content": _build_parser_prompt(request),
        }],
        max_tokens=256,
        temperature=0.0,
        top_p=0.9,
        seed=3407,
    )
    raw = resp.choices[0].message.content or ""
    parsed = _decode_json_object(raw)
    if parsed is None:
        raise RuntimeError(f"failed to parse JSON from parser output: {raw!r}")

    sweep_target = _clean_text(
        parsed.get("sweep_target")
        or parsed.get("goal_prompt")
        or parsed.get("goal")
        or parsed.get("target_prompt")
        or parsed.get("sweep_prompt")
    )
    reason = "Resolved sweep request with model parser."

    if not sweep_target:
        sweep_target = DEFAULT_SWEEP_GOAL_PROMPT

    if sweep_target.lower() == DEFAULT_SWEEP_GOAL_PROMPT:
        return SweepPromptPlan(
            raw_request=request,
            goal_prompt=sweep_target,
            intent="sweep_floor",
            reason=reason,
            parser_source="model",
            parser_raw=raw,
            used_default_goal_prompt=True,
        )

    return SweepPromptPlan(
        raw_request=request,
        goal_prompt=sweep_target,
        intent="sweep_object",
        reason=reason,
        parser_source="model",
        parser_raw=raw,
        used_default_goal_prompt=False,
    )


def parse_human_sweep_request(
    request: str,
    *,
    base_url: str = "http://172.28.102.11:22002/v1",
    model_name: str = "Embodied-R1.5-SFT-0128",
    api_key: str = "EMPTY",
) -> SweepPromptPlan:
    raw_request = _normalize_text(request)
    if not raw_request:
        raise ValueError("request must be non-empty")

    return _parse_with_model(
        raw_request,
        base_url=base_url,
        model_name=model_name,
        api_key=api_key,
    )


__all__ = [
    "DEFAULT_SWEEP_GOAL_PROMPT",
    "SweepPromptPlan",
    "parse_human_sweep_request",
]
