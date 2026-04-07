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


DEFAULT_NAV_WAYPOINT_PROMPT = "a brown coaster on the floor"
DEFAULT_PLACE_TARGET_PROMPT = "the blue plate"


@dataclass
class ShelfPromptTask:
    nav_pick_prompt: str
    nav_waypoint_prompt: str
    place_target_prompt: str
    pick_target: str
    used_default_nav_waypoint_prompt: bool
    used_default_place_target_prompt: bool


@dataclass
class ShelfPromptPlan:
    raw_request: str
    tasks: list[ShelfPromptTask]
    parser_source: str
    parser_raw: Optional[str] = None

    @property
    def nav_pick_prompt(self) -> str:
        return self.tasks[0].nav_pick_prompt if self.tasks else ""

    @property
    def nav_waypoint_prompt(self) -> str:
        return self.tasks[0].nav_waypoint_prompt if self.tasks else ""

    @property
    def place_target_prompt(self) -> str:
        return self.tasks[0].place_target_prompt if self.tasks else ""

    @property
    def pick_target(self) -> str:
        return self.tasks[0].pick_target if self.tasks else ""

    @property
    def used_default_place_target_prompt(self) -> bool:
        if not self.tasks:
            return True
        return self.tasks[0].used_default_place_target_prompt

    @property
    def used_default_nav_waypoint_prompt(self) -> bool:
        if not self.tasks:
            return True
        return self.tasks[0].used_default_nav_waypoint_prompt


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


def _normalize_mapping_keys(value: Any) -> dict[str, Any]:
    if not isinstance(value, dict):
        return {}
    return {str(k).strip().lower(): v for k, v in value.items()}


def _clean_text(value: Any) -> str:
    if value is None:
        return ""
    text = str(value).strip().strip("\"'`")
    text = re.sub(r"\s+", " ", text)
    return text.strip(" ,.;:[]{}()")


def _build_parser_prompt(
    request: str,
) -> str:
    return (
        "You convert a human manipulation request into grounded prompt slots for a robot.\n"
        "Return only one JSON object with these keys:\n"
        '{\n'
        '  "tasks": [\n'
        '    {\n'
        '      "pick_target": "...",\n'
        '      "place_target_prompt": "..."\n'
        '    }\n'
        '  ]\n'
        '}\n'
        "Rules:\n"
        "- Create one task per manipulation sub-request and preserve the original order.\n"
        "- Always return a tasks list, even if there is only one task.\n"
        "- Output short English grounding phrases for pick_target and place_target_prompt.\n"
        "- pick_target must be a visual noun phrase only, not an action sentence.\n"
        "- If multiple pick targets share the same place target, repeat that place_target_prompt for each task.\n"
        "- If a task does not specify a place target, set its place_target_prompt to an empty string.\n"
        f'Request: "{request}"'
    )


def _extract_task_objects(parsed: dict[str, Any]) -> list[dict[str, Any]]:
    tasks: list[dict[str, Any]] = []

    tasks_raw = parsed.get("tasks")
    if isinstance(tasks_raw, list):
        for item in tasks_raw:
            normalized = _normalize_mapping_keys(item)
            if normalized:
                tasks.append(normalized)
    elif isinstance(tasks_raw, dict):
        normalized = _normalize_mapping_keys(tasks_raw)
        if normalized:
            tasks.append(normalized)
    if tasks:
        return tasks

    pick_targets = parsed.get("pick_targets")
    if not isinstance(pick_targets, list):
        pick_targets = parsed.get("pick_target_list")
    place_targets = parsed.get("place_target_prompts")
    if not isinstance(place_targets, list):
        place_targets = parsed.get("place_target_prompt_list")
    if isinstance(pick_targets, list):
        place_list = place_targets if isinstance(place_targets, list) else []
        for idx, pick_target in enumerate(pick_targets):
            place_target_prompt = place_list[idx] if idx < len(
                place_list) else ""
            tasks.append({
                "pick_target": pick_target,
                "place_target_prompt": place_target_prompt,
            })
        if tasks:
            return tasks

    single_task = {
        "pick_target": parsed.get("pick_target"),
        "place_target_prompt": parsed.get("place_target_prompt"),
    }
    if _clean_text(single_task["pick_target"]) or _clean_text(
        single_task["place_target_prompt"]
    ):
        tasks.append(single_task)
    return tasks


def _parse_with_model(
    request: str,
    *,
    default_nav_waypoint_prompt: str,
    default_place_target_prompt: str,
    base_url: str,
    model_name: str,
    api_key: str,
) -> ShelfPromptPlan:
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

    tasks: list[ShelfPromptTask] = []
    for task_data in _extract_task_objects(parsed):
        pick_target_raw = _clean_text(
            task_data.get("pick_target") or task_data.get("nav_pick_prompt")
        )
        place_target_prompt = _clean_text(
            task_data.get("place_target_prompt") or task_data.get(
                "place_target")
        )
        if not pick_target_raw and not place_target_prompt:
            continue
        if not pick_target_raw:
            raise RuntimeError(
                f"missing pick_target in parser output: {raw!r}")

        use_default_place_target_prompt = not bool(place_target_prompt)
        if use_default_place_target_prompt:
            place_target_prompt = default_place_target_prompt

        tasks.append(ShelfPromptTask(
            nav_pick_prompt=pick_target_raw,
            nav_waypoint_prompt=default_nav_waypoint_prompt,
            place_target_prompt=place_target_prompt,
            pick_target=pick_target_raw,
            used_default_nav_waypoint_prompt=True,
            used_default_place_target_prompt=use_default_place_target_prompt,
        ))

    if not tasks:
        raise RuntimeError(f"missing tasks in parser output: {raw!r}")

    return ShelfPromptPlan(
        raw_request=request,
        tasks=tasks,
        parser_source="model",
        parser_raw=raw,
    )


def parse_human_shelf_request(
    request: str,
    *,
    base_url: str = "http://172.28.102.11:22002/v1",
    model_name: str = "Embodied-R1.5-SFT-0128",
    api_key: str = "EMPTY",
) -> ShelfPromptPlan:
    if not _clean_text(request):
        raise ValueError("request must be non-empty")

    return _parse_with_model(
        request,
        default_nav_waypoint_prompt=DEFAULT_NAV_WAYPOINT_PROMPT,
        default_place_target_prompt=DEFAULT_PLACE_TARGET_PROMPT,
        base_url=base_url,
        model_name=model_name,
        api_key=api_key,
    )


def _print_plan(plan: ShelfPromptPlan) -> None:
    print(f"task_count = {len(plan.tasks)}")
    for idx, task in enumerate(plan.tasks, start=1):
        print(f"task[{idx}].pick_target = {task.pick_target}")
        print(f"task[{idx}].nav_pick_prompt = {task.nav_pick_prompt}")
        print(f"task[{idx}].place_target_prompt = {task.place_target_prompt}")
        print(
            f"task[{idx}].used_default_place_target_prompt = "
            f"{task.used_default_place_target_prompt}"
        )
    print(f"parser_source = {plan.parser_source}")
    print(f"parser_raw = {plan.parser_raw}")


def main() -> None:
    cases = [
        "我想要一瓶可乐",
        "帮我把可乐放到蓝色盘子中间",
        "帮我拿一罐可乐",
        "帮我拿一个网球",
        "帮我把黄色螺丝刀放到白色盘子中间",
        "先帮我拿一罐可乐放到蓝色盘子中间，再把网球放到白色盘子中间",
    ]

    for text in cases:
        print(text)
        try:
            plan = parse_human_shelf_request(
                text,
            )
        except Exception as exc:
            print(f"error = {exc}")
        else:
            _print_plan(plan)
        print("---")


__all__ = [
    "DEFAULT_NAV_WAYPOINT_PROMPT",
    "DEFAULT_PLACE_TARGET_PROMPT",
    "ShelfPromptTask",
    "ShelfPromptPlan",
    "parse_human_shelf_request",
]


if __name__ == "__main__":
    main()
