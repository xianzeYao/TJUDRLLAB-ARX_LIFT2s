from __future__ import annotations

import ast
import json
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal, Optional

import cv2
import numpy as np

CURRENT_DIR = Path(__file__).resolve().parent
ROOT_DIR = CURRENT_DIR.parent
ROS2_DIR = ROOT_DIR / "ARX_Realenv" / "ROS2"

if str(CURRENT_DIR) not in sys.path:
    sys.path.append(str(CURRENT_DIR))
if str(ROS2_DIR) not in sys.path:
    sys.path.append(str(ROS2_DIR))

from arx_pointing import predict_multi_points_from_rgb, predict_point_from_rgb  # noqa: E402
from arx_ros2_env import ARXRobotEnv  # noqa: E402
from demo_utils import execute_move_away  # noqa: E402
from point2pos_utils import get_aligned_frames, pixel_to_ref_point_safe  # noqa: E402

DEFAULT_DETECT_RETRIES = 3


@dataclass
class FrontBlockCheckResult:
    blocked: bool
    description: str
    raw: Optional[str] = None


@dataclass
class MoveAwayRunResult:
    blocked: bool
    description: str
    moved_away: bool
    arm: Optional[Literal["left", "right"]] = None
    blocker_ref_point: Optional[np.ndarray] = None
    message: str = ""


def _preprocess_text(text: str) -> str:
    text = re.sub(r"```(?:json|python|text)?\n?(.*?)\n?```",
                  r"\1", text, flags=re.DOTALL)
    match = re.search(r"<answer>(.*?)</answer>", text,
                      flags=re.DOTALL | re.IGNORECASE)
    if match:
        text = match.group(1)
    return text.strip()


def _parse_bool(value: Any) -> Optional[bool]:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)) and value in (0, 1):
        return bool(value)
    if value is None:
        return None

    text = str(value).strip().lower()
    if not text:
        return None
    if text in {"true", "yes", "y", "1"}:
        return True
    if text in {"false", "no", "n", "0"}:
        return False

    matches = re.findall(r"\b(true|false)\b", text)
    if matches:
        return matches[0] == "true"
    return None


def _clean_description(text: Any) -> str:
    if text is None:
        return ""
    cleaned = str(text).strip().strip("\"'`")
    cleaned = re.sub(r"\s+", " ", cleaned)
    cleaned = cleaned.strip(" .,;:[](){}")
    return cleaned


def _decode_block_result(raw: Optional[str]) -> FrontBlockCheckResult:
    cleaned = _preprocess_text(raw or "")
    raw_json = cleaned
    match = re.search(r"(\{.*\})", cleaned, re.DOTALL)
    if match:
        raw_json = match.group(1)

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
        blocked = _parse_bool(normalized.get("blocked"))
        if blocked is None:
            blocked = False
        description = _clean_description(
            normalized.get("description")
            or normalized.get("object")
            or normalized.get("blocker")
            or normalized.get("item")
        )
        if not blocked:
            description = ""
        return FrontBlockCheckResult(
            blocked=bool(blocked),
            description=description,
            raw=raw,
        )

    blocked = _parse_bool(cleaned)
    if blocked is None:
        blocked = False
    description = ""
    if blocked:
        description = re.sub(
            r"^\s*(true|false)\s*[:,\-]?\s*", "", cleaned, flags=re.IGNORECASE)
        description = _clean_description(description)
    if not blocked:
        description = ""
    return FrontBlockCheckResult(
        blocked=bool(blocked),
        description=description,
        raw=raw,
    )


def _build_front_blocking_prompt(pick_prompt: str) -> str:
    return "\n".join(
        [
            f"You are checking whether there is something in front of the {pick_prompt} and blocking it.",
            '"blocked": true or false',
            '"description": short description of the object infront of the {pick_prompt}',
        ]
    )


def _draw_text_lines(
    image: np.ndarray,
    lines: list[str],
    origin: tuple[int, int] = (10, 25),
    line_height: int = 22,
    color: tuple[int, int, int] = (0, 0, 255),
) -> None:
    x, y = origin
    for idx, line in enumerate(lines):
        cv2.putText(
            image,
            line,
            (x, y + idx * line_height),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            color,
            2,
        )


def _capture_color_depth(
    arx: ARXRobotEnv,
    depth_median_n: int,
) -> tuple[Optional[np.ndarray], Optional[np.ndarray]]:
    return get_aligned_frames(arx, depth_median_n=depth_median_n)


def _query_front_block_once(
    color: np.ndarray,
    pick_prompt: str,
) -> FrontBlockCheckResult:
    _, raw = predict_multi_points_from_rgb(
        image=color,
        text_prompt="",
        all_prompt=_build_front_blocking_prompt(pick_prompt),
        assume_bgr=False,
        temperature=0.0,
        max_tokens=256,
        return_raw=True,
    )
    return _decode_block_result(raw)


def check_front_blocking(
    arx: ARXRobotEnv,
    pick_prompt: str,
    *,
    depth_median_n: int = 5,
) -> FrontBlockCheckResult:
    color, _ = _capture_color_depth(arx, depth_median_n=depth_median_n)
    if color is None:
        raise RuntimeError("failed to capture top camera color image")
    return _query_front_block_once(color, pick_prompt)


def _predict_blocker_pixel(
    color: np.ndarray,
    blocker_description: str,
) -> tuple[int, int]:
    blocker_text = blocker_description or "the blocking object"
    u, v = predict_point_from_rgb(
        color,
        text_prompt=blocker_text,
        assume_bgr=False,
        temperature=0.0,
        max_tokens=256,
    )
    return int(round(u)), int(round(v))


def _select_arm_from_pixel(
    pixel: tuple[int, int],
    image_width: int,
) -> Literal["left", "right"]:
    return "left" if pixel[0] < (image_width / 2.0) else "right"


def _confirm_detection(
    color: np.ndarray,
    pixel: tuple[int, int],
    check_result: FrontBlockCheckResult,
    arm: Literal["left", "right"],
) -> Literal["accept", "retry", "abort"]:
    vis = color.copy()
    cv2.circle(vis, pixel, 4, (0, 0, 255), -1)
    _draw_text_lines(
        vis,
        [
            f"blocked={check_result.blocked}",
            f"description={check_result.description or 'unknown'}",
            f"arm={arm}",
        ],
    )
    win = "move_away_detect"
    cv2.namedWindow(win, cv2.WINDOW_NORMAL)
    cv2.imshow(win, vis)
    key = cv2.waitKey(0)
    cv2.destroyWindow(win)
    if key == ord("r"):
        return "retry"
    if key == ord("q"):
        return "abort"
    return "accept"


def create_arx_env() -> ARXRobotEnv:
    return ARXRobotEnv(
        duration_per_step=1.0 / 20.0,
        min_steps=20,
        max_v_xyz=0.15,
        max_a_xyz=0.20,
        max_v_rpy=0.45,
        max_a_rpy=1.00,
        camera_type="all",
        camera_view=("camera_l", "camera_h", "camera_r"),
        img_size=(640, 480),
    )


def move_away(
    arx: ARXRobotEnv,
    pick_prompt: str,
    *,
    debug_raw: bool = True,
    depth_median_n: int = 5,
    home_after_move: bool = True,
) -> MoveAwayRunResult:
    if not pick_prompt:
        raise ValueError("pick_prompt must be non-empty")

    try:
        for _ in range(DEFAULT_DETECT_RETRIES):
            color, depth = _capture_color_depth(
                arx, depth_median_n=depth_median_n)
            if color is None or depth is None:
                continue

            check_result = _query_front_block_once(
                color,
                pick_prompt,
            )
            print(
                "[move_away] "
                f"blocked={check_result.blocked}, "
                f"description={check_result.description!r}"
            )
            if not check_result.blocked:
                return MoveAwayRunResult(
                    blocked=False,
                    description="",
                    moved_away=False,
                    message="no front blocker found",
                )

            try:
                blocker_pixel = _predict_blocker_pixel(
                    color,
                    check_result.description,
                )
            except Exception as exc:
                print(f"[move_away] blocker point prediction failed: {exc}")
                continue

            arm = _select_arm_from_pixel(blocker_pixel, color.shape[1])
            blocker_ref_point = pixel_to_ref_point_safe(
                blocker_pixel,
                depth,
                robot_part=arm,
            )
            if blocker_ref_point is None:
                print(
                    "[move_away] invalid depth for blocker pixel "
                    f"{blocker_pixel}, retry"
                )
                continue

            if debug_raw:
                decision = _confirm_detection(
                    color,
                    blocker_pixel,
                    check_result,
                    arm,
                )
                if decision == "retry":
                    continue
                if decision == "abort":
                    return MoveAwayRunResult(
                        blocked=check_result.blocked,
                        description=check_result.description,
                        moved_away=False,
                        arm=arm,
                        blocker_ref_point=blocker_ref_point,
                        message="user aborted",
                    )

            execute_move_away(arx, blocker_ref_point, arm)
            message = "move away executed"
            if home_after_move:
                success, error_message = arx.set_special_mode(1, side=arm)
                if not success and error_message:
                    message = (
                        "move away executed, "
                        f"but failed to home {arm}: {error_message}"
                    )

            return MoveAwayRunResult(
                blocked=check_result.blocked,
                description=check_result.description,
                moved_away=True,
                arm=arm,
                blocker_ref_point=blocker_ref_point,
                message=message,
            )

        return MoveAwayRunResult(
            blocked=True,
            description="",
            moved_away=False,
            message="blocker check or localization failed after retries",
        )
    finally:
        cv2.destroyAllWindows()


def main() -> None:
    arx = create_arx_env()
    try:
        arx.reset()
        arx.step_lift(12.5)
        result = move_away(
            arx=arx,
            pick_prompt="the blue box",
            debug_raw=True,
            depth_median_n=5,
            home_after_move=True,
        )
        print(
            "[move_away_result] "
            f"blocked={result.blocked}, "
            f"description={result.description!r}, "
            f"moved_away={result.moved_away}, "
            f"arm={result.arm}, "
            f"message={result.message}"
        )
    finally:
        arx.close()


if __name__ == "__main__":
    main()
