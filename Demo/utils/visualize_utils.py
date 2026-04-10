from __future__ import annotations

import os
import select
import sys
import termios
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Literal, Optional

import cv2
import numpy as np

try:
    from PIL import Image, ImageDraw, ImageFont
except ImportError:
    Image = None
    ImageDraw = None
    ImageFont = None

_PIL_FONT_CACHE: dict[int, Optional[Any]] = {}
_FONT_CANDIDATE_PATHS = (
    "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
    "/usr/share/fonts/opentype/noto/NotoSansCJKSC-Regular.otf",
    "/usr/share/fonts/opentype/noto/NotoSerifCJK-Regular.ttc",
    "/usr/share/fonts/truetype/wqy/wqy-zenhei.ttc",
    "/usr/share/fonts/truetype/wqy/wqy-microhei.ttc",
    "/usr/share/fonts/truetype/arphic/uming.ttc",
    "/usr/share/fonts/truetype/arphic/ukai.ttc",
    "/System/Library/Fonts/Hiragino Sans GB.ttc",
    "/System/Library/Fonts/STHeiti Light.ttc",
    "/System/Library/Fonts/STHeiti Medium.ttc",
    "/Library/Fonts/Arial Unicode.ttf",
    "/Library/Fonts/Microsoft/SimHei.ttf",
    "/Library/Fonts/Microsoft/MSYH.TTC",
)


def _load_debug_font(font_size: int) -> Optional[Any]:
    cached = _PIL_FONT_CACHE.get(font_size)
    if cached is not None or font_size in _PIL_FONT_CACHE:
        return cached
    if ImageFont is None:
        _PIL_FONT_CACHE[font_size] = None
        return None

    candidate_paths: list[str] = []
    env_font = os.environ.get("ARX_DEBUG_FONT", "").strip()
    if env_font:
        candidate_paths.append(env_font)
    candidate_paths.extend(_FONT_CANDIDATE_PATHS)

    font = None
    for candidate in candidate_paths:
        if not Path(candidate).exists():
            continue
        try:
            font = ImageFont.truetype(candidate, font_size)
            break
        except OSError:
            continue

    _PIL_FONT_CACHE[font_size] = font
    return font


def _draw_debug_panel_cv2(
    image: np.ndarray,
    lines: list[str],
    *,
    origin: tuple[int, int],
) -> None:
    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = 0.6
    thickness = 2
    line_gap = 8
    x0, y0 = origin
    text_sizes = [
        cv2.getTextSize(line, font, scale, thickness)[0]
        for line in lines
    ]
    box_w = max(size[0] for size in text_sizes) + 16
    box_h = sum(size[1] for size in text_sizes) + line_gap * (
        len(lines) - 1
    ) + 16
    cv2.rectangle(
        image,
        (x0, y0),
        (x0 + box_w, y0 + box_h),
        color=(32, 32, 32),
        thickness=-1,
    )
    cv2.rectangle(
        image,
        (x0, y0),
        (x0 + box_w, y0 + box_h),
        color=(220, 220, 220),
        thickness=1,
    )
    cursor_y = y0 + 10
    for line, size in zip(lines, text_sizes):
        baseline_y = cursor_y + size[1]
        cv2.putText(
            image,
            line,
            (x0 + 8, baseline_y),
            font,
            scale,
            (255, 255, 255),
            thickness,
            cv2.LINE_AA,
        )
        cursor_y = baseline_y + line_gap


def _draw_debug_panel_pil(
    image: np.ndarray,
    lines: list[str],
    *,
    origin: tuple[int, int],
    font: Any,
) -> None:
    x0, y0 = origin
    padding_x = 10
    padding_y = 8
    line_gap = 6

    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    pil_image = Image.fromarray(image_rgb)
    drawer = ImageDraw.Draw(pil_image)
    text_boxes = []
    text_heights = []
    for line in lines:
        bbox = drawer.textbbox((0, 0), line, font=font)
        width = max(0, bbox[2] - bbox[0])
        height = max(font.size, bbox[3] - bbox[1])
        text_boxes.append((width, height))
        text_heights.append(height)

    box_w = max(width for width, _ in text_boxes) + padding_x * 2
    box_h = sum(text_heights) + line_gap * (len(lines) - 1) + padding_y * 2
    drawer.rectangle(
        [(x0, y0), (x0 + box_w, y0 + box_h)],
        fill=(32, 32, 32),
        outline=(220, 220, 220),
        width=1,
    )

    cursor_y = y0 + padding_y
    for line, (_, height) in zip(lines, text_boxes):
        drawer.text(
            (x0 + padding_x, cursor_y),
            line,
            font=font,
            fill=(255, 255, 255),
        )
        cursor_y += height + line_gap

    image[:] = cv2.cvtColor(np.asarray(pil_image), cv2.COLOR_RGB2BGR)


@dataclass
class VisualizeContext:
    stop_checker: Optional[Callable[[], bool]] = None


def _draw_debug_panel(
    image: np.ndarray,
    lines: list[str],
    *,
    origin: tuple[int, int] = (12, 16),
) -> None:
    if not lines:
        return
    font = _load_debug_font(font_size=22)
    if font is not None and Image is not None and ImageDraw is not None:
        _draw_debug_panel_pil(image, lines, origin=origin, font=font)
        return
    _draw_debug_panel_cv2(image, lines, origin=origin)


def should_stop(visualize: Optional[VisualizeContext]) -> bool:
    if visualize is None or visualize.stop_checker is None:
        return False
    try:
        return bool(visualize.stop_checker())
    except Exception as exc:
        print(f"[visualize_utils] stop checker failed: {exc}")
        return False


def get_key_nonblock() -> Optional[str]:
    try:
        if not sys.stdin.isatty():
            return None
        dr, _, _ = select.select([sys.stdin], [], [], 0)
    except (AttributeError, OSError, ValueError):
        return None
    if not dr:
        return None
    try:
        return sys.stdin.read(1)
    except OSError:
        return None


def init_keyboard():
    try:
        if not sys.stdin.isatty():
            return None
        fd = sys.stdin.fileno()
        old_settings = termios.tcgetattr(fd)
        new_settings = termios.tcgetattr(fd)
        new_settings[3] = new_settings[3] & ~termios.ICANON & ~termios.ECHO
        termios.tcsetattr(fd, termios.TCSADRAIN, new_settings)
        return old_settings
    except (AttributeError, OSError, termios.error, ValueError):
        return None


def restore_keyboard(old_settings) -> None:
    if old_settings is None:
        return
    try:
        termios.tcsetattr(sys.stdin.fileno(), termios.TCSADRAIN, old_settings)
    except (AttributeError, OSError, termios.error, ValueError):
        return


def confirm_debug_image(
    image: np.ndarray,
    *,
    window_name: str,
    stop_checker: Optional[Callable[[], bool]] = None,
) -> Literal[True, False, None]:
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.imshow(window_name, image)
    while True:
        if stop_checker is not None:
            try:
                if stop_checker():
                    cv2.destroyWindow(window_name)
                    return None
            except Exception:
                pass
        key = get_key_nonblock()
        if key == "q":
            cv2.destroyWindow(window_name)
            return None
        if key == "r":
            cv2.destroyWindow(window_name)
            return False
        if key == "e":
            cv2.destroyWindow(window_name)
            return True

        cv_key = cv2.waitKey(50)
        if cv_key < 0:
            continue

        if cv_key == ord("q"):
            cv2.destroyWindow(window_name)
            return None
        if cv_key == ord("r"):
            cv2.destroyWindow(window_name)
            return False
        if cv_key == ord("e"):
            cv2.destroyWindow(window_name)
            return True
        continue


def dispatch_debug_image(
    visualize: Optional[VisualizeContext],
    *,
    source: str,
    panel: str,
    image: np.ndarray,
    window_name: str,
    **payload: Any,
) -> Literal[True, False, None]:
    emit_debug_image(
        visualize,
        source=source,
        panel=panel,
        image=image,
        **payload,
    )
    stop_checker = None if visualize is None else visualize.stop_checker
    return confirm_debug_image(
        image,
        window_name=window_name,
        stop_checker=stop_checker,
    )


def render_nav_goal_debug_view(
    color: np.ndarray,
    points,
    goal_pixel,
    *,
    roi_polygon: Optional[np.ndarray] = None,
    nav_prompt: Optional[str] = None,
) -> np.ndarray:
    vis = color.copy()
    if roi_polygon is not None:
        cv2.polylines(
            vis,
            [roi_polygon.reshape((-1, 1, 2))],
            isClosed=True,
            color=(255, 255, 0),
            thickness=2,
        )
    for point in points:
        cv2.circle(
            vis,
            center=(int(round(point[0])), int(round(point[1]))),
            radius=5,
            color=(0, 0, 255),
            thickness=-1,
        )

    cv2.circle(
        vis,
        center=(int(round(goal_pixel[0])), int(round(goal_pixel[1]))),
        radius=9,
        color=(0, 255, 0),
        thickness=2,
    )

    if nav_prompt:
        _draw_debug_panel(
            vis,
            [f"Prompt: {nav_prompt}", "E: accept  R: refresh  Q: quit"],
        )
    return vis


def render_dual_swap_debug_view(
    color: np.ndarray,
    pixel: tuple[int, int],
    *,
    object_prompt: str = "",
) -> np.ndarray:
    vis = color.copy()
    cv2.circle(vis, pixel, 4, (0, 0, 255), -1)
    cv2.circle(vis, pixel, 10, (255, 255, 255), 2)
    lines = []
    if object_prompt:
        lines.append(f"Object: {object_prompt}")
    lines.append("E: accept  R: refresh  Q: quit")
    _draw_debug_panel(vis, lines)
    return vis


def render_pick_place_debug_view(
    color: np.ndarray,
    *,
    arm: str,
    pick_prompt: str,
    place_prompt: str,
    pick_px: Optional[tuple[int, int]],
    place_px: Optional[tuple[int, int]],
) -> np.ndarray:
    vis = color.copy()
    lines = [f"Arm: {arm}"]
    if pick_px is not None:
        cv2.circle(vis, pick_px, 3, (0, 0, 255), -1)
        cv2.circle(vis, pick_px, 10, (255, 255, 255), 2)
        lines.append(f"Pick: {pick_prompt}")
    if place_px is not None:
        cv2.circle(vis, place_px, 3, (255, 0, 0), -1)
        cv2.circle(vis, place_px, 10, (255, 255, 255), 2)
        lines.append(f"Place: {place_prompt}")
    lines.append("E: accept  R: refresh  Q: quit")
    _draw_debug_panel(vis, lines)
    return vis


def render_multi_points_debug_view(
    color: np.ndarray,
    *,
    points: list[tuple[int, int]],
    title: Optional[str] = None,
) -> np.ndarray:
    vis = color.copy()
    for idx, point in enumerate(points, start=1):
        cv2.circle(vis, point, 3, (0, 0, 255), -1)
        cv2.circle(vis, point, 10, (255, 255, 255), 2)
        cv2.putText(
            vis,
            str(idx),
            (point[0] + 6, point[1] - 6),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )

    lines = []
    if title:
        lines.append(title)
    lines.append(f"Points: {len(points)}")
    lines.append("E: accept  R: refresh  Q: quit")
    _draw_debug_panel(vis, lines)
    return vis


def render_move_away_debug_view(
    color: np.ndarray,
    *,
    pixel: tuple[int, int],
    blocked: bool,
    description: str,
    arm: str,
) -> np.ndarray:
    vis = color.copy()
    cv2.circle(vis, pixel, 4, (0, 0, 255), -1)
    cv2.circle(vis, pixel, 10, (255, 255, 255), 2)
    _draw_debug_panel(
        vis,
        [
            f"Blocked: {blocked}",
            f"Description: {description or 'unknown'}",
            f"Arm: {arm}",
            "E: accept  R: refresh  Q: quit",
        ],
    )
    return vis
