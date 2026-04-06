from __future__ import annotations

import select
import sys
import termios
from dataclasses import dataclass
from typing import Any, Callable, Literal, Optional

import cv2
import numpy as np

from demo_utils import draw_text_lines

VisualEventHandler = Callable[[str, dict[str, Any]], None]


@dataclass
class VisualizeContext:
    on_event: Optional[VisualEventHandler] = None
    stop_checker: Optional[Callable[[], bool]] = None
    page_debug: bool = False


def emit_event(
    visualize: Optional[VisualizeContext],
    event: str,
    **payload: Any,
) -> None:
    if visualize is None or visualize.on_event is None:
        return
    try:
        visualize.on_event(event, payload)
    except Exception as exc:
        print(f"[visualize_utils] event handler failed for {event!r}: {exc}")


def emit_log(
    visualize: Optional[VisualizeContext],
    *,
    source: str,
    message: str,
    stage: Optional[str] = None,
    **payload: Any,
) -> None:
    emit_event(
        visualize,
        "log",
        source=source,
        stage=stage,
        message=message,
        **payload,
    )


def emit_stage(
    visualize: Optional[VisualizeContext],
    *,
    source: str,
    stage: str,
    message: str,
    **payload: Any,
) -> None:
    emit_event(
        visualize,
        "stage",
        source=source,
        stage=stage,
        message=message,
        **payload,
    )


def emit_result(
    visualize: Optional[VisualizeContext],
    *,
    source: str,
    status: str,
    message: Optional[str] = None,
    **payload: Any,
) -> None:
    emit_event(
        visualize,
        "result",
        source=source,
        status=status,
        message=message,
        **payload,
    )


def emit_debug_image(
    visualize: Optional[VisualizeContext],
    *,
    source: str,
    panel: str,
    image: np.ndarray,
    **payload: Any,
) -> None:
    if visualize is None or not visualize.page_debug:
        return
    emit_event(
        visualize,
        "debug",
        source=source,
        panel=panel,
        image=image,
        **payload,
    )


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
) -> Literal[True, False, None]:
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.imshow(window_name, image)
    while True:
        key = get_key_nonblock()
        if key == "q":
            cv2.destroyWindow(window_name)
            return None

        cv_key = cv2.waitKey(50)
        if cv_key < 0:
            continue

        cv2.destroyWindow(window_name)
        if cv_key == ord("q"):
            return None
        if cv_key == ord("r"):
            return False
        return True


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
    if visualize is not None and visualize.page_debug:
        return True
    return confirm_debug_image(image, window_name=window_name)


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
        overlay_lines = [f"prompt: {nav_prompt}",
                         "ENTER: accept  R: refresh  Q: quit"]
        font = cv2.FONT_HERSHEY_SIMPLEX
        scale = 0.6
        thickness = 2
        line_gap = 8
        x0 = 12
        y0 = 16
        text_sizes = [
            cv2.getTextSize(line, font, scale, thickness)[0]
            for line in overlay_lines
        ]
        box_w = max(size[0] for size in text_sizes) + 16
        box_h = sum(size[1] for size in text_sizes) + line_gap * (
            len(overlay_lines) - 1
        ) + 16
        cv2.rectangle(
            vis,
            (x0, y0),
            (x0 + box_w, y0 + box_h),
            color=(32, 32, 32),
            thickness=-1,
        )
        cv2.rectangle(
            vis,
            (x0, y0),
            (x0 + box_w, y0 + box_h),
            color=(220, 220, 220),
            thickness=1,
        )
        cursor_y = y0 + 10
        for line, size in zip(overlay_lines, text_sizes):
            baseline_y = cursor_y + size[1]
            cv2.putText(
                vis,
                line,
                (x0 + 8, baseline_y),
                font,
                scale,
                (255, 255, 255),
                thickness,
                cv2.LINE_AA,
            )
            cursor_y = baseline_y + line_gap
    return vis


def render_dual_swap_debug_view(
    color: np.ndarray,
    pixel: tuple[int, int],
) -> np.ndarray:
    vis = color.copy()
    cv2.circle(vis, pixel, 3, (0, 0, 255), -1)
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
        lines.append(f"Pick: {pick_prompt}")
    if place_px is not None:
        cv2.circle(vis, place_px, 3, (255, 0, 0), -1)
        lines.append(f"Place: {place_prompt}")
    if lines:
        draw_text_lines(
            vis,
            lines,
            origin=(10, 25),
            line_height=22,
            color=(0, 0, 255),
            scale=0.6,
            thickness=2,
        )
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
    draw_text_lines(
        vis,
        [
            f"blocked={blocked}",
            f"description={description or 'unknown'}",
            f"arm={arm}",
        ],
    )
    return vis
