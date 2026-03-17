from __future__ import annotations

import json
import random
from pathlib import Path

import numpy as np

DISPLAY_WIDTH = 1280
DISPLAY_HEIGHT = 800


def _import_cv2():
    try:
        import cv2
    except ImportError as exc:
        raise ImportError(
            "visualize_lerobot_v21.py requires `opencv-python` / `cv2`."
        ) from exc
    return cv2


def _parse_episode_selector(selector: int | str, total_episodes: int) -> list[int]:
    if total_episodes <= 0:
        raise ValueError("Dataset has no episodes.")

    if isinstance(selector, int):
        indices = [selector]
    else:
        raw = str(selector).strip().lower()
        if raw in {"", "all"}:
            indices = list(range(total_episodes))
        elif raw == "random":
            indices = [random.randrange(total_episodes)]
        elif raw.isdigit():
            indices = [int(raw)]
        elif raw.startswith("[") and raw.endswith("]"):
            body = raw[1:-1]
            parts = [part.strip() for part in body.split(",")]
            if len(parts) != 2 or not all(part.lstrip("-").isdigit() for part in parts):
                raise ValueError(
                    "Episode selector range must look like '[1,5]'."
                )
            start, end = (int(parts[0]), int(parts[1]))
            if start > end:
                raise ValueError(
                    "Episode selector range start must be <= end.")
            indices = list(range(start, end + 1))
        else:
            raise ValueError(
                "Unsupported episode selector. Use 'all', 'random', an integer like '3', or a range like '[1,5]'."
            )

    normalized: list[int] = []
    for idx in indices:
        if idx < 0 or idx >= total_episodes:
            raise IndexError(
                f"Episode index {idx} is out of range for total_episodes={total_episodes}."
            )
        normalized.append(idx)
    return normalized


def _load_jsonl(path: Path) -> list[dict]:
    rows: list[dict] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            rows.append(json.loads(line))
    return rows


def _camera_order_key(video_key: str) -> tuple[int, str]:
    order = {
        "camera_l": 0,
        "camera_h": 1,
        "camera_r": 2,
    }
    camera_name = video_key.split(".")[-1]
    return (order.get(camera_name, 999), video_key)


def _resize_to_height(frame: np.ndarray, target_height: int, cv2) -> np.ndarray:
    height, width = frame.shape[:2]
    if height == target_height:
        return frame
    scale = target_height / float(height)
    target_width = max(1, int(round(width * scale)))
    return cv2.resize(frame, (target_width, target_height), interpolation=cv2.INTER_AREA)


def _fit_canvas_to_window(
    canvas: np.ndarray,
    target_width: int,
    target_height: int,
    cv2,
) -> np.ndarray:
    height, width = canvas.shape[:2]
    scale = min(target_width / float(width), target_height / float(height))
    scaled_width = max(1, int(round(width * scale)))
    scaled_height = max(1, int(round(height * scale)))
    resized = cv2.resize(canvas, (scaled_width, scaled_height),
                         interpolation=cv2.INTER_AREA)

    output = np.zeros((target_height, target_width, 3), dtype=np.uint8)
    y0 = (target_height - scaled_height) // 2
    x0 = (target_width - scaled_width) // 2
    output[y0:y0 + scaled_height, x0:x0 + scaled_width] = resized
    return output


def _compose_frame(
    frames: list[tuple[str, np.ndarray]],
    title: str,
    subtitle: str,
    cv2,
) -> np.ndarray:
    target_height = min(frame.shape[0] for _, frame in frames)
    labeled_frames: list[tuple[str, np.ndarray]] = []
    for camera_key, frame in frames:
        frame = _resize_to_height(frame, target_height, cv2)
        canvas = frame.copy()
        cv2.putText(
            canvas,
            camera_key,
            (12, 28),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 255, 255),
            2,
            cv2.LINE_AA,
        )
        labeled_frames.append((camera_key, canvas))

    if len(labeled_frames) == 3:
        frame_map = {camera_key: frame for camera_key, frame in labeled_frames}
        left = frame_map.get("observation.images.camera_l",
                             labeled_frames[0][1])
        middle = frame_map.get(
            "observation.images.camera_h", labeled_frames[1][1])
        right = frame_map.get(
            "observation.images.camera_r", labeled_frames[2][1])

        top_row = np.concatenate([left, right], axis=1)
        bottom_row = np.zeros_like(top_row)
        x0 = (top_row.shape[1] - middle.shape[1]) // 2
        bottom_row[:, x0:x0 + middle.shape[1]] = middle
        merged = np.concatenate([top_row, bottom_row], axis=0)
    else:
        merged = np.concatenate([frame for _, frame in labeled_frames], axis=1)

    header = np.zeros((64, merged.shape[1], 3), dtype=np.uint8)
    cv2.putText(
        header,
        title,
        (12, 26),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )
    cv2.putText(
        header,
        subtitle,
        (12, 52),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (180, 180, 180),
        1,
        cv2.LINE_AA,
    )
    return np.concatenate([header, merged], axis=0)


def _play_episode(
    dataset_root: Path,
    fps: int,
    episode_row: dict,
) -> bool:
    cv2 = _import_cv2()

    episode_index = int(episode_row["episode_index"])
    tasks = episode_row.get("tasks", [])
    task_text = tasks[0] if tasks else ""
    sorted_videos = sorted(
        episode_row.get("videos", {}).items(),
        key=lambda item: _camera_order_key(item[0]),
    )
    video_map = dict(sorted_videos)
    if not video_map:
        print(f"episode_{episode_index:06d}: no videos found, skipped")
        return True

    captures: list[tuple[str, object]] = []
    for video_key, relative_path in video_map.items():
        video_path = dataset_root / relative_path
        capture = cv2.VideoCapture(str(video_path))
        if not capture.isOpened():
            raise RuntimeError(f"Failed to open video: {video_path}")
        captures.append((video_key, capture))

    window_name = "LeRobot v2.1 Viewer"
    delay_ms = max(1, int(round(1000.0 / max(float(fps), 1.0))))
    frame_index = 0
    paused = False
    last_canvas: np.ndarray | None = None
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window_name, DISPLAY_WIDTH, DISPLAY_HEIGHT)

    print(
        f"Viewing episode_{episode_index:06d} | task={task_text!r} | controls: space pause/resume, r restart, n/enter next, q/esc quit"
    )

    try:
        while True:
            if not paused or last_canvas is None:
                frames: list[tuple[str, np.ndarray]] = []
                for video_key, capture in captures:
                    ok, frame = capture.read()
                    if not ok:
                        return True
                    frames.append((video_key, frame))

                title = f"episode_{episode_index:06d}"
                subtitle = f"frame={frame_index}  task={task_text}"
                last_canvas = _compose_frame(frames, title, subtitle, cv2)
                frame_index += 1

            display_canvas = _fit_canvas_to_window(
                last_canvas, DISPLAY_WIDTH, DISPLAY_HEIGHT, cv2)
            cv2.imshow(window_name, display_canvas)
            key = cv2.waitKey(0 if paused else delay_ms) & 0xFF

            if key in (27, ord("q")):
                return False
            if key in (13, ord("n")):
                return True
            if key == ord(" "):
                paused = not paused
                continue
            if key == ord("r"):
                for _, capture in captures:
                    capture.set(cv2.CAP_PROP_POS_FRAMES, 0)
                frame_index = 0
                paused = False
                last_canvas = None
                continue
    finally:
        for _, capture in captures:
            capture.release()
        cv2.destroyWindow(window_name)


def visualize_lerobot_v21(
    dataset_root: Path | str,
    episode_selector: int | str = "all",
) -> None:
    dataset_root = Path(dataset_root)
    if not dataset_root.is_dir():
        raise FileNotFoundError(f"Dataset root not found: {dataset_root}")

    info_path = dataset_root / "meta" / "info.json"
    episodes_path = dataset_root / "meta" / "episodes.jsonl"
    if not info_path.is_file():
        raise FileNotFoundError(f"Missing dataset info file: {info_path}")
    if not episodes_path.is_file():
        raise FileNotFoundError(
            f"Missing episode metadata file: {episodes_path}")

    info = json.loads(info_path.read_text(encoding="utf-8"))
    episode_rows = sorted(
        _load_jsonl(episodes_path),
        key=lambda row: int(row["episode_index"]),
    )
    total_episodes = int(info.get("total_episodes", len(episode_rows)))
    if total_episodes != len(episode_rows):
        total_episodes = len(episode_rows)

    selected_indices = _parse_episode_selector(
        episode_selector, total_episodes)
    fps = int(info.get("fps", 15))

    for episode_index in selected_indices:
        keep_going = _play_episode(
            dataset_root, fps, episode_rows[episode_index])
        if not keep_going:
            break


def main() -> None:
    visualize_lerobot_v21(
        dataset_root="lerobot_v21/gravity_dual",
        episode_selector="all",
    )

    # visualize_lerobot_v21(
    #     dataset_root="lerobot_v21/gravity_dual",
    #     episode_selector="random",
    # )

    # visualize_lerobot_v21(
    #     dataset_root="lerobot_v21/gravity_dual",
    #     episode_selector="[1,5]",
    # )


if __name__ == "__main__":
    main()
