from __future__ import annotations

import time
from pathlib import Path

import numpy as np

try:
    from collect_utils import (
        ARXRobotEnv,
        _split_dual,
        load_episode,
        wait_until_status_ready,
    )
except ImportError:  # pragma: no cover
    from .collect_utils import (
        ARXRobotEnv,
        _split_dual,
        load_episode,
        wait_until_status_ready,
    )


def _episode_dim(episode) -> int:
    if not episode.frames:
        raise RuntimeError("Episode has no frames.")
    return int(np.asarray(episode.frames[0].action, dtype=np.float32).reshape(-1).shape[0])


def _dual_action_dict(action: np.ndarray) -> dict[str, np.ndarray]:
    left, right = _split_dual(action)
    return {"left": left, "right": right}


def _single_action_dict(action: np.ndarray, side: str) -> dict[str, np.ndarray]:
    payload = np.asarray(action, dtype=np.float32).reshape(-1)
    if payload.shape[0] != 7:
        raise ValueError(
            f"Expected single-arm action dim 7, got {payload.shape[0]}")
    return {side: payload.copy()}


def _apply_base(env: ARXRobotEnv, frame, last_height: float | None) -> float | None:
    if frame.action_base is None:
        return last_height
    action_base = np.asarray(frame.action_base, dtype=np.float32).reshape(-1)
    if action_base.shape[0] < 4:
        return last_height
    target_height = float(action_base[3])
    if last_height is None or abs(target_height - last_height) > 0.05:
        env.step_lift(target_height)
        last_height = target_height
    env.step_base(float(action_base[0]), float(
        action_base[1]), float(action_base[2]))
    return last_height


def replay_episode(
    env: ARXRobotEnv,
    episode_dir: Path | str,
    speed: float = 1.0,
    start_index: int = 0,
    end_index: int = -1,
    single_side: str | None = None,
) -> None:
    episode_path = Path(episode_dir)
    if not episode_path.is_dir():
        raise FileNotFoundError(f"Episode directory not found: {episode_path}")
    episode = load_episode(episode_path)
    frame_dim = _episode_dim(episode)

    if frame_dim == 7:
        is_single = True
    elif frame_dim == 14:
        is_single = False
    else:
        raise ValueError(
            f"Unsupported action dim: expected 7 or 14, got {frame_dim}")
    print(
        f"Auto-detected {'single' if is_single else 'dual'}-arm replay from {frame_dim}D action.")

    action_kind = str(episode.action_kind)
    if action_kind not in {"joint", "eef"}:
        raise ValueError(f"Unsupported action kind: {action_kind}")

    frames = list(episode.frames)
    end_slice = None if int(end_index) < 0 else int(end_index)
    frames = frames[int(start_index):end_slice]
    if not frames:
        raise RuntimeError("Selected frame range is empty.")

    replay_side = single_side or episode.side
    if is_single and replay_side not in {"left", "right"}:
        raise ValueError("Single-arm replay requires a valid side.")

    wait_until_status_ready(env, include_base=(
        not is_single and episode.include_base))
    print(
        f"Replaying {len(frames)} frames from mode={episode.mode}, action_kind={action_kind}")

    last_height: float | None = None
    for idx, frame in enumerate(frames):
        t0 = time.perf_counter()
        action = np.asarray(frame.action, dtype=np.float32)
        if is_single:
            action_dict = _single_action_dict(action, replay_side)
        else:
            action_dict = _dual_action_dict(action)
            if episode.include_base:
                last_height = _apply_base(env, frame, last_height)

        if action_kind == "joint":
            env.step_raw_joint(action_dict)
        else:
            env.step_raw_eef(action_dict)

        if idx + 1 >= len(frames):
            continue
        dt = float(frames[idx + 1].timestamp -
                   frame.timestamp) / max(float(speed), 1e-6)
        sleep_need = dt - (time.perf_counter() - t0)
        if sleep_need > 0.0:
            time.sleep(sleep_need)

    try:
        env.step_base(0.0, 0.0, 0.0)
    except Exception:
        pass


def main() -> None:
    arx = ARXRobotEnv(camera_type="color", camera_view=(),
                      dir=None, video=False, img_size=(224, 224))
    try:
        arx.reset()
        arx.step_lift(14.5)
        replay_episode(arx, episode_dir="episodes_raw/test/episode_000000")
    finally:
        arx.close()


if __name__ == "__main__":
    main()
