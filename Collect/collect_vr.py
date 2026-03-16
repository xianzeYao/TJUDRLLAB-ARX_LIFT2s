from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np

try:
    from collect_utils import (
        ARXRobotEnv,
        DualVRCollector,
        EpisodeFrame,
        create_episode_buffer,
        find_next_episode_index,
        record_episode_interactive,
        save_episode,
    )
except ImportError:  # pragma: no cover
    from .collect_utils import (
        ARXRobotEnv,
        DualVRCollector,
        EpisodeFrame,
        create_episode_buffer,
        find_next_episode_index,
        record_episode_interactive,
        save_episode,
    )


def _normalize_camera_names(camera_names: tuple[str, ...] | list[str]) -> tuple[str, ...]:
    names = tuple(str(name) for name in camera_names)
    if not names:
        raise ValueError("camera_names cannot be empty")
    return names


def _single_vr_capture(collector: DualVRCollector, side: str, frame_idx: int) -> tuple[Optional[EpisodeFrame], Optional[str]]:
    frame, error = collector.capture_frame(frame_idx)
    if frame is None:
        return None, error

    if side == "left":
        sl = slice(0, 7)
    elif side == "right":
        sl = slice(7, 14)
    else:
        raise ValueError("leader_side must be 'left' or 'right'")

    return EpisodeFrame(
        frame_idx=int(frame.frame_idx),
        timestamp=float(frame.timestamp),
        qpos=np.asarray(frame.qpos[sl], dtype=np.float32).copy(),
        qvel=np.asarray(frame.qvel[sl], dtype=np.float32).copy(),
        effort=np.asarray(frame.effort[sl], dtype=np.float32).copy(),
        eef=np.asarray(frame.eef[sl], dtype=np.float32).copy(),
        action=np.asarray(frame.action[sl], dtype=np.float32).copy(),
        images=dict(frame.images),
        images_depth=dict(frame.images_depth),
        topic_stamps=dict(frame.topic_stamps),
    ), None


def collect_vr_episode(
    env: ARXRobotEnv,
    arm_mode: str = "dual",
    out_dir: Path | str = "episodes_raw",
    frame_rate: float = 15.0,
    max_frames: int = 0,
    max_episodes: int = 0,
    action_kind: str = "joint",
    camera_names: tuple[str, ...] = ("camera_h",),
    with_depth: bool = False,
    img_size: tuple[int, int] = (640, 480),
    task: str = "",
    leader_side: str = "left",
    include_base: bool = False,
) -> Path | None:
    arm_mode = str(arm_mode).strip().lower()
    camera_names = _normalize_camera_names(camera_names)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    last_saved_episode: Path | None = None
    saved_episodes = 0

    if action_kind not in {"joint", "eef"}:
        raise ValueError("VR collect only supports action_kind='joint' or 'eef'")

    collector = DualVRCollector(
        env=env,
        camera_names=camera_names,
        include_camera=True,
        include_base=(arm_mode == "dual" and include_base),
        use_depth=with_depth,
        action_kind=action_kind,
        left_vr_topic="/ARX_VR_L",
        right_vr_topic="/ARX_VR_R",
        img_size=img_size,
    )

    if arm_mode == "single":
        start_prompt = "Press Enter to start VR single-arm recording, or type 'q' to quit: "
    elif arm_mode == "dual":
        start_prompt = "Press Enter to start VR dual-arm recording, or type 'q' to quit: "
    else:
        raise ValueError("arm_mode must be 'single' or 'dual'")

    try:
        collector.wait_until_ready()
        while True:
            if max_episodes > 0 and saved_episodes >= int(max_episodes):
                print(f"Reached max saved episodes: {saved_episodes}")
                return last_saved_episode

            command = input(start_prompt).strip().lower()
            if command == "q":
                return last_saved_episode
            if command != "":
                print("Invalid choice. Press Enter to start or 'q' to quit.")
                continue

            collector.reset_episode()
            if arm_mode == "single":
                episode = create_episode_buffer(
                    episode_idx=find_next_episode_index(out_dir),
                    mode="single",
                    frame_rate=frame_rate,
                    action_kind=action_kind,
                    include_camera=True,
                    include_base=False,
                    camera_names=camera_names,
                    config={
                        "task": task,
                        "mode": "single",
                        "collection_kind": "vr_single",
                        "action_kind": action_kind,
                        "leader_side": leader_side,
                        "camera_names": list(camera_names),
                        "with_depth": with_depth,
                    },
                    side=str(leader_side),
                )
                capture_fn = lambda frame_idx: _single_vr_capture(collector, str(leader_side), frame_idx)
            else:
                episode = create_episode_buffer(
                    episode_idx=find_next_episode_index(out_dir),
                    mode="dual",
                    frame_rate=frame_rate,
                    action_kind=action_kind,
                    include_camera=True,
                    include_base=include_base,
                    camera_names=camera_names,
                    config={
                        "task": task,
                        "mode": "dual",
                        "collection_kind": "vr_dual",
                        "action_kind": action_kind,
                        "include_base": include_base,
                        "camera_names": list(camera_names),
                        "with_depth": with_depth,
                    },
                )
                capture_fn = collector.capture_frame

            quit_requested = record_episode_interactive(
                episode=episode,
                capture_fn=capture_fn,
                frame_rate=frame_rate,
                max_frames=max_frames,
                prompt_start=False,
            )

            if episode.frame_count == 0:
                print("No frames captured; episode discarded.")
                if quit_requested:
                    return last_saved_episode
                continue

            while True:
                save_choice = input("Save episode? [y] save / [n] discard / [q] quit: ").strip().lower()
                if save_choice in {"", "y"}:
                    episode_dir = save_episode(episode, out_dir)
                    print(f"Saved episode to {episode_dir}")
                    last_saved_episode = episode_dir
                    saved_episodes += 1
                    break
                if save_choice == "n":
                    print("Episode discarded.")
                    break
                if save_choice == "q":
                    print("Episode discarded. Quit requested.")
                    return last_saved_episode
                print("Invalid choice. Please enter y, n, or q.")

            if quit_requested:
                print("Quit requested.")
                return last_saved_episode
    finally:
        collector.close()


def main() -> None:
    env = ARXRobotEnv(
        camera_type="color",
        camera_view=("camera_h", "camera_l", "camera_r"),
        dir=None,
        video=False,
        img_size=(640, 480),
    )
    try:
        collect_vr_episode(
            env,
            arm_mode="dual",
            out_dir="episodes_raw/vr_dual",
            frame_rate=15.0,
            action_kind="joint",
            camera_names=("camera_h", "camera_l", "camera_r"),
            include_base=False,
        )
        # collect_vr_episode(env, arm_mode="single", out_dir="episodes_raw/vr_single", leader_side="left")
    finally:
        env.close()


if __name__ == "__main__":
    main()
