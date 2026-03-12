from __future__ import annotations

from pathlib import Path

try:
    from collect_utils import (
        ARXRobotEnv,
        DualVRCollector,
        create_episode_buffer,
        find_next_episode_index,
        record_episode_interactive,
        save_episode,
    )
except ImportError:  # pragma: no cover
    from .collect_utils import (
        ARXRobotEnv,
        DualVRCollector,
        create_episode_buffer,
        find_next_episode_index,
        record_episode_interactive,
        save_episode,
    )


def collect_vr_two_arms_episode(
    env: ARXRobotEnv,
    out_dir: Path | str = "episodes_raw",
    camera_names: tuple[str, ...] = ("camera_h",),
    include_camera: bool = True,
    include_base: bool = True,
    use_depth: bool = False,
    frame_rate: float = 15.0,
    max_frames: int = 0,
    max_episodes: int = 0,
    task: str = "",
    action_kind: str = "joint",
    left_vr_topic: str = "/ARX_VR_L",
    right_vr_topic: str = "/ARX_VR_R",
    img_size: tuple[int, int] = (640, 480),
) -> Path | None:
    collector = DualVRCollector(
        env=env,
        camera_names=camera_names,
        include_camera=include_camera,
        include_base=include_base,
        use_depth=use_depth,
        action_kind=action_kind,
        left_vr_topic=left_vr_topic,
        right_vr_topic=right_vr_topic,
        img_size=img_size,
    )
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    last_saved_episode: Path | None = None
    saved_episodes = 0
    try:
        collector.wait_until_ready()
        while True:
            if max_episodes > 0 and saved_episodes >= int(max_episodes):
                print(f"Reached max saved episodes: {saved_episodes}")
                return last_saved_episode
            command = input("Press Enter to home arms and start recording, or type 'q' to quit: ").strip().lower()
            if command == "q":
                return last_saved_episode
            if command != "":
                print("Invalid choice. Press Enter to start or 'q' to quit.")
                continue

            success, error_message = collector.env.set_special_mode(1)
            if not success:
                raise RuntimeError(f"Failed to home arms: {error_message}")
            collector.reset_episode()
            episode = create_episode_buffer(
                episode_idx=find_next_episode_index(out_dir),
                mode="dual",
                frame_rate=frame_rate,
                action_kind=action_kind,
                include_camera=include_camera,
                include_base=include_base,
                camera_names=camera_names,
                config={
                    "task": task,
                    "mode": "dual",
                    "action_kind": action_kind,
                    "include_camera": include_camera,
                    "include_base": include_base,
                    "camera_names": list(camera_names),
                    "use_depth": use_depth,
                },
            )
            quit_requested = record_episode_interactive(
                episode=episode,
                capture_fn=collector.capture_frame,
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
    env = ARXRobotEnv(camera_type="color", camera_view=("camera_h",), dir=None, video=False, img_size=(640, 480))
    try:
        collect_vr_two_arms_episode(env)
    finally:
        env.close()


if __name__ == "__main__":
    main()
