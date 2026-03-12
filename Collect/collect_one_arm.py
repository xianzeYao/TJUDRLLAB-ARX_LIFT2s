from __future__ import annotations

from pathlib import Path

try:
    from collect_utils import (
        ARXRobotEnv,
        SingleArmMirrorCollector,
        create_episode_buffer,
        find_next_episode_index,
        record_episode_interactive,
        save_episode,
    )
except ImportError:  # pragma: no cover
    from .collect_utils import (
        ARXRobotEnv,
        SingleArmMirrorCollector,
        create_episode_buffer,
        find_next_episode_index,
        record_episode_interactive,
        save_episode,
    )


def collect_one_arm_episode(
    env: ARXRobotEnv,
    leader_side: str = "left",
    out_dir: Path | str = "episodes_raw",
    frame_rate: float = 15.0,
    control_rate: float = 80.0,
    max_frames: int = 0,
    max_episodes: int = 0,
    action_kind: str = "joint",
    include_camera: bool = False,
    camera_names: tuple[str, ...] = (),
    use_depth: bool = False,
    mirror: bool = True,
    img_size: tuple[int, int] = (640, 480),
    joint_lowpass_alpha: float = 0.2,
    joint_deadband: float | tuple[float, ...] = 0.004,
    eef_lowpass_alpha: float = 0.25,
    task: str = "",
    eef_deadband: float | tuple[float, ...] = (
        0.001,
        0.001,
        0.001,
        0.01,
        0.01,
        0.01,
        0.02,
    ),
) -> Path | None:
    if leader_side not in {"left", "right"}:
        raise ValueError("leader_side must be 'left' or 'right'")
    if action_kind not in {"joint", "eef"}:
        raise ValueError(
            "single-arm collect only supports action_kind='joint' or 'eef'")

    collector = SingleArmMirrorCollector(
        env=env,
        leader_side=leader_side,
        camera_names=camera_names,
        include_camera=include_camera,
        use_depth=use_depth,
        action_kind=action_kind,
        mirror=mirror,
        img_size=img_size,
        control_rate=control_rate,
        joint_lowpass_alpha=joint_lowpass_alpha,
        joint_deadband=joint_deadband,
        eef_lowpass_alpha=eef_lowpass_alpha,
        eef_deadband=eef_deadband,
    )
    record_side = collector.record_side
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

            command = input(
                "Press Enter to enable gravity and start recording, or type 'q' to quit: ").strip().lower()
            if command == "q":
                return last_saved_episode
            if command != "":
                print("Invalid choice. Press Enter to start or 'q' to quit.")
                continue

            collector.prepare()
            episode = create_episode_buffer(
                episode_idx=find_next_episode_index(out_dir),
                mode="single",
                frame_rate=frame_rate,
                action_kind=action_kind,
                include_camera=include_camera,
                include_base=False,
                camera_names=camera_names,
                config={
                    "mode": "single",
                    "side": record_side,
                    "leader_side": leader_side,
                    "follower_side": collector.follow_side,
                    "record_side": record_side,
                    "action_kind": action_kind,
                    "mirror": mirror,
                    "save_rate": frame_rate,
                    "control_rate": control_rate,
                    "joint_lowpass_alpha": joint_lowpass_alpha,
                    "joint_deadband": joint_deadband,
                    "eef_lowpass_alpha": eef_lowpass_alpha,
                    "eef_deadband": list(eef_deadband) if isinstance(eef_deadband, tuple) else eef_deadband,
                    "include_camera": include_camera,
                    "camera_names": list(camera_names),
                    "use_depth": use_depth,
                    "task": task,
                },
                side=record_side,
            )
            quit_requested = record_episode_interactive(
                episode=episode,
                capture_fn=collector.capture_frame,
                frame_rate=frame_rate,
                max_frames=max_frames,
                prompt_start=False,
            )
            collector.stop_control()
            if episode.frame_count == 0:
                print("No frames captured; episode discarded.")
                if quit_requested:
                    return last_saved_episode
                continue

            while True:
                save_choice = input(
                    "Save episode? [y] save / [n] discard / [q] quit: ").strip().lower()
                if save_choice in {"", "y"}:
                    episode_dir = save_episode(episode, out_dir)
                    print(f"Saved single-arm episode to {episode_dir}")
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
    arx = ARXRobotEnv(camera_type="color", camera_view=(),
                      dir=None, video=False, img_size=(640, 480))
    try:
        arx.reset()
        arx.step_lift(14.5)
        collect_one_arm_episode(
            arx,
            leader_side="left",
            out_dir="episodes_raw/test",
            action_kind="eef",
            mirror=True,
            frame_rate=15,
            control_rate=50,
        )
    finally:
        arx.close()


if __name__ == "__main__":
    main()
