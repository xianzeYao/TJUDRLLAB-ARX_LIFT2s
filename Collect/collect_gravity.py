from __future__ import annotations

from pathlib import Path
from typing import Optional

try:
    from collect_utils import (
        ARXRobotEnv,
        DualArmGravityCollector,
        SingleArmMirrorCollector,
        create_episode_buffer,
        find_next_episode_index,
        record_episode_interactive,
        save_episode,
    )
except ImportError:  # pragma: no cover
    from collect_utils import (
        ARXRobotEnv,
        DualArmGravityCollector,
        SingleArmMirrorCollector,
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


def collect_gravity_episode(
    env: ARXRobotEnv,
    arm_mode: str = "single",
    out_dir: Path | str = "episodes_raw",
    frame_rate: float = 20.0,
    max_frames: int = 0,
    max_episodes: int = 0,
    action_kind: str = "joint",
    camera_names: tuple[str, ...] = ("camera_h",),
    with_depth: bool = False,
    img_size: tuple[int, int] = (640, 480),
    task: str = "stack the two paper cups on top of the paper cup closest to the shelf one by one and place the stacked cups on the shelf",
    leader_side: str = "left",
    mirror: bool = True,
    control_rate: float = 35.0,
    joint_lowpass_alpha: float = 1.0,
    joint_deadband: float | tuple[float, ...] = 0.0,
    eef_lowpass_alpha: float = 0.20,
    eef_deadband: float | tuple[float, ...] = (
        0.0012,
        0.0012,
        0.0012,
        0.012,
        0.012,
        0.012,
        0.02,
    ),
) -> Path | None:
    arm_mode = str(arm_mode).strip().lower()
    camera_names = _normalize_camera_names(camera_names)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    last_saved_episode: Path | None = None
    saved_episodes = 0

    if arm_mode == "single":
        collector = SingleArmMirrorCollector(
            env=env,
            leader_side=str(leader_side),
            camera_names=camera_names,
            include_camera=True,
            use_depth=with_depth,
            action_kind=action_kind,
            mirror=mirror,
            img_size=img_size,
            control_rate=control_rate,
            joint_lowpass_alpha=joint_lowpass_alpha,
            joint_deadband=joint_deadband,
            eef_lowpass_alpha=eef_lowpass_alpha,
            eef_deadband=eef_deadband,
        )
        start_prompt = "Press Enter to start gravity single-arm recording, or type 'q' to quit: "
    elif arm_mode == "dual":
        collector = DualArmGravityCollector(
            env=env,
            camera_names=camera_names,
            include_camera=True,
            use_depth=with_depth,
            action_kind=action_kind,
            img_size=img_size,
        )
        start_prompt = "Press Enter to start gravity dual-arm recording, or type 'q' to quit: "
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

            collector.prepare()
            try:
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
                            "collection_kind": "gravity_single",
                            "action_kind": action_kind,
                            "leader_side": leader_side,
                            "mirror": mirror,
                            "record_side": collector.record_side,
                            "camera_names": list(camera_names),
                            "with_depth": with_depth,
                            "control_rate": control_rate,
                            "joint_lowpass_alpha": joint_lowpass_alpha,
                            "joint_deadband": joint_deadband,
                            "eef_lowpass_alpha": eef_lowpass_alpha,
                            "eef_deadband": eef_deadband,
                        },
                        side=collector.record_side,
                    )
                else:
                    episode = create_episode_buffer(
                        episode_idx=find_next_episode_index(out_dir),
                        mode="dual",
                        frame_rate=frame_rate,
                        action_kind=action_kind,
                        include_camera=True,
                        include_base=False,
                        camera_names=camera_names,
                        config={
                            "task": task,
                            "mode": "dual",
                            "collection_kind": "gravity_dual",
                            "action_kind": action_kind,
                            "camera_names": list(camera_names),
                            "with_depth": with_depth,
                        },
                    )

                quit_requested = record_episode_interactive(
                    episode=episode,
                    capture_fn=collector.capture_frame,
                    frame_rate=frame_rate,
                    max_frames=max_frames,
                    prompt_start=False,
                )
            finally:
                if arm_mode == "single":
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
    env = ARXRobotEnv(camera_type="color", camera_view=(
        "camera_h", "camera_l", "camera_r"), dir=None, video=False, img_size=(640, 480))
    try:
        env.reset()
        env.step_lift(14.5)
        collect_gravity_episode(
            env,
            arm_mode="single",
            out_dir="episodes_raw/gravity_single",
            frame_rate=20.0,
            max_frames=0,
            max_episodes=0,
            action_kind="joint",
            camera_names=("camera_h", "camera_l", "camera_r"),
            with_depth=False,
            img_size=(640, 480),
            task="",
            leader_side="left",
            mirror=True,
            control_rate=35.0,
            joint_lowpass_alpha=1.0,
            joint_deadband=0.0,
            eef_lowpass_alpha=0.2,
            eef_deadband=(0.0012, 0.0012, 0.0012, 0.012, 0.012, 0.012, 0.02),
        )
        # collect_gravity_episode(
        #     env,
        #     arm_mode="dual",
        #     out_dir="episodes_raw/gravity_dual",
        #     frame_rate=20.0,
        #     max_frames=0,
        #     max_episodes=0,
        #     action_kind="joint",
        #     camera_names=("camera_h",),
        #     with_depth=False,
        #     img_size=(640, 480),
        #     task="",
        #     leader_side="left",
        #     mirror=True,
        #     control_rate=35.0,
        #     joint_lowpass_alpha=1.0,
        #     joint_deadband=0.0,
        #     eef_lowpass_alpha=0.20,
        #     eef_deadband=(0.0012, 0.0012, 0.0012, 0.012, 0.012, 0.012, 0.02),
        # )
    finally:
        env.close()


if __name__ == "__main__":
    main()
