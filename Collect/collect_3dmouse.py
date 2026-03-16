from __future__ import annotations

from pathlib import Path
from typing import Optional

try:
    from collect_utils import (
        ARXRobotEnv,
        DualArmSpaceMouseCollector,
        SingleArmSpaceMouseCollector,
        create_episode_buffer,
        find_next_episode_index,
        record_episode_interactive,
        save_episode,
    )
except ImportError:  # pragma: no cover
    from .collect_utils import (
        ARXRobotEnv,
        DualArmSpaceMouseCollector,
        SingleArmSpaceMouseCollector,
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


def collect_3dmouse_episode(
    env: ARXRobotEnv,
    arm_mode: str = "single",
    out_dir: Path | str = "episodes_raw",
    frame_rate: float = 15.0,
    max_frames: int = 0,
    max_episodes: int = 0,
    action_kind: str = "eef",
    camera_names: tuple[str, ...] = ("camera_h",),
    with_depth: bool = False,
    img_size: tuple[int, int] = (640, 480),
    task: str = "",
    leader_side: str = "left",
    control_rate: float = 60.0,
    translation_scale: float = 0.10,
    rotation_scale: float = 0.60,
    gripper_step: float = 2.5,
    translation_deadzone: float = 0.05,
    rotation_deadzone: float = 0.05,
    response_exponent: float = 1.5,
    translation_axis_signs: tuple[float, float, float] = (1.0, 1.0, 1.0),
    rotation_axis_signs: tuple[float, float, float] = (1.0, 1.0, 1.0),
    home_on_start: bool = True,
) -> Path | None:
    arm_mode = str(arm_mode).strip().lower()
    camera_names = _normalize_camera_names(camera_names)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    last_saved_episode: Path | None = None
    saved_episodes = 0

    if action_kind != "eef":
        raise ValueError("3D mouse collect only supports action_kind='eef'")

    if arm_mode == "single":
        collector = SingleArmSpaceMouseCollector(
            env=env,
            side=str(leader_side),
            camera_names=camera_names,
            include_camera=True,
            use_depth=with_depth,
            img_size=img_size,
            control_rate=control_rate,
            translation_speed=translation_scale,
            rotation_speed=rotation_scale,
            gripper_speed=gripper_step,
            translation_deadzone=translation_deadzone,
            rotation_deadzone=rotation_deadzone,
            response_exponent=response_exponent,
            translation_axis_signs=translation_axis_signs,
            rotation_axis_signs=rotation_axis_signs,
        )
        start_prompt = "Press Enter to start 3D mouse single-arm recording, or type 'q' to quit: "
    elif arm_mode == "dual":
        collector = DualArmSpaceMouseCollector(
            env=env,
            camera_names=camera_names,
            include_camera=True,
            use_depth=with_depth,
            img_size=img_size,
            control_rate=control_rate,
            translation_speed=translation_scale,
            rotation_speed=rotation_scale,
            gripper_speed=gripper_step,
            translation_deadzone=translation_deadzone,
            rotation_deadzone=rotation_deadzone,
            response_exponent=response_exponent,
            translation_axis_signs=translation_axis_signs,
            rotation_axis_signs=rotation_axis_signs,
            initial_active_side=str(leader_side),
        )
        start_prompt = "Press Enter to start 3D mouse dual-arm recording, or type 'q' to quit: "
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

            if home_on_start:
                if arm_mode == "single":
                    success, error_message = env.set_special_mode(1, side=str(leader_side))
                    if not success:
                        raise RuntimeError(f"Failed to home {leader_side} arm: {error_message}")
                else:
                    success, error_message = env.set_special_mode(1)
                    if not success:
                        raise RuntimeError(f"Failed to home both arms: {error_message}")

            collector.prepare()
            try:
                if arm_mode == "single":
                    episode = create_episode_buffer(
                        episode_idx=find_next_episode_index(out_dir),
                        mode="single",
                        frame_rate=frame_rate,
                        action_kind="eef",
                        include_camera=True,
                        include_base=False,
                        camera_names=camera_names,
                        config={
                            "task": task,
                            "mode": "single",
                            "collection_kind": "3dmouse_single",
                            "action_kind": "eef",
                            "side": leader_side,
                            "camera_names": list(camera_names),
                            "with_depth": with_depth,
                            "control_rate": control_rate,
                            "translation_scale": translation_scale,
                            "rotation_scale": rotation_scale,
                            "gripper_step": gripper_step,
                            "translation_deadzone": translation_deadzone,
                            "rotation_deadzone": rotation_deadzone,
                            "response_exponent": response_exponent,
                            "translation_axis_signs": list(translation_axis_signs),
                            "rotation_axis_signs": list(rotation_axis_signs),
                        },
                        side=str(leader_side),
                    )
                else:
                    episode = create_episode_buffer(
                        episode_idx=find_next_episode_index(out_dir),
                        mode="dual",
                        frame_rate=frame_rate,
                        action_kind="eef",
                        include_camera=True,
                        include_base=False,
                        camera_names=camera_names,
                        config={
                            "task": task,
                            "mode": "dual",
                            "collection_kind": "3dmouse_dual",
                            "action_kind": "eef",
                            "leader_side": leader_side,
                            "camera_names": list(camera_names),
                            "with_depth": with_depth,
                            "control_rate": control_rate,
                            "translation_scale": translation_scale,
                            "rotation_scale": rotation_scale,
                            "gripper_step": gripper_step,
                            "translation_deadzone": translation_deadzone,
                            "rotation_deadzone": rotation_deadzone,
                            "response_exponent": response_exponent,
                            "translation_axis_signs": list(translation_axis_signs),
                            "rotation_axis_signs": list(rotation_axis_signs),
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
                collector.stop_control()

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
        collect_3dmouse_episode(
            env,
            arm_mode="single",
            out_dir="episodes_raw/3dmouse_single",
            frame_rate=15.0,
            leader_side="left",
        )
        # collect_3dmouse_episode(env, arm_mode="dual", out_dir="episodes_raw/3dmouse_dual")
    finally:
        env.close()


if __name__ == "__main__":
    main()
