from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Optional

from base_calib_collect import (
    CommandSpec,
    ImuTracker,
    collect_samples,
    ensure_output_dir,
    query_robot_type,
    write_jsonl,
)

THIS_DIR = Path(__file__).resolve().parent
ROS2_DIR = (THIS_DIR / "../ROS2").resolve()
if str(ROS2_DIR) not in sys.path:
    sys.path.insert(0, str(ROS2_DIR))

DEFAULT_OUT_DIR = THIS_DIR / "Testdata4NavSpeed"
DEFAULT_VX_CMD = 0.75
DEFAULT_VY_CMD = 0.75
DEFAULT_DURATIONS = (2.0, 4.0, 6.0)
DEFAULT_SAMPLE_HZ = 20.0
DEFAULT_PRE_STOP_DURATION_S = 0.5
DEFAULT_POST_STOP_DURATION_S = 0.5


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Collect fixed-duration translation runs and optional manual displacement measurements."
    )
    parser.add_argument(
        "--axis",
        choices=("x", "y", "both"),
        default="both",
        help="Select which translation axis to test in this run.",
    )
    parser.add_argument("--vx-cmd", type=float, default=DEFAULT_VX_CMD)
    parser.add_argument("--vy-cmd", type=float, default=DEFAULT_VY_CMD)
    parser.add_argument(
        "--durations",
        type=str,
        default=",".join(str(x) for x in DEFAULT_DURATIONS),
        help="Comma-separated durations in seconds, e.g. 2,4,6",
    )
    parser.add_argument("--sample-hz", type=float, default=DEFAULT_SAMPLE_HZ)
    parser.add_argument("--pre-stop-duration", type=float,
                        default=DEFAULT_PRE_STOP_DURATION_S)
    parser.add_argument("--post-stop-duration", type=float,
                        default=DEFAULT_POST_STOP_DURATION_S)
    parser.add_argument("--repeat", type=int, default=1)
    parser.add_argument("--include-negative", action="store_true")
    parser.add_argument("--skip-reset", action="store_true")
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT_DIR)
    return parser.parse_args()


def parse_durations(raw: str) -> list[float]:
    values: list[float] = []
    for token in raw.split(","):
        token = token.strip()
        if not token:
            continue
        values.append(float(token))
    if not values:
        raise ValueError(
            "--durations must contain at least one positive value")
    if any(v <= 0.0 for v in values):
        raise ValueError("all durations must be > 0")
    return values


def build_sequence(
    axis: str,
    vx_cmd: float,
    vy_cmd: float,
    include_negative: bool,
) -> list[CommandSpec]:
    seq: list[CommandSpec] = []
    if axis in ("x", "both"):
        seq.append(CommandSpec("vx_pos", vx_cmd, 0.0, 0.0))
        if include_negative:
            seq.append(CommandSpec("vx_neg", -vx_cmd, 0.0, 0.0))
    if axis in ("y", "both"):
        seq.append(CommandSpec("vy_pos", 0.0, vy_cmd, 0.0))
        if include_negative:
            seq.append(CommandSpec("vy_neg", 0.0, -vy_cmd, 0.0))
    return seq


def prompt_float(label: str) -> Optional[float]:
    if not sys.stdin.isatty():
        return None
    try:
        raw = input(label).strip()
    except EOFError:
        return None
    if not raw:
        return None
    return float(raw)


def main() -> None:
    args = parse_args()
    if args.sample_hz <= 0.0:
        raise ValueError("--sample-hz must be > 0")
    if args.repeat <= 0:
        raise ValueError("--repeat must be > 0")
    durations = parse_durations(args.durations)
    robot_type, robot_type_source = query_robot_type()
    out_root = ensure_output_dir(args.out)
    session_name = datetime.now().strftime("session_%Y%m%d_%H%M%S")
    out_dir = ensure_output_dir(out_root / session_name)

    from arx_ros2_env import ARXRobotEnv

    arx = ARXRobotEnv(
        duration_per_step=1.0 / 20.0,
        min_steps=20,
        max_v_xyz=0.15,
        max_a_xyz=0.20,
        max_v_rpy=0.45,
        max_a_rpy=1.00,
        camera_type="all",
        camera_view=("camera_h",),
        img_size=(640, 480),
    )
    imu_tracker = ImuTracker(arx.node)
    measurements: list[dict] = []
    sequence = build_sequence(
        axis=str(args.axis),
        vx_cmd=float(args.vx_cmd),
        vy_cmd=float(args.vy_cmd),
        include_negative=bool(args.include_negative),
    )
    try:
        if not args.skip_reset:
            arx.reset()
        for repeat_idx in range(args.repeat):
            for duration_s in durations:
                for cmd in sequence:
                    print(
                        f"Running {cmd.label} for {duration_s:.2f}s "
                        f"(vx={cmd.vx:.2f}, vy={cmd.vy:.2f}, vz={cmd.vz:.2f})"
                    )
                    rows = collect_samples(
                        arx=arx,
                        imu_tracker=imu_tracker,
                        cmd=cmd,
                        command_duration=float(duration_s),
                        pre_stop_duration=float(args.pre_stop_duration),
                        post_stop_duration=float(args.post_stop_duration),
                        sample_hz=float(args.sample_hz),
                        robot_type=robot_type,
                        robot_type_source=robot_type_source,
                        repeat_index=repeat_idx,
                    )
                    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                    out_path = out_dir / (
                        f"{repeat_idx:02d}_{cmd.label}_{duration_s:.2f}s_{stamp}.jsonl"
                    )
                    write_jsonl(out_path, rows)
                    primary = prompt_float(
                        "Measured primary displacement in meters (blank to skip): "
                    )
                    cross = prompt_float(
                        "Measured cross displacement in meters (blank to skip): "
                    )
                    measurements.append(
                        {
                            "repeat_index": repeat_idx,
                            "command_label": cmd.label,
                            "cmd_vx": cmd.vx,
                            "cmd_vy": cmd.vy,
                            "cmd_vz": cmd.vz,
                            "duration_s": float(duration_s),
                            "robot_type": int(robot_type),
                            "robot_type_source": robot_type_source,
                            "data_file": out_path.name,
                            "measured_primary_displacement_m": primary,
                            "measured_cross_displacement_m": cross,
                            "estimated_primary_speed_mps": (
                                None if primary is None else float(
                                    primary) / float(duration_s)
                            ),
                        }
                    )
                    print(f"saved -> {out_path}")
        manifest = {
            "created_at": datetime.now().isoformat(timespec="seconds"),
            "robot_type": int(robot_type),
            "robot_type_source": robot_type_source,
            "defaults": {
                "axis": str(args.axis),
                "vx_cmd": float(args.vx_cmd),
                "vy_cmd": float(args.vy_cmd),
                "durations": durations,
                "sample_hz": float(args.sample_hz),
                "repeat": int(args.repeat),
                "include_negative": bool(args.include_negative),
            },
            "measurements": measurements,
            "goal": "estimate real translation speed using fixed-duration runs and manual displacement measurements",
        }
        manifest_path = out_dir / "measurements.json"
        manifest_path.write_text(
            json.dumps(manifest, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        print(f"saved measurement summary -> {manifest_path}")
    finally:
        try:
            arx.step_base(0.0, 0.0, 0.0)
        except Exception:
            pass
        arx.close()


if __name__ == "__main__":
    main()
