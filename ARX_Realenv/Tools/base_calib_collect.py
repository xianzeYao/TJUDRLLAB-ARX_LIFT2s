from __future__ import annotations

import argparse
import csv
import json
import math
import signal
import subprocess
import sys
import threading
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable, Optional

THIS_DIR = Path(__file__).resolve().parent
ROOT_DIR = THIS_DIR.parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

WHEEL_RADIUS_M = 0.150
CENTER_TO_WHEEL_M = 0.376
WHEEL_MAPPING = {
    "base_wheel1": "rear",
    "base_wheel2": "front_right",
    "base_wheel3": "front_left",
}
DEFAULT_VX_CMD = 0.75
DEFAULT_VY_CMD = 0.75
DEFAULT_VZ_CMD = 1.0
DEFAULT_COMMAND_DURATION_S = 2.5
DEFAULT_PRE_STOP_DURATION_S = 0.5
DEFAULT_POST_STOP_DURATION_S = 0.5
DEFAULT_SAMPLE_HZ = 20.0
DEFAULT_ROBOT_TYPE = 0
DEFAULT_IMU_TOPIC = "/arx_imu"


@dataclass(frozen=True)
class CommandSpec:
    label: str
    vx: float
    vy: float
    vz: float


class ImuTracker:
    def __init__(self, node, topic: str = DEFAULT_IMU_TOPIC) -> None:
        from sensor_msgs.msg import Imu

        self._lock = threading.Lock()
        self._latest_msg: Optional[Any] = None
        self._latest_wall_time: Optional[float] = None
        self._subscription = node.create_subscription(
            Imu, topic, self._on_imu, 1)

    def _on_imu(self, msg) -> None:
        with self._lock:
            self._latest_msg = msg
            self._latest_wall_time = time.time()

    def snapshot(self) -> dict[str, Any]:
        with self._lock:
            msg = self._latest_msg
            wall_time = self._latest_wall_time

        if msg is None:
            return {
                "imu_available": False,
                "imu_wall_time": wall_time,
                "imu_gyro_x": None,
                "imu_gyro_y": None,
                "imu_gyro_z": None,
                "imu_orientation_x": None,
                "imu_orientation_y": None,
                "imu_orientation_z": None,
                "imu_orientation_w": None,
                "imu_yaw": None,
            }

        qx = float(msg.orientation.x)
        qy = float(msg.orientation.y)
        qz = float(msg.orientation.z)
        qw = float(msg.orientation.w)
        return {
            "imu_available": True,
            "imu_wall_time": wall_time,
            "imu_gyro_x": float(msg.angular_velocity.x),
            "imu_gyro_y": float(msg.angular_velocity.y),
            "imu_gyro_z": float(msg.angular_velocity.z),
            "imu_orientation_x": qx,
            "imu_orientation_y": qy,
            "imu_orientation_z": qz,
            "imu_orientation_w": qw,
            "imu_yaw": _quat_to_yaw(qx, qy, qz, qw),
        }


def _quat_to_yaw(qx: float, qy: float, qz: float, qw: float) -> float:
    siny_cosp = 2.0 * (qw * qz + qx * qy)
    cosy_cosp = 1.0 - 2.0 * (qy * qy + qz * qz)
    return float(math.atan2(siny_cosp, cosy_cosp))


def _parse_robot_type(stdout: str) -> Optional[int]:
    for token in reversed(stdout.replace(":", " ").split()):
        try:
            return int(token)
        except ValueError:
            continue
    return None


def query_robot_type() -> tuple[int, str]:
    candidates = [
        ("/lift", DEFAULT_ROBOT_TYPE),
        ("/x7s", 1),
    ]
    for node_name, fallback_value in candidates:
        try:
            proc = subprocess.run(
                ["ros2", "param", "get", node_name, "robot_type"],
                capture_output=True,
                text=True,
                timeout=3.0,
                check=False,
            )
        except (OSError, subprocess.SubprocessError):
            continue
        if proc.returncode != 0:
            continue
        value = _parse_robot_type(proc.stdout)
        if value is not None:
            return value, f"ros2 param get {node_name} robot_type"
        return fallback_value, f"ros2 param get {node_name} robot_type (fallback parse)"
    return DEFAULT_ROBOT_TYPE, "fallback_default"


def build_default_sequence(
    vx_cmd: float,
    vy_cmd: float,
    vz_cmd: float,
    include_negative: bool,
) -> list[CommandSpec]:
    sequence = [
        CommandSpec("vx_pos", vx_cmd, 0.0, 0.0),
        CommandSpec("vy_pos", 0.0, vy_cmd, 0.0),
        CommandSpec("vz_pos", 0.0, 0.0, vz_cmd),
    ]
    if include_negative:
        sequence.extend(
            [
                CommandSpec("vx_neg", -vx_cmd, 0.0, 0.0),
                CommandSpec("vy_neg", 0.0, -vy_cmd, 0.0),
                CommandSpec("vz_neg", 0.0, 0.0, -vz_cmd),
            ]
        )
    sequence.append(CommandSpec("stop", 0.0, 0.0, 0.0))
    return sequence


def format_cmd_value(value: float) -> str:
    if abs(value) < 1e-9:
        return "0"
    return f"{value:+.2f}".replace(".", "p")


def ensure_output_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fieldnames = sorted({key for row in rows for key in row.keys()})
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def collect_samples(
    arx: ARXRobotEnv,
    imu_tracker: ImuTracker,
    cmd: CommandSpec,
    *,
    command_duration: float,
    pre_stop_duration: float,
    post_stop_duration: float,
    sample_hz: float,
    robot_type: int,
    robot_type_source: str,
    repeat_index: int,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    period_s = 1.0 / sample_hz if sample_hz > 0 else 0.05
    run_started_at = time.time()

    _send_stop(arx)
    _sleep_until(pre_stop_duration)
    command_started_at = time.time()
    arx.step_base(cmd.vx, cmd.vy, cmd.vz)

    try:
        rows.extend(
            _record_phase(
                arx=arx,
                imu_tracker=imu_tracker,
                cmd=cmd,
                robot_type=robot_type,
                robot_type_source=robot_type_source,
                repeat_index=repeat_index,
                phase="command",
                phase_started_at=command_started_at,
                duration_s=command_duration,
                period_s=period_s,
                run_started_at=run_started_at,
            )
        )
    finally:
        _send_stop(arx)

    post_stop_started_at = time.time()
    rows.extend(
        _record_phase(
            arx=arx,
            imu_tracker=imu_tracker,
            cmd=CommandSpec(f"{cmd.label}_post_stop", 0.0, 0.0, 0.0),
            robot_type=robot_type,
            robot_type_source=robot_type_source,
            repeat_index=repeat_index,
            phase="post_stop",
            phase_started_at=post_stop_started_at,
            duration_s=post_stop_duration,
            period_s=period_s,
            run_started_at=run_started_at,
            requested_cmd=(cmd.vx, cmd.vy, cmd.vz),
        )
    )
    return rows


def _record_phase(
    *,
    arx: ARXRobotEnv,
    imu_tracker: ImuTracker,
    cmd: CommandSpec,
    robot_type: int,
    robot_type_source: str,
    repeat_index: int,
    phase: str,
    phase_started_at: float,
    duration_s: float,
    period_s: float,
    run_started_at: float,
    requested_cmd: Optional[tuple[float, float, float]] = None,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    end_time = phase_started_at + max(duration_s, 0.0)
    sample_index = 0

    while True:
        now = time.time()
        if now > end_time and sample_index > 0:
            break
        status = arx.get_robot_status()
        base_status = status.get("base") if isinstance(status, dict) else None
        imu_snapshot = imu_tracker.snapshot()
        row = build_row(
            base_status=base_status,
            imu_snapshot=imu_snapshot,
            label=cmd.label,
            cmd_vx=cmd.vx,
            cmd_vy=cmd.vy,
            cmd_vz=cmd.vz,
            repeat_index=repeat_index,
            robot_type=robot_type,
            robot_type_source=robot_type_source,
            phase=phase,
            sample_index=sample_index,
            wall_time=now,
            elapsed_from_run_start_s=now - run_started_at,
            elapsed_from_phase_start_s=now - phase_started_at,
            requested_cmd=requested_cmd,
        )
        rows.append(row)
        sample_index += 1

        next_tick = phase_started_at + sample_index * period_s
        sleep_need = next_tick - time.time()
        if sleep_need > 0:
            time.sleep(sleep_need)
        if time.time() > end_time and sample_index > 0:
            break
    return rows


def build_row(
    *,
    base_status,
    imu_snapshot: dict[str, Any],
    label: str,
    cmd_vx: float,
    cmd_vy: float,
    cmd_vz: float,
    repeat_index: int,
    robot_type: int,
    robot_type_source: str,
    phase: str,
    sample_index: int,
    wall_time: float,
    elapsed_from_run_start_s: float,
    elapsed_from_phase_start_s: float,
    requested_cmd: Optional[tuple[float, float, float]] = None,
) -> dict[str, Any]:
    temp_float_data = getattr(base_status, "temp_float_data", None)

    def _temp_at(index: int) -> Optional[float]:
        if temp_float_data is None:
            return None
        try:
            return float(temp_float_data[index])
        except Exception:
            return None

    row = {
        "timestamp_iso": datetime.fromtimestamp(wall_time).isoformat(timespec="milliseconds"),
        "timestamp_wall_s": wall_time,
        "elapsed_from_run_start_s": float(elapsed_from_run_start_s),
        "elapsed_from_phase_start_s": float(elapsed_from_phase_start_s),
        "phase": phase,
        "sample_index": sample_index,
        "command_label": label,
        "cmd_vx": float(cmd_vx),
        "cmd_vy": float(cmd_vy),
        "cmd_vz": float(cmd_vz),
        "repeat_index": int(repeat_index),
        "robot_type": int(robot_type),
        "robot_type_source": robot_type_source,
        "base_height": float(getattr(base_status, "height", float("nan")))
        if base_status is not None
        else None,
        "base_wheel1": _temp_at(1),
        "base_wheel2": _temp_at(2),
        "base_wheel3": _temp_at(3),
        "wheel1_name": WHEEL_MAPPING["base_wheel1"],
        "wheel2_name": WHEEL_MAPPING["base_wheel2"],
        "wheel3_name": WHEEL_MAPPING["base_wheel3"],
        "wheel_radius_m": WHEEL_RADIUS_M,
        "center_to_wheel_m": CENTER_TO_WHEEL_M,
        "body_forward_definition": "arrow_up",
    }
    if requested_cmd is not None:
        row["requested_cmd_vx"] = float(requested_cmd[0])
        row["requested_cmd_vy"] = float(requested_cmd[1])
        row["requested_cmd_vz"] = float(requested_cmd[2])
    row.update(imu_snapshot)
    return row


def _sleep_until(duration_s: float) -> None:
    if duration_s <= 0.0:
        return
    deadline = time.monotonic() + duration_s
    while True:
        remaining = deadline - time.monotonic()
        if remaining <= 0.0:
            return
        time.sleep(min(remaining, 0.05))


def _send_stop(arx: ARXRobotEnv) -> None:
    arx.step_base(0.0, 0.0, 0.0)


def dump_manifest(
    out_dir: Path,
    *,
    args: argparse.Namespace,
    robot_type: int,
    robot_type_source: str,
    sequence: list[CommandSpec],
) -> None:
    payload = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "robot_type": int(robot_type),
        "robot_type_source": robot_type_source,
        "wheel_radius_m": WHEEL_RADIUS_M,
        "center_to_wheel_m": CENTER_TO_WHEEL_M,
        "wheel_mapping": WHEEL_MAPPING,
        "body_forward_definition": "arrow_up",
        "defaults": {
            "vx_cmd": float(args.vx_cmd),
            "vy_cmd": float(args.vy_cmd),
            "vz_cmd": float(args.vz_cmd),
            "command_duration": float(args.duration),
            "pre_stop_duration": float(args.pre_stop_duration),
            "post_stop_duration": float(args.post_stop_duration),
            "sample_hz": float(args.sample_hz),
            "repeat": int(args.repeat),
            "include_negative": bool(args.include_negative),
        },
        "sequence": [
            {
                "label": item.label,
                "vx": item.vx,
                "vy": item.vy,
                "vz": item.vz,
            }
            for item in sequence
        ],
        "analysis_hint": {
            "discard_startup_window_s": min(0.5, float(args.duration) / 4.0),
            "summary": "Compute steady-state mean/std for base_wheel1/2/3 and imu_gyro_z.",
        },
    }
    manifest_path = out_dir / "manifest.json"
    manifest_path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Collect wheel-speed and IMU samples for ARX three-wheel omni-base calibration."
    )
    parser.add_argument("--vx-cmd", type=float, default=DEFAULT_VX_CMD)
    parser.add_argument("--vy-cmd", type=float, default=DEFAULT_VY_CMD)
    parser.add_argument("--vz-cmd", type=float, default=DEFAULT_VZ_CMD)
    parser.add_argument("--duration", type=float,
                        default=DEFAULT_COMMAND_DURATION_S)
    parser.add_argument("--pre-stop-duration", type=float,
                        default=DEFAULT_PRE_STOP_DURATION_S)
    parser.add_argument("--post-stop-duration", type=float,
                        default=DEFAULT_POST_STOP_DURATION_S)
    parser.add_argument("--sample-hz", type=float, default=DEFAULT_SAMPLE_HZ)
    parser.add_argument("--repeat", type=int, default=1)
    parser.add_argument("--include-negative", action="store_true")
    parser.add_argument(
        "--out",
        type=Path,
        default=THIS_DIR / "Testdata4Nav",
        help="Output directory for manifest and experiment files.",
    )
    parser.add_argument(
        "--format",
        choices=("jsonl", "csv"),
        default="jsonl",
        help="Output format for experiment files.",
    )
    parser.add_argument(
        "--skip-reset",
        action="store_true",
        help="Skip env.reset() if the robot is already in a safe ready state.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    from ARX_Realenv.ROS2.arx_ros2_env import ARXRobotEnv

    if args.duration < 0.0:
        raise ValueError("--duration must be >= 0")
    if args.pre_stop_duration < 0.0:
        raise ValueError("--pre-stop-duration must be >= 0")
    if args.post_stop_duration < 0.0:
        raise ValueError("--post-stop-duration must be >= 0")
    if args.sample_hz <= 0.0:
        raise ValueError("--sample-hz must be > 0")
    if args.repeat <= 0:
        raise ValueError("--repeat must be > 0")

    out_root = ensure_output_dir(args.out)
    session_name = datetime.now().strftime("session_%Y%m%d_%H%M%S")
    out_dir = ensure_output_dir(out_root / session_name)
    robot_type, robot_type_source = query_robot_type()
    sequence = build_default_sequence(
        vx_cmd=float(args.vx_cmd),
        vy_cmd=float(args.vy_cmd),
        vz_cmd=float(args.vz_cmd),
        include_negative=bool(args.include_negative),
    )

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
    stop_requested = False

    def _handle_signal(signum, _frame) -> None:
        nonlocal stop_requested
        stop_requested = True
        print(f"Signal {signum} received, stopping after current cycle.")

    old_sigint = signal.signal(signal.SIGINT, _handle_signal)
    old_sigterm = signal.signal(signal.SIGTERM, _handle_signal)
    try:
        dump_manifest(
            out_dir,
            args=args,
            robot_type=robot_type,
            robot_type_source=robot_type_source,
            sequence=sequence,
        )
        if not args.skip_reset:
            arx.reset()
        _send_stop(arx)
        _sleep_until(args.pre_stop_duration)

        for repeat_idx in range(args.repeat):
            if stop_requested:
                break
            for cmd in sequence:
                if stop_requested:
                    break
                rows = collect_samples(
                    arx=arx,
                    imu_tracker=imu_tracker,
                    cmd=cmd,
                    command_duration=float(args.duration),
                    pre_stop_duration=float(args.pre_stop_duration),
                    post_stop_duration=float(args.post_stop_duration),
                    sample_hz=float(args.sample_hz),
                    robot_type=robot_type,
                    robot_type_source=robot_type_source,
                    repeat_index=repeat_idx,
                )
                stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                cmd_tag = (
                    f"vx{format_cmd_value(cmd.vx)}_"
                    f"vy{format_cmd_value(cmd.vy)}_"
                    f"vz{format_cmd_value(cmd.vz)}"
                )
                suffix = "csv" if args.format == "csv" else "jsonl"
                out_path = out_dir / \
                    f"{repeat_idx:02d}_{cmd.label}_{cmd_tag}_{stamp}.{suffix}"
                if args.format == "csv":
                    write_csv(out_path, rows)
                else:
                    write_jsonl(out_path, rows)
                print(
                    f"saved {len(rows)} samples -> {out_path} "
                    f"(robot_type={robot_type}, source={robot_type_source})"
                )
    finally:
        signal.signal(signal.SIGINT, old_sigint)
        signal.signal(signal.SIGTERM, old_sigterm)
        try:
            _send_stop(arx)
        except Exception:
            pass
        arx.close()


if __name__ == "__main__":
    main()
