from __future__ import annotations

import argparse
import json
import sys
import threading
import time
import traceback
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

import numpy as np

CURRENT_DIR = Path(__file__).resolve().parent
ROOT_DIR = CURRENT_DIR.parent
ROS2_DIR = ROOT_DIR / "ARX_Realenv" / "ROS2"

if str(CURRENT_DIR) not in sys.path:
    sys.path.append(str(CURRENT_DIR))
if str(ROS2_DIR) not in sys.path:
    sys.path.append(str(ROS2_DIR))

from arx_ros2_env import ARXRobotEnv  # noqa: E402
from nav_dual_sweep import nav_dual_sweep  # noqa: E402
from smart_shelf_search import smart_shelf_search_from_request  # noqa: E402
from visualize_utils import VisualizeContext  # noqa: E402


def _serialize_json(value: Any) -> Any:
    if value is None or isinstance(value, (str, bool, int, float)):
        return value
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.integer, np.floating)):
        return value.item()
    if isinstance(value, dict):
        return {str(k): _serialize_json(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_serialize_json(v) for v in value]
    if hasattr(value, "__dict__"):
        return {
            str(k): _serialize_json(v)
            for k, v in vars(value).items()
            if not str(k).startswith("_")
        }
    return str(value)


def _dump_json(value: Any) -> str:
    return json.dumps(
        _serialize_json(value),
        ensure_ascii=False,
        indent=2,
        sort_keys=True,
    )


def _timestamp() -> str:
    return time.strftime("%H:%M:%S")


@dataclass
class RuntimeSnapshot:
    frames: dict[str, np.ndarray] = field(default_factory=dict)
    status: dict[str, Any] = field(default_factory=dict)
    nav_debug_image: Optional[np.ndarray] = None
    manip_debug_image: Optional[np.ndarray] = None
    logs: list[str] = field(default_factory=list)
    active_task: str = "idle"
    task_state: str = "idle"
    current_stage: str = "Waiting for task"
    parsed_request: Optional[dict[str, Any]] = None
    telemetry: dict[str, Any] = field(default_factory=dict)
    result: Optional[dict[str, Any]] = None
    last_error: str = ""


class DemoController:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._state = RuntimeSnapshot()
        self._env: Optional[ARXRobotEnv] = None
        self._camera_thread: Optional[threading.Thread] = None
        self._task_thread: Optional[threading.Thread] = None
        self._shutdown = threading.Event()
        self._stop_task = threading.Event()
        self._camera_error_logged = False

    def ensure_env(self) -> ARXRobotEnv:
        if self._env is None:
            self._append_log("Create ARX environment")
            self._env = ARXRobotEnv(
                duration_per_step=1.0 / 20.0,
                min_steps=20,
                max_v_xyz=0.15,
                max_a_xyz=0.1,
                max_v_rpy=0.5,
                max_a_rpy=0.6,
                camera_type="all",
                camera_view=("camera_l", "camera_h", "camera_r"),
                img_size=(640, 480),
            )
        self._start_camera_thread()
        return self._env

    def _start_camera_thread(self) -> None:
        if self._camera_thread is not None and self._camera_thread.is_alive():
            return
        self._camera_thread = threading.Thread(
            target=self._camera_loop,
            name="gradio-camera-loop",
            daemon=True,
        )
        self._camera_thread.start()

    def _camera_loop(self) -> None:
        while not self._shutdown.is_set():
            env = self._env
            if env is None:
                time.sleep(0.2)
                continue
            try:
                frames, status = env.get_camera(
                    target_size=(640, 480),
                    return_status=True,
                )
                with self._lock:
                    self._state.frames = dict(frames)
                    self._state.status = dict(status)
                    self._state.last_error = ""
                self._camera_error_logged = False
            except Exception as exc:
                with self._lock:
                    self._state.last_error = str(exc)
                if not self._camera_error_logged:
                    self._append_log(f"Camera poll failed: {exc}")
                    self._camera_error_logged = True
            time.sleep(0.15)

    def close(self) -> None:
        self._shutdown.set()
        self._stop_task.set()
        if self._env is not None:
            try:
                self._env.close()
            except Exception:
                pass

    def _append_log(self, message: str) -> None:
        line = f"[{_timestamp()}] {message}"
        with self._lock:
            self._state.logs.append(line)
            self._state.logs = self._state.logs[-200:]

    def _set_task_header(self, task_name: str, task_state: str, stage: str) -> None:
        with self._lock:
            self._state.active_task = task_name
            self._state.task_state = task_state
            self._state.current_stage = stage

    def handle_event(self, event: str, payload: dict[str, Any]) -> None:
        source = str(payload.get("source", "demo"))
        stage = payload.get("stage")
        message = payload.get("message")

        if event == "log":
            prefix = f"{source}"
            if stage:
                prefix += f"/{stage}"
            self._append_log(f"{prefix}: {message}")
            return

        if event == "stage":
            stage_text = str(message or stage or source)
            with self._lock:
                self._state.current_stage = stage_text
                self._state.telemetry["latest_stage"] = {
                    "source": source,
                    "stage": stage,
                    "message": message,
                }
            return

        if event == "debug":
            panel = str(payload.get("panel", "manip"))
            image = payload.get("image")
            if isinstance(image, np.ndarray):
                with self._lock:
                    if panel == "nav":
                        self._state.nav_debug_image = image
                    else:
                        self._state.manip_debug_image = image
            return

        with self._lock:
            if event == "parsed_request":
                self._state.parsed_request = dict(payload)
            elif event == "result":
                self._state.result = dict(payload)
                status = str(payload.get("status", "completed"))
                if status in {"success", "completed"}:
                    self._state.task_state = "completed"
                elif status in {"failed", "error"}:
                    self._state.task_state = "failed"
                elif status in {"stopped", "canceled"}:
                    self._state.task_state = status
            else:
                self._state.telemetry[event] = dict(payload)

    def _begin_task(self, task_name: str, debug_enabled: bool) -> Optional[str]:
        if self._task_thread is not None and self._task_thread.is_alive():
            return "Another task is still running."
        self.ensure_env()
        self._stop_task = threading.Event()
        with self._lock:
            self._state.active_task = task_name
            self._state.task_state = "starting"
            self._state.current_stage = "Preparing task"
            self._state.nav_debug_image = None
            self._state.manip_debug_image = None
            self._state.parsed_request = None
            self._state.telemetry = {"debug_enabled": debug_enabled}
            self._state.result = None
            self._state.logs = []
            self._state.last_error = ""
        return None

    def start_nav_dual_sweep(
        self,
        prompt: str,
        page_debug: bool,
        distance: float,
        nav_lift_height: float,
        nav_depth_median_n: int,
        swap_depth_median_n: int,
        vote_times: int,
    ) -> str:
        if not prompt.strip():
            return "Prompt is required."
        error = self._begin_task("nav dual sweep", page_debug)
        if error:
            return error

        def _runner() -> None:
            try:
                env = self.ensure_env()
                self._append_log("Reset robot before nav dual sweep")
                self._set_task_header(
                    "nav dual sweep", "running", "Reset robot")
                env.reset()
                visualize = VisualizeContext(
                    on_event=self.handle_event,
                    stop_checker=self._stop_task.is_set,
                    page_debug=bool(page_debug),
                )
                nav_dual_sweep(
                    env,
                    goal=prompt.strip(),
                    distance=float(distance),
                    nav_lift_height=float(nav_lift_height),
                    nav_debug_raw=bool(page_debug),
                    swap_debug_raw=bool(page_debug),
                    nav_depth_median_n=int(nav_depth_median_n),
                    swap_depth_median_n=int(swap_depth_median_n),
                    vote_times=int(vote_times),
                    visualize=visualize,
                )
                if self._stop_task.is_set():
                    self.handle_event(
                        "result",
                        {
                            "source": "nav_dual_sweep",
                            "status": "stopped",
                            "message": "nav dual sweep stopped",
                        },
                    )
                else:
                    self.handle_event(
                        "result",
                        {
                            "source": "nav_dual_sweep",
                            "status": "completed",
                            "message": "nav dual sweep finished",
                        },
                    )
            except Exception as exc:
                self._append_log(f"Task failed: {exc}")
                self._append_log(traceback.format_exc().strip())
                with self._lock:
                    self._state.task_state = "failed"
                    self._state.current_stage = "Task failed"
                    self._state.last_error = str(exc)
            finally:
                with self._lock:
                    if self._state.task_state == "running":
                        self._state.task_state = "completed"

        self._task_thread = threading.Thread(
            target=_runner,
            name="nav-dual-sweep-task",
            daemon=True,
        )
        self._task_thread.start()
        return "Started nav dual sweep."

    def start_smart_shelf_search(
        self,
        request: str,
        nav_debug: bool,
        pick_debug: bool,
        first_nav_height: float,
        rotate_recover: bool,
        depth_median_n: int,
    ) -> str:
        if not request.strip():
            return "Request is required."
        page_debug = bool(nav_debug or pick_debug)
        error = self._begin_task("smart shelf search", page_debug)
        if error:
            return error

        def _runner() -> None:
            try:
                env = self.ensure_env()
                self._append_log("Reset robot before smart shelf search")
                self._set_task_header(
                    "smart shelf search", "running", "Reset robot")
                env.reset()
                visualize = VisualizeContext(
                    on_event=self.handle_event,
                    stop_checker=self._stop_task.is_set,
                    page_debug=page_debug,
                )
                result = smart_shelf_search_from_request(
                    arx=env,
                    request=request.strip(),
                    first_nav_height=float(first_nav_height),
                    rotate_recover=bool(rotate_recover),
                    nav_debug=bool(nav_debug),
                    debug_pick_place=bool(pick_debug),
                    depth_median_n=int(depth_median_n),
                    visualize=visualize,
                )
                message = str(result.get("message", ""))
                status = "success" if result.get("success") else "failed"
                if self._stop_task.is_set() or message.lower().startswith("stopped"):
                    status = "stopped"
                self.handle_event(
                    "result",
                    {
                        "source": "smart_shelf_search",
                        "status": status,
                        "message": message,
                        "result": result,
                    },
                )
            except Exception as exc:
                self._append_log(f"Task failed: {exc}")
                self._append_log(traceback.format_exc().strip())
                with self._lock:
                    self._state.task_state = "failed"
                    self._state.current_stage = "Task failed"
                    self._state.last_error = str(exc)

        self._task_thread = threading.Thread(
            target=_runner,
            name="smart-shelf-search-task",
            daemon=True,
        )
        self._task_thread.start()
        return "Started smart shelf search."

    def stop_current_task(self) -> str:
        self._stop_task.set()
        self._append_log("Stop requested from page")
        with self._lock:
            if self._state.task_state in {"running", "starting"}:
                self._state.task_state = "stopping"
                self._state.current_stage = "Stopping task"
        return "Stop signal sent."

    def _summarize_status(self) -> str:
        with self._lock:
            state = self._state
            status = dict(state.status)
            active_task = state.active_task
            task_state = state.task_state
            current_stage = state.current_stage
            last_error = state.last_error
        base = status.get("base")
        base_height = None
        if base is not None and hasattr(base, "height"):
            try:
                base_height = float(base.height)
            except Exception:
                base_height = None
        lines = [
            f"**Task**: {active_task}",
            f"**State**: {task_state}",
            f"**Stage**: {current_stage}",
        ]
        if base_height is not None:
            lines.append(f"**Base Height**: {base_height:.2f}")
        if last_error:
            lines.append(f"**Last Error**: {last_error}")
        return "  \n".join(lines)

    def read_ui_state(self):
        try:
            self.ensure_env()
        except Exception as exc:
            self._append_log(f"Environment init failed: {exc}")

        with self._lock:
            state = self._state
            frames = dict(state.frames)
            nav_debug_image = state.nav_debug_image
            manip_debug_image = state.manip_debug_image
            parsed_request = state.parsed_request
            telemetry = dict(state.telemetry)
            result = state.result
            logs = list(state.logs)

        if result is not None:
            telemetry = dict(telemetry)
            telemetry["result"] = result

        return (
            self._summarize_status(),
            frames.get("camera_l_color"),
            frames.get("camera_h_color"),
            frames.get("camera_r_color"),
            nav_debug_image,
            manip_debug_image,
            _dump_json(parsed_request or {}),
            _dump_json(telemetry or {}),
            "\n".join(logs[-80:]),
        )


def _require_gradio():
    try:
        import gradio as gr
    except ImportError as exc:
        raise RuntimeError(
            "gradio is not installed. Run `pip install gradio` first."
        ) from exc
    return gr


def build_app(controller: DemoController):
    gr = _require_gradio()

    with gr.Blocks(title="ARX Demo Console") as demo:
        gr.Markdown(
            "# ARX Demo Console\n"
            "Three camera feeds stay live. When page debug is enabled, navigation "
            "and manipulation overlays are also pushed into the debug panels below."
        )

        action_feedback = gr.Textbox(
            label="Action Feedback",
            interactive=False,
            lines=1,
        )
        status_md = gr.Markdown()

        with gr.Row():
            cam_l = gr.Image(label="camera_l", interactive=False)
            cam_h = gr.Image(label="camera_h", interactive=False)
            cam_r = gr.Image(label="camera_r", interactive=False)

        with gr.Row():
            nav_debug = gr.Image(label="Navigation Debug", interactive=False)
            manip_debug = gr.Image(
                label="Manipulation Debug", interactive=False)

        with gr.Row():
            parsed_request_box = gr.Textbox(
                label="Resolved Request / Prompts",
                lines=12,
                interactive=False,
            )
            telemetry_box = gr.Textbox(
                label="Runtime Telemetry",
                lines=12,
                interactive=False,
            )

        log_box = gr.Textbox(
            label="Runtime Log",
            lines=14,
            interactive=False,
        )

        with gr.Tabs():
            with gr.Tab("Nav Dual Sweep"):
                nav_prompt = gr.Textbox(
                    label="Prompt",
                    value="paper cup or paper ball or bottle on the floor",
                )
                nav_page_debug = gr.Checkbox(
                    label="Debug Overlay In Page",
                    value=False,
                )
                with gr.Accordion("Advanced", open=False):
                    nav_distance = gr.Number(label="Stop Distance", value=0.51)
                    nav_lift_height = gr.Number(
                        label="Nav Lift Height", value=0.0)
                    nav_depth_median_n = gr.Number(
                        label="Nav Depth Median N",
                        value=5,
                        precision=0,
                    )
                    swap_depth_median_n = gr.Number(
                        label="Swap Depth Median N",
                        value=5,
                        precision=0,
                    )
                    vote_times = gr.Number(
                        label="Vote Times",
                        value=3,
                        precision=0,
                    )
                start_nav_btn = gr.Button(
                    "Start Nav Dual Sweep", variant="primary")

            with gr.Tab("Smart Shelf Search"):
                shelf_request = gr.Textbox(
                    label="Request",
                    value="我要一个蓝色盒子",
                )
                shelf_nav_debug = gr.Checkbox(
                    label="Show Navigation Debug In Page",
                    value=False,
                )
                shelf_pick_debug = gr.Checkbox(
                    label="Show Pick/Place Debug In Page",
                    value=False,
                )
                with gr.Accordion("Advanced", open=False):
                    first_nav_height = gr.Number(
                        label="First Nav Height",
                        value=14.5,
                    )
                    rotate_recover = gr.Checkbox(
                        label="Rotate Recover",
                        value=True,
                    )
                    shelf_depth_median_n = gr.Number(
                        label="Depth Median N",
                        value=10,
                        precision=0,
                    )
                start_shelf_btn = gr.Button(
                    "Start Smart Shelf Search",
                    variant="primary",
                )

        stop_btn = gr.Button("Stop Current Task", variant="stop")

        start_nav_btn.click(
            fn=controller.start_nav_dual_sweep,
            inputs=[
                nav_prompt,
                nav_page_debug,
                nav_distance,
                nav_lift_height,
                nav_depth_median_n,
                swap_depth_median_n,
                vote_times,
            ],
            outputs=[action_feedback],
            queue=False,
        )
        start_shelf_btn.click(
            fn=controller.start_smart_shelf_search,
            inputs=[
                shelf_request,
                shelf_nav_debug,
                shelf_pick_debug,
                first_nav_height,
                rotate_recover,
                shelf_depth_median_n,
            ],
            outputs=[action_feedback],
            queue=False,
        )
        stop_btn.click(
            fn=controller.stop_current_task,
            outputs=[action_feedback],
            queue=False,
        )

        timer = gr.Timer(0.3)
        timer.tick(
            fn=controller.read_ui_state,
            outputs=[
                status_md,
                cam_l,
                cam_h,
                cam_r,
                nav_debug,
                manip_debug,
                parsed_request_box,
                telemetry_box,
                log_box,
            ],
            queue=False,
        )

    return demo


def main() -> None:
    parser = argparse.ArgumentParser(description="ARX demo Gradio console")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=7860)
    args = parser.parse_args()

    controller = DemoController()
    demo = build_app(controller)
    try:
        demo.queue().launch(
            server_name=args.host,
            server_port=args.port,
            share=False,
        )
    finally:
        controller.close()


if __name__ == "__main__":
    main()
