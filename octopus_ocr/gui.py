from __future__ import annotations

import base64
import json
import logging
import subprocess
import sys
import threading
from pathlib import Path
from typing import Literal, Protocol, Sequence

from octopus_ocr.gui_service import DEFAULT_PADDLE_MODEL, GuiRunResult, run_gui_pipeline
from octopus_ocr.inputs import expand_input_paths
from octopus_ocr.ocr import OcrEngineName, OcrUnavailableError
from octopus_ocr.pipeline import ProgressEvent


class GuiWindow(Protocol):
    def evaluate_js(self, script: str) -> object: ...

    def create_file_dialog(
        self, dialog_type: int, *, allow_multiple: bool = False, file_types: tuple[str, ...] = (),
    ) -> Sequence[str] | None: ...


class GuiApi:
    def __init__(self) -> None:
        self.window: GuiWindow | None = None
        self._paths: list[Path] = []
        self._running = False
        self._last_result: GuiRunResult | None = None
        self._run_error: str | None = None
        self._progress: ProgressEvent | None = None
        self._revision = 0
        self._lock = threading.Lock()

    def add_paths(self, paths: object) -> dict[str, object]:
        with self._lock:
            if self._running:
                return self._inputs_snapshot(error="Wait for the current run to finish before changing files.")
            if not isinstance(paths, list) or not all(isinstance(path, str) for path in paths):
                return self._inputs_snapshot(error="Choose supported file or folder paths.")
            combined = [*self._paths, *(Path(path) for path in paths)]
            expanded = expand_input_paths(combined)
            added = len(expanded) - len(self._paths)
            if expanded != self._paths:
                self._reset_result()
            self._paths = expanded
            snapshot = self._inputs_snapshot(
                error="No new supported files found. Choose PNG, JPG, JPEG, MOV, MP4, or M4V files."
                if paths and added == 0 else None,
            )
        self._notify(snapshot)
        return snapshot

    def get_inputs(self) -> dict[str, object]:
        with self._lock:
            return self._inputs_snapshot()

    def remove_input(self, path: object) -> dict[str, object]:
        with self._lock:
            if self._running:
                return self._inputs_snapshot(error="Wait for the current run to finish before changing files.")
            if not isinstance(path, str):
                return self._inputs_snapshot(error="Choose a file from the queue to remove.")
            remaining = [item for item in self._paths if str(item) != path]
            if remaining != self._paths:
                self._paths = remaining
                self._reset_result()
            snapshot = self._inputs_snapshot()
        self._notify(snapshot)
        return snapshot

    def choose_files(self) -> dict[str, object]:
        import webview

        if self.window is None:
            with self._lock:
                return self._inputs_snapshot(error="Window is not ready.")
        selected = self.window.create_file_dialog(
            int(webview.OPEN_DIALOG),
            allow_multiple=True,
            file_types=("Supported files (*.png;*.jpg;*.jpeg;*.mp4;*.mov;*.m4v)",),
        )
        if selected:
            return self.add_paths(list(selected))
        return self.get_inputs()

    def choose_folder(self) -> dict[str, object]:
        import webview

        if self.window is None:
            with self._lock:
                return self._inputs_snapshot(error="Window is not ready.")
        selected = self.window.create_file_dialog(int(webview.FOLDER_DIALOG))
        if selected:
            return self.add_paths([selected[0]])
        return self.get_inputs()

    def clear_inputs(self) -> dict[str, object]:
        with self._lock:
            if self._running:
                return self._inputs_snapshot(error="Wait for the current run to finish before changing files.")
            self._paths = []
            self._reset_result()
            snapshot = self._inputs_snapshot()
        self._notify(snapshot)
        return snapshot

    def run(self, options: object) -> dict[str, object]:
        if not isinstance(options, dict):
            return self._reject_run("Choose recognition settings before starting a run.")
        engine = options.get("ocr_engine", "tesseract")
        if not isinstance(engine, str) or engine not in {"tesseract", "paddle"}:
            return self._reject_run("Choose Tesseract or Paddle for recognition.")
        paddle_model = options.get("paddle_model", DEFAULT_PADDLE_MODEL)
        if paddle_model is None or paddle_model == "":
            paddle_model = DEFAULT_PADDLE_MODEL
        if not isinstance(paddle_model, str):
            return self._reject_run("Choose a Paddle model name.")
        ocr_engine: OcrEngineName = "paddle" if engine == "paddle" else "tesseract"
        with self._lock:
            if self._running:
                return {"accepted": False, "error": "A run is already in progress.", "snapshot": self._inputs_snapshot()}
            if not self._paths:
                return {"accepted": False, "error": "Drop at least one supported image, video, or folder.", "snapshot": self._inputs_snapshot()}
            paths = list(self._paths)
            self._running = True
            self._last_result = None
            self._run_error = None
            self._progress = None
            self._revision += 1
            snapshot = self._inputs_snapshot()

        self._notify(snapshot)
        try:
            thread = threading.Thread(
                target=self._run_background,
                args=(paths, ocr_engine, paddle_model),
                daemon=True,
            )
            thread.start()
        except Exception as exc:
            message = str(exc) or exc.__class__.__name__
            self._finish_with_error(message)
            return self._reject_run(message)
        return {"accepted": True}

    def _reject_run(self, message: str) -> dict[str, object]:
        return {"accepted": False, "error": message, "snapshot": self.get_inputs()}

    def open_output(self, key: object) -> dict[str, object]:
        with self._lock:
            result = self._last_result
        if result is None:
            return {"ok": False, "error": "No completed run is available."}
        path = result.output_files.get(key) if isinstance(key, str) else None
        if path is None:
            return {"ok": False, "error": f"Unknown output: {key}"}
        try:
            process = subprocess.run(["open", str(path)], capture_output=True, text=True, check=False)
        except OSError as exc:
            return {"ok": False, "error": str(exc)}
        if process.returncode != 0:
            return {"ok": False, "error": process.stderr.strip() or f"Could not open {path}"}
        return {"ok": True}

    def _run_background(self, paths: list[Path], ocr_engine: OcrEngineName, paddle_model: str) -> None:
        try:
            result = run_gui_pipeline(
                paths,
                ocr_engine=ocr_engine,
                paddle_model=paddle_model,
                progress=self._on_progress,
            )
        except OcrUnavailableError as exc:
            self._finish_with_error(str(exc))
        except Exception as exc:  # noqa: BLE001 - surface run failures in the GUI.
            self._finish_with_error(str(exc) or exc.__class__.__name__)
        else:
            with self._lock:
                self._last_result = result
                self._running = False
                self._revision += 1
                snapshot = self._inputs_snapshot()
            self._notify(snapshot)

    def _finish_with_error(self, message: str) -> None:
        with self._lock:
            self._running = False
            self._run_error = message
            self._revision += 1
            snapshot = self._inputs_snapshot()
        self._notify(snapshot)

    def _on_progress(self, event: ProgressEvent) -> None:
        with self._lock:
            self._progress = event
            self._revision += 1
            snapshot = self._inputs_snapshot()
        self._notify(snapshot)

    def _reset_result(self) -> None:
        self._last_result = None
        self._run_error = None
        self._progress = None
        self._revision += 1

    def _inputs_snapshot(self, *, error: str | None = None) -> dict[str, object]:
        payload: dict[str, object] = {
            "revision": self._revision,
            "paths": [str(path) for path in self._paths],
            "count": len(self._paths),
            "phase": self._phase(),
            "progress": _progress_to_dict(self._progress) if self._progress is not None else None,
            "result": self._last_result.to_dict() if self._last_result is not None else None,
            "run_error": self._run_error,
        }
        if error is not None:
            payload["error"] = error
        return payload

    def _phase(self) -> Literal["idle", "running", "complete", "error"]:
        if self._running:
            return "running"
        if self._last_result is not None:
            return "complete"
        return "error" if self._run_error is not None else "idle"

    def _notify(self, payload: dict[str, object]) -> None:
        if self.window is None:
            return
        script = f"window.OctopusOCR.onStateChanged({json.dumps(payload)})"
        try:
            self.window.evaluate_js(script)
        except Exception:
            # UI delivery must not abort OCR or leave a claimed run without a worker.
            logging.getLogger(__name__).warning("Could not update the desktop window", exc_info=True)


def main(argv: list[str] | None = None) -> int:
    _ = argv
    try:
        import webview
        from webview.dom import DOMEventHandler
    except ImportError:
        print("Error: pywebview is not installed. Run 'uv sync --dev' and try again.", file=sys.stderr)
        return 2

    api = GuiApi()
    window = webview.create_window(
        "Octopus OCR",
        html=_load_html(),
        js_api=api,
        width=1080,
        height=780,
        min_size=(780, 620),
        background_color="#f4f7f6",
        text_select=True,
    )
    if window is None:
        print("Error: could not create the desktop window.", file=sys.stderr)
        return 2
    api.window = window

    def bind() -> None:
        def on_drag(event: dict[str, object]) -> None:
            _ = event

        def on_drop(event: dict[str, object]) -> None:
            transfer = event.get("dataTransfer") or event.get("domTransfer") or {}
            if not isinstance(transfer, dict):
                return
            files = transfer.get("files", [])
            if not isinstance(files, list):
                return
            paths: list[str] = []
            for file in files:
                if isinstance(file, dict):
                    path = file.get("pywebviewFullPath")
                    if isinstance(path, str) and path:
                        paths.append(path)
            if paths:
                api.add_paths(paths)

        document = window.dom.document
        document.on("dragenter", DOMEventHandler(on_drag, True, True))
        document.on("dragover", DOMEventHandler(on_drag, True, True, debounce=250))
        document.on("drop", DOMEventHandler(on_drop, True, True))

    webview.start(bind)
    return 0


def _progress_to_dict(event: ProgressEvent) -> dict[str, object]:
    return {
        "phase": event.phase,
        "message": event.message,
        "image_index": event.image_index,
        "image_count": event.image_count,
        "row_index": event.row_index,
        "rows_in_image": event.rows_in_image,
        "completed_rows": event.completed_rows,
        "total_rows": event.total_rows,
        "output_path": str(event.output_path) if event.output_path is not None else None,
    }


def _load_html() -> str:
    assets = Path(__file__).with_name("assets")
    ui = assets / "ui"
    icon = base64.b64encode((assets / "octopus_ocr_icon.svg").read_bytes()).decode("ascii")
    return (
        (ui / "index.html").read_text(encoding="utf-8")
        .replace("/* OCTOPUS_STYLES */", (ui / "app.css").read_text(encoding="utf-8"))
        .replace("/* OCTOPUS_SCRIPT */", (ui / "app.js").read_text(encoding="utf-8"))
        .replace("{{APP_ICON}}", f"data:image/svg+xml;base64,{icon}")
    )


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
