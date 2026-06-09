from __future__ import annotations

import json
import subprocess
import sys
import threading
from pathlib import Path
from typing import Any, cast

from octopus_ocr.gui_service import DEFAULT_PADDLE_MODEL, GuiRunResult, run_gui_pipeline
from octopus_ocr.inputs import expand_input_paths
from octopus_ocr.ocr import OcrEngineName, OcrUnavailableError
from octopus_ocr.pipeline import ProgressEvent


class GuiApi:
    def __init__(self) -> None:
        self.window: Any | None = None
        self._paths: list[Path] = []
        self._running = False
        self._last_result: GuiRunResult | None = None
        self._lock = threading.Lock()

    def add_paths(self, paths: list[str]) -> dict[str, object]:
        with self._lock:
            combined = [*self._paths, *(Path(path) for path in paths)]
            self._paths = expand_input_paths(combined)
            snapshot = self._inputs_snapshot()
        self._notify("onInputsChanged", snapshot)
        return snapshot

    def choose_files(self) -> dict[str, object]:
        import webview

        if self.window is None:
            return self._inputs_snapshot(error="Window is not ready.")
        selected = self.window.create_file_dialog(
            webview.OPEN_DIALOG,
            allow_multiple=True,
            file_types=("Supported files (*.png;*.jpg;*.jpeg;*.mp4;*.mov;*.m4v)",),
        )
        if selected:
            return self.add_paths(list(selected))
        return self._inputs_snapshot()

    def choose_folder(self) -> dict[str, object]:
        import webview

        if self.window is None:
            return self._inputs_snapshot(error="Window is not ready.")
        selected = self.window.create_file_dialog(webview.FOLDER_DIALOG)
        if selected:
            return self.add_paths([selected[0]])
        return self._inputs_snapshot()

    def clear_inputs(self) -> dict[str, object]:
        with self._lock:
            self._paths = []
            self._last_result = None
            snapshot = self._inputs_snapshot()
        self._notify("onInputsChanged", snapshot)
        return snapshot

    def run(self, options: dict[str, str]) -> dict[str, object]:
        with self._lock:
            if self._running:
                return {"accepted": False, "error": "A run is already in progress."}
            if not self._paths:
                return {"accepted": False, "error": "Drop at least one supported image, video, or folder."}
            paths = list(self._paths)
            self._running = True

        ocr_engine = cast(OcrEngineName, options.get("ocr_engine", "tesseract"))
        paddle_model = options.get("paddle_model") or DEFAULT_PADDLE_MODEL
        self._notify("onRunStarted", {"message": "Starting OCR"})
        thread = threading.Thread(
            target=self._run_background,
            args=(paths, ocr_engine, paddle_model),
            daemon=True,
        )
        thread.start()
        return {"accepted": True}

    def open_output(self, key: str) -> dict[str, object]:
        with self._lock:
            result = self._last_result
        if result is None:
            return {"ok": False, "error": "No completed run is available."}
        path = result.output_files.get(key)
        if path is None:
            return {"ok": False, "error": f"Unknown output: {key}"}
        subprocess.run(["open", str(path)], check=False)
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
            self._notify("onRunComplete", result.to_dict())

    def _finish_with_error(self, message: str) -> None:
        with self._lock:
            self._running = False
        self._notify("onRunError", {"message": message})

    def _on_progress(self, event: ProgressEvent) -> None:
        self._notify("onProgress", _progress_to_dict(event))

    def _inputs_snapshot(self, *, error: str | None = None) -> dict[str, object]:
        payload: dict[str, object] = {
            "paths": [str(path) for path in self._paths],
            "count": len(self._paths),
        }
        if error is not None:
            payload["error"] = error
        return payload

    def _notify(self, callback: str, payload: dict[str, object]) -> None:
        if self.window is None:
            return
        script = f"window.OctopusOCR.{callback}({json.dumps(payload)})"
        self.window.evaluate_js(script)


def main(argv: list[str] | None = None) -> int:
    _ = argv
    try:
        import webview
        from webview.dom import DOMEventHandler
    except ImportError:
        print("Error: pywebview is not installed. Run 'uv sync --dev' and try again.", file=sys.stderr)
        return 2

    api = GuiApi()
    window: Any = webview.create_window(
        "Octopus OCR",
        html=_HTML,
        js_api=api,
        width=940,
        height=700,
        min_size=(760, 560),
        text_select=True,
    )
    api.window = window

    def bind() -> None:
        def on_drag(event: dict[str, object]) -> None:
            _ = event

        def on_drop(event: dict[str, Any]) -> None:
            transfer = event.get("dataTransfer") or event.get("domTransfer") or {}
            files = transfer.get("files", [])
            paths = [file.get("pywebviewFullPath") for file in files if file.get("pywebviewFullPath")]
            if paths:
                api.add_paths(paths)

        events = cast(Any, window.dom.document.events)
        events.dragenter += DOMEventHandler(on_drag, True, True)
        events.dragover += DOMEventHandler(on_drag, True, True, debounce=250)
        events.drop += DOMEventHandler(on_drop, True, True)

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


_HTML = r"""
<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <style>
    :root {
      color-scheme: light;
      --bg: #f7f7f4;
      --panel: #ffffff;
      --ink: #202124;
      --muted: #6b7069;
      --line: #d8dad4;
      --accent: #126d63;
      --accent-strong: #0b534b;
      --warn: #a13d2d;
    }

    * {
      box-sizing: border-box;
    }

    body {
      margin: 0;
      min-height: 100vh;
      background: var(--bg);
      color: var(--ink);
      font: 14px/1.45 -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
    }

    main {
      display: grid;
      grid-template-rows: auto auto 1fr auto;
      gap: 16px;
      min-height: 100vh;
      padding: 22px;
    }

    header {
      display: flex;
      justify-content: space-between;
      gap: 18px;
      align-items: end;
    }

    h1 {
      margin: 0;
      font-size: 24px;
      font-weight: 650;
      letter-spacing: 0;
    }

    .subtitle {
      margin-top: 4px;
      color: var(--muted);
    }

    .toolbar {
      display: flex;
      gap: 8px;
      flex-wrap: wrap;
      justify-content: flex-end;
    }

    button, select, input {
      border: 1px solid var(--line);
      border-radius: 6px;
      background: #fff;
      color: var(--ink);
      min-height: 34px;
      padding: 7px 10px;
      font: inherit;
    }

    button {
      cursor: default;
      user-select: none;
    }

    button.primary {
      background: var(--accent);
      border-color: var(--accent);
      color: #fff;
      font-weight: 600;
    }

    button.primary:hover {
      background: var(--accent-strong);
    }

    button:disabled {
      opacity: 0.5;
    }

    .dropzone {
      display: grid;
      place-items: center;
      min-height: 150px;
      border: 1.5px dashed #aeb4aa;
      border-radius: 8px;
      background: #fbfbf8;
      text-align: center;
      padding: 24px;
    }

    .dropzone strong {
      display: block;
      font-size: 18px;
      margin-bottom: 5px;
    }

    .dropzone span {
      color: var(--muted);
    }

    .settings {
      display: grid;
      grid-template-columns: 190px minmax(220px, 1fr) auto;
      gap: 10px;
      align-items: end;
    }

    label {
      display: grid;
      gap: 5px;
      color: var(--muted);
      font-size: 12px;
      font-weight: 600;
    }

    label > span {
      color: var(--muted);
    }

    .content {
      display: grid;
      grid-template-columns: minmax(260px, 1fr) minmax(300px, 0.9fr);
      gap: 16px;
      min-height: 0;
    }

    section {
      min-height: 0;
    }

    h2 {
      margin: 0 0 8px;
      font-size: 14px;
      font-weight: 650;
      letter-spacing: 0;
    }

    .list {
      height: 100%;
      min-height: 220px;
      max-height: 315px;
      overflow: auto;
      border: 1px solid var(--line);
      border-radius: 8px;
      background: var(--panel);
    }

    .empty {
      padding: 20px;
      color: var(--muted);
    }

    .path {
      padding: 9px 11px;
      border-bottom: 1px solid #eceee8;
      white-space: nowrap;
      overflow: hidden;
      text-overflow: ellipsis;
      font-family: ui-monospace, SFMono-Regular, Menlo, monospace;
      font-size: 12px;
    }

    .path:last-child {
      border-bottom: 0;
    }

    .progress-panel {
      border: 1px solid var(--line);
      border-radius: 8px;
      background: var(--panel);
      padding: 14px;
    }

    .progress-track {
      height: 12px;
      border-radius: 999px;
      background: #e4e7df;
      overflow: hidden;
      margin: 10px 0 8px;
    }

    .progress-fill {
      width: 0%;
      height: 100%;
      background: var(--accent);
      transition: width 180ms ease;
    }

    .status {
      min-height: 22px;
      color: var(--muted);
    }

    .elapsed {
      margin-top: 2px;
      min-height: 20px;
      color: var(--muted);
      font-variant-numeric: tabular-nums;
    }

    .error {
      margin-top: 10px;
      color: var(--warn);
      min-height: 20px;
    }

    .results {
      display: none;
      margin-top: 14px;
      padding-top: 12px;
      border-top: 1px solid var(--line);
    }

    .results.visible {
      display: block;
    }

    .summary {
      display: grid;
      grid-template-columns: repeat(3, 1fr);
      gap: 8px;
      margin-bottom: 12px;
    }

    .metric {
      border: 1px solid #eceee8;
      border-radius: 6px;
      padding: 8px;
      background: #fbfbf8;
    }

    .metric b {
      display: block;
      font-size: 18px;
    }

    .metric span {
      color: var(--muted);
      font-size: 12px;
    }

    .output-buttons {
      display: flex;
      flex-wrap: wrap;
      gap: 8px;
    }

    @media (max-width: 760px) {
      main {
        padding: 16px;
      }

      header, .settings, .content {
        grid-template-columns: 1fr;
      }

      header {
        align-items: start;
      }

      .toolbar {
        justify-content: flex-start;
      }
    }
  </style>
</head>
<body>
  <main>
    <header>
      <div>
        <h1>Octopus OCR</h1>
        <div class="subtitle">Drop screenshots or folders, then export review CSV and OFX files.</div>
      </div>
      <div class="toolbar">
        <button id="pick-files">Choose Files</button>
        <button id="pick-folder">Choose Folder</button>
        <button id="clear">Clear</button>
      </div>
    </header>

    <div class="dropzone" id="dropzone">
      <div>
        <strong>Drop screenshots here</strong>
        <span>PNG, JPG, JPEG, MOV, MP4, M4V; folders are expanded one level.</span>
      </div>
    </div>

    <div class="settings">
      <label>
        <span>OCR engine</span>
        <select id="ocr-engine">
          <option value="tesseract">Tesseract</option>
          <option value="paddle">Paddle</option>
        </select>
      </label>
      <label>
        <span>Paddle model</span>
        <input id="paddle-model" value="en_PP-OCRv5_mobile_rec">
      </label>
      <button class="primary" id="run" disabled>Run OCR</button>
    </div>

    <div class="content">
      <section>
        <h2 id="input-title">Selected inputs</h2>
        <div class="list" id="input-list">
          <div class="empty">No inputs selected.</div>
        </div>
      </section>

      <section>
        <h2>Run status</h2>
        <div class="progress-panel">
          <div class="progress-track"><div class="progress-fill" id="progress-fill"></div></div>
          <div class="status" id="status">Ready.</div>
          <div class="elapsed" id="elapsed">Elapsed: 0.0s</div>
          <div class="error" id="error"></div>
          <div class="results" id="results">
            <div class="summary">
              <div class="metric"><b id="rows">0</b><span>Rows</span></div>
              <div class="metric"><b id="transactions">0</b><span>Transactions</span></div>
              <div class="metric"><b id="warnings">0</b><span>Warnings</span></div>
            </div>
            <div class="output-buttons">
              <button data-output="review">Review CSV</button>
              <button data-output="monthly_totals">Totals CSV</button>
              <button data-output="ofx">OFX</button>
              <button data-output="transactions">JSON</button>
              <button data-output="folder">Output Folder</button>
            </div>
          </div>
        </div>
      </section>
    </div>
  </main>

  <script>
    const state = { running: false, count: 0, startedAt: null, elapsedTimer: null };
    const list = document.getElementById('input-list');
    const title = document.getElementById('input-title');
    const runButton = document.getElementById('run');
    const fill = document.getElementById('progress-fill');
    const statusEl = document.getElementById('status');
    const elapsedEl = document.getElementById('elapsed');
    const errorEl = document.getElementById('error');
    const resultsEl = document.getElementById('results');

    function setRunning(running) {
      state.running = running;
      runButton.disabled = running || state.count === 0;
      document.getElementById('pick-files').disabled = running;
      document.getElementById('pick-folder').disabled = running;
      document.getElementById('clear').disabled = running;
    }

    function renderInputs(payload) {
      state.count = payload.count || 0;
      title.textContent = `Selected inputs (${state.count})`;
      runButton.disabled = state.running || state.count === 0;
      if (payload.error) errorEl.textContent = payload.error;
      list.innerHTML = '';
      if (!payload.paths || payload.paths.length === 0) {
        list.innerHTML = '<div class="empty">No inputs selected.</div>';
        return;
      }
      for (const path of payload.paths) {
        const item = document.createElement('div');
        item.className = 'path';
        item.title = path;
        item.textContent = path;
        list.appendChild(item);
      }
    }

    function progressPercent(event) {
      if (event.phase === 'complete') return 100;
      if (event.phase === 'export') return 96;
      if (event.phase === 'ocr' && event.total_rows > 0) {
        return 20 + Math.round((event.completed_rows / event.total_rows) * 76);
      }
      if (event.phase === 'detect' && event.image_count > 0) {
        return Math.min(20, Math.round((event.image_index / event.image_count) * 20));
      }
      return 4;
    }

    function formatElapsed(seconds) {
      if (seconds < 60) return `${seconds.toFixed(1)}s`;
      const minutes = Math.floor(seconds / 60);
      const remaining = seconds - minutes * 60;
      return `${minutes}m ${remaining.toFixed(1).padStart(4, '0')}s`;
    }

    function setElapsed(seconds) {
      elapsedEl.textContent = `Elapsed: ${formatElapsed(seconds)}`;
    }

    function updateElapsed() {
      if (!state.startedAt) return;
      setElapsed((Date.now() - state.startedAt) / 1000);
    }

    function stopElapsedTimer(finalSeconds = null) {
      if (state.elapsedTimer) {
        window.clearInterval(state.elapsedTimer);
        state.elapsedTimer = null;
      }
      if (finalSeconds !== null) setElapsed(finalSeconds);
      state.startedAt = null;
    }

    function startElapsedTimer() {
      stopElapsedTimer();
      state.startedAt = Date.now();
      setElapsed(0);
      state.elapsedTimer = window.setInterval(updateElapsed, 250);
    }

    window.OctopusOCR = {
      onInputsChanged(payload) {
        renderInputs(payload);
      },
      onRunStarted(payload) {
        setRunning(true);
        errorEl.textContent = '';
        resultsEl.classList.remove('visible');
        fill.style.width = '2%';
        statusEl.textContent = payload.message || 'Starting OCR';
        startElapsedTimer();
      },
      onProgress(event) {
        fill.style.width = `${progressPercent(event)}%`;
        statusEl.textContent = event.message || 'Working';
      },
      onRunComplete(result) {
        setRunning(false);
        stopElapsedTimer(result.total_seconds);
        fill.style.width = '100%';
        statusEl.textContent = `Complete in ${result.total_seconds.toFixed(2)}s. Output: ${result.output_dir}`;
        document.getElementById('rows').textContent = result.detected_rows;
        document.getElementById('transactions').textContent = result.exported_transactions;
        document.getElementById('warnings').textContent = result.warning_rows;
        resultsEl.classList.add('visible');
      },
      onRunError(payload) {
        setRunning(false);
        stopElapsedTimer();
        statusEl.textContent = 'Run failed.';
        errorEl.textContent = payload.message || 'Unknown error.';
      }
    };

    document.getElementById('pick-files').addEventListener('click', () => {
      window.pywebview.api.choose_files().then(renderInputs);
    });
    document.getElementById('pick-folder').addEventListener('click', () => {
      window.pywebview.api.choose_folder().then(renderInputs);
    });
    document.getElementById('clear').addEventListener('click', () => {
      window.pywebview.api.clear_inputs().then(renderInputs);
      fill.style.width = '0%';
      statusEl.textContent = 'Ready.';
      stopElapsedTimer(0);
      errorEl.textContent = '';
      resultsEl.classList.remove('visible');
    });
    runButton.addEventListener('click', () => {
      window.pywebview.api.run({
        ocr_engine: document.getElementById('ocr-engine').value,
        paddle_model: document.getElementById('paddle-model').value
      }).then((response) => {
        if (!response.accepted) errorEl.textContent = response.error || 'Could not start run.';
      });
    });
    document.querySelectorAll('[data-output]').forEach((button) => {
      button.addEventListener('click', () => {
        window.pywebview.api.open_output(button.dataset.output).then((response) => {
          if (!response.ok) errorEl.textContent = response.error || 'Could not open output.';
        });
      });
    });
  </script>
</body>
</html>
"""


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
