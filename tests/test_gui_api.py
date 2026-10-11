from __future__ import annotations

from pathlib import Path
import json
import threading
import subprocess
import sys
from types import SimpleNamespace

import pytest

from octopus_ocr.gui import GuiApi
from octopus_ocr.gui_service import GuiRunResult


def completed_run(tmp_path: Path) -> GuiRunResult:
    return GuiRunResult(
        output_dir=tmp_path / "output",
        input_count=1,
        detected_rows=10,
        exported_transactions=9,
        warning_rows=1,
        total_seconds=2.0,
    )


@pytest.mark.parametrize("action", ["add", "remove", "clear"])
def test_source_changes_invalidate_previous_exports(tmp_path: Path, action: str) -> None:
    first = tmp_path / "first.png"
    second = tmp_path / "second.png"
    first.touch()
    second.touch()
    api = GuiApi()
    api.add_paths([str(first)])
    api._last_result = completed_run(tmp_path)

    if action == "add":
        snapshot = api.add_paths([str(second)])
        assert snapshot["count"] == 2
    elif action == "remove":
        snapshot = api.remove_input(str(first))
        assert snapshot["count"] == 0
    else:
        snapshot = api.clear_inputs()
        assert snapshot["count"] == 0

    assert api.open_output("review") == {"ok": False, "error": "No completed run is available."}


def test_cancelled_or_duplicate_selection_preserves_previous_exports(tmp_path: Path) -> None:
    source = tmp_path / "source.png"
    source.touch()
    api = GuiApi()
    api.add_paths([str(source)])
    result = completed_run(tmp_path)
    api._last_result = result

    assert api.add_paths([])["paths"] == [str(source)]
    assert api.add_paths([str(source)])["count"] == 1
    assert api.remove_input("not-in-the-queue")["count"] == 1
    assert api._last_result is result


def test_running_job_rejects_source_changes_and_clears_stale_exports(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    first = tmp_path / "first.png"
    second = tmp_path / "second.png"
    first.touch()
    second.touch()
    api = GuiApi()
    api.add_paths([str(first)])
    api._last_result = completed_run(tmp_path)
    # Keep the worker pending so all source mutation routes can be checked.
    monkeypatch.setattr("octopus_ocr.gui.threading.Thread", lambda **kwargs: SimpleNamespace(start=lambda: None))

    assert api.run({"ocr_engine": "tesseract"}) == {"accepted": True}
    for snapshot in [api.add_paths([str(second)]), api.remove_input(str(first)), api.clear_inputs()]:
        assert snapshot["paths"] == [str(first)]
        assert "error" in snapshot
    assert api.get_inputs()["paths"] == [str(first)]
    assert api.open_output("review")["ok"] is False
    assert api.run({"ocr_engine": "tesseract"})["accepted"] is False


def test_unsupported_drop_reports_error_without_discarding_existing_sources(tmp_path: Path) -> None:
    source = tmp_path / "source.png"
    unsupported = tmp_path / "notes.txt"
    source.touch()
    unsupported.touch()
    api = GuiApi()
    api.add_paths([str(source)])

    snapshot = api.add_paths([str(unsupported)])

    assert snapshot["paths"] == [str(source)]
    assert "error" in snapshot


def test_unknown_engine_is_rejected_without_starting_a_job(tmp_path: Path) -> None:
    source = tmp_path / "source.png"
    source.touch()
    api = GuiApi()
    api.add_paths([str(source)])

    assert api.run({"ocr_engine": "invalid"})["accepted"] is False
    snapshot = api.clear_inputs()
    assert snapshot["paths"] == []
    assert snapshot["count"] == 0


class RecordingWindow:
    def __init__(self) -> None:
        self.snapshots: list[dict[str, object]] = []

    def evaluate_js(self, script: str) -> object:
        snapshot = json.loads(script.removeprefix("window.OctopusOCR.onStateChanged(")[:-1])
        self.snapshots.append(snapshot)
        return None

    def create_file_dialog(self, dialog_type: int, *, allow_multiple: bool = False, file_types: tuple[str, ...] = ()) -> None:
        return None


@pytest.mark.parametrize("legacy_constants", [False, True])
def test_native_dialog_values_support_legacy_and_proxy_constants(monkeypatch: pytest.MonkeyPatch, legacy_constants: bool) -> None:
    if legacy_constants:
        monkeypatch.setitem(sys.modules, "webview", SimpleNamespace(OPEN_DIALOG=10, FOLDER_DIALOG=20))
    else:
        import webview

        assert int(webview.OPEN_DIALOG) == 10
        assert int(webview.FOLDER_DIALOG) == 20

    class DialogWindow(RecordingWindow):
        dialog_types: list[int] = []

        def create_file_dialog(self, dialog_type: int, *, allow_multiple: bool = False, file_types: tuple[str, ...] = ()) -> None:
            self.dialog_types.append(dialog_type)
            return None

    api = GuiApi()
    api.window = window = DialogWindow()
    assert api.choose_files()["phase"] == "idle"
    assert api.choose_folder()["phase"] == "idle"
    assert window.dialog_types == [10, 20]

@pytest.mark.parametrize("options", [None, [], {"ocr_engine": []}, {"paddle_model": False}, {"paddle_model": []}])
def test_malformed_bridge_options_do_not_claim_a_run(tmp_path: Path, options: object) -> None:
    source = tmp_path / "source.png"
    source.touch()
    api = GuiApi()
    api.add_paths([str(source)])

    response = api.run(options)

    assert response["accepted"] is False
    assert api.get_inputs()["phase"] == "idle"
    assert api.clear_inputs()["count"] == 0


def test_notification_failure_does_not_prevent_the_worker(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    class BrokenWindow(RecordingWindow):
        def evaluate_js(self, script: str) -> object:
            raise RuntimeError("window closed")

    source = tmp_path / "source.png"
    source.touch()
    api = GuiApi()
    api.add_paths([str(source)])
    api.window = BrokenWindow()
    monkeypatch.setattr("octopus_ocr.gui.run_gui_pipeline", lambda *args, **kwargs: completed_run(tmp_path))
    monkeypatch.setattr(
        "octopus_ocr.gui.threading.Thread",
        lambda *, target, args, daemon: SimpleNamespace(start=lambda: target(*args)),
    )

    assert api.run({})["accepted"] is True
    assert api.get_inputs()["phase"] == "complete"
    assert api._last_result == completed_run(tmp_path)


def test_thread_start_failure_releases_the_queue(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    def fail_start() -> None:
        raise RuntimeError("cannot start worker")

    source = tmp_path / "source.png"
    source.touch()
    api = GuiApi()
    api.add_paths([str(source)])
    monkeypatch.setattr("octopus_ocr.gui.threading.Thread", lambda **kwargs: SimpleNamespace(start=fail_start))

    response = api.run({})

    assert response["accepted"] is False
    assert response["error"] == "cannot start worker"
    assert api.get_inputs()["phase"] == "error"
    assert api.clear_inputs()["phase"] == "idle"


@pytest.mark.parametrize("launch_error", [False, True])
def test_output_launch_failures_are_reported(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, launch_error: bool) -> None:
    api = GuiApi()
    api._last_result = completed_run(tmp_path)

    def launch(*args, **kwargs):
        if launch_error:
            raise OSError("launcher is unavailable")
        return subprocess.CompletedProcess(args[0], returncode=1, stderr="output file was removed")

    monkeypatch.setattr("octopus_ocr.gui.subprocess.run", launch)

    response = api.open_output("review")

    assert response["ok"] is False
    assert response["error"] == ("launcher is unavailable" if launch_error else "output file was removed")


def test_concurrent_queue_updates_retain_their_snapshot_revisions(tmp_path: Path) -> None:
    entered = threading.Event()
    release = threading.Event()

    class DelayedWindow(RecordingWindow):
        def evaluate_js(self, script: str) -> object:
            snapshot = json.loads(script.removeprefix("window.OctopusOCR.onStateChanged(")[:-1])
            if snapshot["count"] == 1:
                entered.set()
                assert release.wait(5)
            return super().evaluate_js(script)

    first = tmp_path / "first.png"
    second = tmp_path / "second.png"
    first.touch()
    second.touch()
    api = GuiApi()
    window = DelayedWindow()
    api.window = window
    worker = threading.Thread(target=api.add_paths, args=([str(first)],))
    worker.start()
    try:
        assert entered.wait(5)
        latest = api.add_paths([str(second)])
    finally:
        release.set()
        worker.join(5)
    assert not worker.is_alive()
    assert [snapshot["revision"] for snapshot in window.snapshots] == [latest["revision"], 1]
    assert api.get_inputs()["paths"] == [str(first), str(second)]


def test_completion_cannot_restore_exports_after_a_new_input_snapshot(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    entered = threading.Event()
    release = threading.Event()

    class DelayedCompletion(RecordingWindow):
        def evaluate_js(self, script: str) -> object:
            snapshot = json.loads(script.removeprefix("window.OctopusOCR.onStateChanged(")[:-1])
            if snapshot["phase"] == "complete":
                entered.set()
                assert release.wait(5)
            return super().evaluate_js(script)

    source = tmp_path / "source.png"
    source.touch()
    api = GuiApi()
    api.add_paths([str(source)])
    api.window = window = DelayedCompletion()
    monkeypatch.setattr("octopus_ocr.gui.run_gui_pipeline", lambda *args, **kwargs: completed_run(tmp_path))
    worker = threading.Thread(target=api._run_background, args=([source], "tesseract", "model"))
    worker.start()
    try:
        assert entered.wait(5)
        latest = api.clear_inputs()
    finally:
        release.set()
        worker.join(5)
    assert not worker.is_alive()
    assert [snapshot["phase"] for snapshot in window.snapshots] == ["idle", "complete"]
    assert window.snapshots[0]["revision"] == latest["revision"]
    assert window.snapshots[1]["revision"] != latest["revision"]
    assert api.open_output("review")["ok"] is False
