from __future__ import annotations

from pathlib import Path

import pytest

from octopus_ocr.inputs import OUTPUT_LABEL_MAX_BYTES, expand_input_paths, prepare_image_paths, source_output_labels
from octopus_ocr.video import VideoExtractionResult


def test_expand_input_paths_accepts_supported_files_and_folders(tmp_path: Path) -> None:
    folder = tmp_path / "drop"
    folder.mkdir()
    png = folder / "B.PNG"
    jpg = folder / "a.jpg"
    video = folder / "scroll.MOV"
    ignored = folder / "notes.txt"
    nested = folder / "nested"
    nested.mkdir()
    nested_image = nested / "nested.png"
    for path in [png, jpg, video, ignored, nested_image]:
        path.write_bytes(b"x")

    paths = expand_input_paths([folder])

    assert paths == [jpg, png, video]


def test_expand_input_paths_dedupes_and_sorts_by_filename(tmp_path: Path) -> None:
    first = tmp_path / "02.png"
    second = tmp_path / "01.jpeg"
    unsupported = tmp_path / "03.gif"
    for path in [first, second, unsupported]:
        path.write_bytes(b"x")

    paths = expand_input_paths([first, second, first, unsupported])

    assert paths == [second, first]


def test_prepare_image_paths_preserves_source_order_and_video_reports(monkeypatch, tmp_path: Path) -> None:
    first_image = tmp_path / "b.png"
    last_image = tmp_path / "a.png"
    video = tmp_path / "screen recording.mp4"
    extracted_frame = tmp_path / "frames" / "screen_recording" / "frame_0001.png"
    started: list[Path] = []

    def fake_extract(source: Path, frames_dir: Path, *, sample_fps: float) -> VideoExtractionResult:
        assert source == video
        assert frames_dir == extracted_frame.parent
        assert sample_fps == 3.0
        return VideoExtractionResult([extracted_frame], ["possible skipped transactions"], 10, 8)

    monkeypatch.setattr("octopus_ocr.inputs.extract_video_keyframes", fake_extract)
    result = prepare_image_paths(
        [first_image, video, last_image], tmp_path / "frames", sample_fps=3.0, on_video_start=started.append,
    )

    assert result.image_paths == [first_image, extracted_frame, last_image]
    assert started == [video]
    assert result.videos[0].source_path == video
    assert result.videos[0].extraction.warnings == ["possible skipped transactions"]
    assert result.videos[0].seconds >= 0


@pytest.mark.parametrize(
    "names",
    [
        ("a/scroll.mp4", "b/scroll.mov"),
        ("a/scroll video.mp4", "b/scroll_video.mp4"),
        ("a/SCROLL.mp4", "b/scroll.mp4"),
    ],
)
def test_prepare_image_paths_preserves_same_stem_video_frames(monkeypatch, tmp_path: Path, names) -> None:
    videos = [tmp_path / name for name in names]

    def fake_extract(source: Path, frames_dir: Path, *, sample_fps: float) -> VideoExtractionResult:
        frames_dir.mkdir(parents=True)
        frame = frames_dir / "frame_0001.png"
        frame.write_text(str(source), encoding="utf-8")
        return VideoExtractionResult([frame], [], 1, 1)

    monkeypatch.setattr("octopus_ocr.inputs.extract_video_keyframes", fake_extract)
    result = prepare_image_paths(videos, tmp_path / "frames")

    assert len({str(path).casefold() for path in result.image_paths}) == 2
    assert [path.read_text(encoding="utf-8") for path in result.image_paths] == [str(path) for path in videos]


def test_prepare_image_paths_reserves_unique_stems_before_allocating_collision_names(monkeypatch, tmp_path: Path) -> None:
    from hashlib import sha256

    first = tmp_path / "a" / "scroll.mp4"
    second = tmp_path / "b" / "scroll.mp4"
    digest = sha256(str(first.resolve()).encode("utf-8")).hexdigest()[:12]
    unique = tmp_path / f"scroll_{digest}.mp4"

    def fake_extract(source: Path, frames_dir: Path, *, sample_fps: float) -> VideoExtractionResult:
        return VideoExtractionResult([frames_dir / "frame_0001.png"], [], 1, 1)

    monkeypatch.setattr("octopus_ocr.inputs.extract_video_keyframes", fake_extract)
    result = prepare_image_paths([first, second, unique], tmp_path / "frames")

    assert len(set(result.image_paths)) == 3
    assert result.image_paths[-1].parent.name == unique.stem


def test_source_output_labels_distinguishes_unicode_equivalent_names(tmp_path: Path) -> None:
    from unicodedata import normalize

    paths = [tmp_path / "a" / "caf\u00e9.png", tmp_path / "b" / "cafe\u0301.png"]
    labels = source_output_labels(paths)

    assert len({normalize("NFC", label).casefold() for label in labels.values()}) == 2


def test_source_output_labels_preserves_unique_original_stems(tmp_path: Path) -> None:
    paths = [tmp_path / "screen recording.png", tmp_path / "caf\u00e9.png"]

    assert source_output_labels(paths) == {path: path.stem for path in paths}


@pytest.mark.parametrize("stem", ["x" * 230, "店" * 80])
def test_source_output_labels_leaves_room_for_debug_filename_suffixes(tmp_path: Path, stem: str) -> None:
    paths = [tmp_path / "a" / f"{stem}.png", tmp_path / "b" / f"{stem}.png"]
    labels = source_output_labels(paths)

    assert len(set(labels.values())) == 2
    for label in labels.values():
        assert len(label.encode("utf-8")) <= OUTPUT_LABEL_MAX_BYTES
        output = tmp_path / f"{label}_annotated.png"
        output.write_text(label, encoding="utf-8")
        assert output.read_text(encoding="utf-8") == label


def test_source_output_labels_detects_collisions_introduced_by_truncation(tmp_path: Path) -> None:
    prefix = "x" * OUTPUT_LABEL_MAX_BYTES
    paths = [tmp_path / f"{prefix}_first.png", tmp_path / f"{prefix}_second.png"]
    labels = source_output_labels(paths)

    assert len(set(labels.values())) == 2
    assert all(len(label.encode("utf-8")) <= OUTPUT_LABEL_MAX_BYTES for label in labels.values())
