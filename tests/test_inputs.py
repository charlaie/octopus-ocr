from __future__ import annotations

from pathlib import Path

from octopus_ocr.inputs import expand_input_paths


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
