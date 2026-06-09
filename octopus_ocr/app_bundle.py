from __future__ import annotations

from io import BytesIO
import os
import struct
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Callable

from PIL import Image


APP_NAME = "Octopus OCR"
LOGO_SVG = Path(__file__).with_name("assets") / "octopus_ocr_icon.svg"
SvgRenderer = Callable[[Path, int], bytes]


def main(argv: list[str] | None = None) -> int:
    args = argv or []
    repo_root = Path(__file__).resolve().parents[1]
    app_path = Path(args[0]) if args else repo_root / "dist" / f"{APP_NAME}.app"
    make_dev_app(app_path, repo_root=repo_root)
    print(f"Wrote {app_path}")
    return 0


def make_dev_app(app_path: Path, *, repo_root: Path, icon_renderer: SvgRenderer | None = None) -> None:
    contents = app_path / "Contents"
    macos = contents / "MacOS"
    resources = contents / "Resources"
    macos.mkdir(parents=True, exist_ok=True)
    resources.mkdir(parents=True, exist_ok=True)

    (contents / "Info.plist").write_text(_info_plist(), encoding="utf-8")
    (resources / "OctopusOCR.svg").write_text(LOGO_SVG.read_text(encoding="utf-8"), encoding="utf-8")
    _write_app_icon(resources / "AppIcon.icns", svg_path=LOGO_SVG, renderer=icon_renderer or _render_svg_png)
    launcher = macos / "Octopus OCR"
    launcher.write_text(_launcher_script(repo_root), encoding="utf-8")
    launcher.chmod(launcher.stat().st_mode | 0o111)


def _info_plist() -> str:
    return """<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN"
  "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
  <key>CFBundleExecutable</key>
  <string>Octopus OCR</string>
  <key>CFBundleIdentifier</key>
  <string>local.octopus-ocr.gui</string>
  <key>CFBundleName</key>
  <string>Octopus OCR</string>
  <key>CFBundleDisplayName</key>
  <string>Octopus OCR</string>
  <key>CFBundleIconFile</key>
  <string>AppIcon</string>
  <key>CFBundleIconName</key>
  <string>AppIcon</string>
  <key>CFBundlePackageType</key>
  <string>APPL</string>
  <key>CFBundleShortVersionString</key>
  <string>0.1.0</string>
  <key>LSMinimumSystemVersion</key>
  <string>12.0</string>
  <key>NSHighResolutionCapable</key>
  <true/>
</dict>
</plist>
"""


def _launcher_script(repo_root: Path) -> str:
    quoted_repo_root = str(repo_root).replace('"', '\\"')
    path = os.environ.get("PATH", "/opt/homebrew/bin:/usr/local/bin:/usr/bin:/bin:/usr/sbin:/sbin")
    return f"""#!/bin/sh
export PATH="{path}"
cd "{quoted_repo_root}" || exit 1
exec uv run octopus-ocr-gui
"""


def _write_app_icon(path: Path, *, svg_path: Path, renderer: SvgRenderer) -> None:
    chunks: list[tuple[bytes, bytes]] = []
    for size, code in [
        (16, b"icp4"),
        (32, b"icp5"),
        (64, b"icp6"),
        (128, b"ic07"),
        (256, b"ic08"),
        (512, b"ic09"),
        (1024, b"ic10"),
    ]:
        chunks.append((code, renderer(svg_path, size)))

    total_length = 8 + sum(8 + len(data) for _, data in chunks)
    content = bytearray(b"icns" + struct.pack(">I", total_length))
    for code, data in chunks:
        content.extend(code)
        content.extend(struct.pack(">I", 8 + len(data)))
        content.extend(data)
    path.write_bytes(content)


def _render_svg_png(svg_path: Path, size: int) -> bytes:
    with tempfile.TemporaryDirectory() as tmpdir:
        output_dir = Path(tmpdir)
        proc = subprocess.run(
            ["qlmanage", "-t", "-s", str(size), "-o", str(output_dir), str(svg_path)],
            capture_output=True,
            text=True,
            check=False,
        )
        if proc.returncode != 0:
            message = (proc.stderr or proc.stdout).strip() or "qlmanage could not render SVG icon"
            raise RuntimeError(message)

        thumbnail_path = output_dir / f"{svg_path.name}.png"
        if not thumbnail_path.exists():
            matches = list(output_dir.glob("*.png"))
            if not matches:
                raise RuntimeError("qlmanage did not produce a PNG thumbnail for the SVG icon")
            thumbnail_path = matches[0]

        image = Image.open(thumbnail_path).convert("RGBA")
        if image.size != (size, size):
            image = image.resize((size, size), Image.Resampling.LANCZOS)
        buffer = BytesIO()
        image.save(buffer, format="PNG")
        return buffer.getvalue()


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
