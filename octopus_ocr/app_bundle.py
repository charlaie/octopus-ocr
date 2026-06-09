from __future__ import annotations

import os
import sys
from pathlib import Path


APP_NAME = "Octopus OCR"


def main(argv: list[str] | None = None) -> int:
    args = argv or []
    repo_root = Path(__file__).resolve().parents[1]
    app_path = Path(args[0]) if args else repo_root / "dist" / f"{APP_NAME}.app"
    make_dev_app(app_path, repo_root=repo_root)
    print(f"Wrote {app_path}")
    return 0


def make_dev_app(app_path: Path, *, repo_root: Path) -> None:
    contents = app_path / "Contents"
    macos = contents / "MacOS"
    resources = contents / "Resources"
    macos.mkdir(parents=True, exist_ok=True)
    resources.mkdir(parents=True, exist_ok=True)

    (contents / "Info.plist").write_text(_info_plist(), encoding="utf-8")
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
  <key>CFBundlePackageType</key>
  <string>APPL</string>
  <key>CFBundleShortVersionString</key>
  <string>0.1.0</string>
  <key>LSMinimumSystemVersion</key>
  <string>12.0</string>
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


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
