"""Persistent external editor command settings for the Primary Matrix root menu."""

from __future__ import annotations

import json
import shlex
import subprocess
from datetime import datetime, timezone
from pathlib import Path

DEFAULT_EXTERNAL_EDITOR_COMMAND = "featherpad"
EXTERNAL_EDITOR_MENU_LABEL = "Choose Outside Editor"

_CONFIG_PATH = Path(__file__).resolve().parents[1] / "config" / "outside_editor.json"


def load_external_editor_command() -> str:
    """Return saved editor command, defaulting to featherpad."""
    try:
        payload = json.loads(_CONFIG_PATH.read_text(encoding="utf-8"))
        cmd = str(payload.get("command", "")).strip()
        if cmd:
            return cmd
    except Exception:
        pass
    return DEFAULT_EXTERNAL_EDITOR_COMMAND


def save_external_editor_command(command: str) -> str:
    """Persist editor command and return normalized value."""
    normalized = str(command or "").strip() or DEFAULT_EXTERNAL_EDITOR_COMMAND
    _CONFIG_PATH.parent.mkdir(parents=True, exist_ok=True)
    _CONFIG_PATH.write_text(
        json.dumps({"command": normalized}, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return normalized


def open_in_external_editor(file_path: str | Path, command: str | None = None) -> bool:
    """Open a file with the configured external editor command."""
    editor_cmd = str(command or load_external_editor_command()).strip()
    if not editor_cmd:
        editor_cmd = DEFAULT_EXTERNAL_EDITOR_COMMAND
    target = str(Path(file_path).expanduser().resolve())
    try:
        argv = shlex.split(editor_cmd)
        if not argv:
            argv = [DEFAULT_EXTERNAL_EDITOR_COMMAND]
        subprocess.Popen(argv + [target])
        return True
    except Exception:
        return False


def build_session_output_file(base_dir: str | Path) -> Path:
    """Build session output filename: output_terminal_YYYYMMDD_HHMMSS.txt."""
    folder = Path(base_dir).expanduser().resolve()
    folder.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    return folder / f"output_terminal_{stamp}.txt"
