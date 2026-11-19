"""Tiny clipboard helper using the platform's own command-line tools.

Only text the user asked to copy or paste passes through here; nothing is read
from the clipboard unless the user runs a command with ``--paste``.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import sys


def _copy_commands() -> list[list[str]]:
    if sys.platform == "darwin":
        return [["pbcopy"]]
    if sys.platform.startswith("win"):
        return [["clip"]]
    return [["wl-copy"], ["xclip", "-selection", "clipboard"], ["xsel", "--clipboard", "--input"]]


def _paste_commands() -> list[list[str]]:
    if sys.platform == "darwin":
        return [["pbpaste"]]
    if sys.platform.startswith("win"):
        return [["powershell", "-NoProfile", "-Command", "Get-Clipboard -Raw"]]
    return [
        ["wl-paste", "--no-newline"],
        ["xclip", "-selection", "clipboard", "-o"],
        ["xsel", "--clipboard", "--output"],
    ]


def _env() -> dict[str, str]:
    env = dict(os.environ)
    env.setdefault("LANG", "en_US.UTF-8")
    return env


def copy_text(text: str) -> bool:
    """Copy ``text`` to the clipboard. Returns False if no clipboard tool works."""
    for cmd in _copy_commands():
        if shutil.which(cmd[0]) is None:
            continue
        encoding = "utf-16-le" if cmd[0] == "clip" else "utf-8"
        try:
            subprocess.run(cmd, input=text.encode(encoding), check=True, timeout=5, env=_env())
            return True
        except (OSError, subprocess.SubprocessError):
            continue
    return False


def paste_text() -> str | None:
    """Return the clipboard text, or None if no clipboard tool works."""
    for cmd in _paste_commands():
        if shutil.which(cmd[0]) is None:
            continue
        try:
            done = subprocess.run(cmd, capture_output=True, check=True, timeout=5, env=_env())
            return done.stdout.decode("utf-8", errors="replace")
        except (OSError, subprocess.SubprocessError):
            continue
    return None
