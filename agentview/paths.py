# Copyright 2026 Edvard Pohjavirta
# Licensed under the Apache License, Version 2.0.
# See the LICENSE file in the project root for the full text.

"""Mapping between a project cwd and its Claude Code transcript directory."""
from __future__ import annotations

import os
from pathlib import Path

_REPLACE = {"/": "-", ".": "-", " ": "-"}


def encode_cwd(path: str) -> str:
    """Encode a cwd the way Claude Code names its projects directory."""
    p = str(path).rstrip("/") or "/"
    return "".join(_REPLACE.get(ch, ch) for ch in p)


def projects_root() -> Path:
    return Path(os.path.expanduser("~/.claude/projects"))


def session_dir(cwd: str, root: Path | None = None) -> Path:
    return (root or projects_root()) / encode_cwd(cwd)
