"""Helpers for robust script imports across execution modes."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Union


def ensure_scripts_importable(entry_file: Union[Path, str]) -> None:
    """Ensure both project root and scripts root are on sys.path.

    This keeps direct execution (python scripts/foo.py or subdir scripts) and
    package-style execution both working without per-script path fix functions.
    """
    entry_path = Path(entry_file).resolve()

    scripts_root = None
    for parent in entry_path.parents:
        if parent.name == "scripts":
            scripts_root = parent
            break

    if scripts_root is None:
        scripts_root = entry_path.parent
    repo_root = scripts_root.parent

    new_paths = []
    for root in (repo_root, scripts_root):
        root_str = str(root)
        if root_str not in sys.path:
            new_paths.append(root_str)

    if new_paths:
        sys.path[:0] = new_paths

    # Prefer workspace-local scripts package when a foreign module is loaded.
    mod = sys.modules.get("scripts")
    if mod is not None:
        mod_file = str(getattr(mod, "__file__", ""))
        if mod_file and not mod_file.startswith(str(repo_root)):
            del sys.modules["scripts"]
