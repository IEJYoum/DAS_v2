"""Small, explicit path translation for the shared Coussens data mount."""

from __future__ import annotations

import os
import re
from pathlib import Path


POSIX_MOUNT = "/mnt/rdscoussens"
WINDOWS_SHARE = r"\\rdsdcw.ohsu.edu\Coussens-secure"
WINDOWS_SHARE_ALIASES = (
    WINDOWS_SHARE,
    r"\\rdsdcw\Coussens-secure",
)


def native_path_text(path_like: object, *, os_name: str | None = None) -> str:
    """Translate only the known shared mount between Windows and Linux forms."""
    raw = str(path_like or "").strip().strip('"').strip("'")
    if raw == "":
        return ""
    target_os = os.name if os_name is None else str(os_name)
    slash_path = raw.replace("\\", "/")

    if target_os != "nt":
        for share in WINDOWS_SHARE_ALIASES:
            share_path = share.replace("\\", "/").lower()
            if slash_path.lower() == share_path or slash_path.lower().startswith(share_path + "/"):
                return POSIX_MOUNT + slash_path[len(share_path):]
        return raw

    posix_prefix = POSIX_MOUNT.lower()
    if slash_path.lower() == posix_prefix or slash_path.lower().startswith(posix_prefix + "/"):
        suffix = slash_path[len(POSIX_MOUNT):].replace("/", "\\")
        return WINDOWS_SHARE + suffix
    return raw


def is_foreign_path_text(path_like: object, *, os_name: str | None = None) -> bool:
    """Return true for an untranslatable absolute path from another OS."""
    text = native_path_text(path_like, os_name=os_name)
    target_os = os.name if os_name is None else str(os_name)
    slash_path = text.replace("\\", "/")
    if target_os != "nt":
        return bool(re.match(r"^[A-Za-z]:/", slash_path)) or (
            slash_path.startswith("//") and not slash_path.startswith("//mnt/")
        )
    return text.startswith('/') and not text.startswith('//')


def resolve_native_path(path_like, *, base_dir=None, must_exist=False, kind=None) -> Path:
    """Resolve a native path with an explicit anchor; never invent a foreign mount."""
    text = native_path_text(path_like)
    if not text or is_foreign_path_text(text):
        raise ValueError(f'Empty or foreign path for this operating system: {path_like!r}')
    path = Path(text).expanduser()
    if not path.is_absolute():
        path = (resolve_native_path(base_dir) if base_dir is not None else Path.cwd()) / path
    path = path.resolve()
    if must_exist and not path.exists():
        raise FileNotFoundError(path)
    if path.exists() and kind == 'file' and not path.is_file():
        raise ValueError(f'Expected a file: {path}')
    if path.exists() and kind == 'directory' and not path.is_dir():
        raise ValueError(f'Expected a directory: {path}')
    return path
