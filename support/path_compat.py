"""Small, explicit path translation for the shared Coussens data mount."""

from __future__ import annotations

import os
import re


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
    return slash_path.startswith("/mnt/") or slash_path.startswith("/home/")
