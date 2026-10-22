"""Filesystem helpers for flaky network-backed project folders."""

from __future__ import annotations

import os
import json
import time
from pathlib import Path
from typing import Callable, Type
from uuid import uuid4


def atomic_write_json(path: str | Path, value) -> str:
    """Publish a JSON artifact through the shared network-aware writer."""
    return atomic_write_with_retry(path, lambda tmp: Path(tmp).write_text(
        json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8"),
        retry_seconds=1, max_minutes=1, require_nonempty=True)


def create_run_directory(parent: str | Path, *, prefix: str = "run_") -> Path:
    """Create a unique output folder with ordinary inherited permissions.

    tempfile.mkdtemp requests mode 0700. Some Windows SMB servers create that
    directory but reject its private ACL, causing tempfile to retry indefinitely.
    Only actual name collisions are retried here; access failures propagate.
    """
    parent = Path(parent)
    parent.mkdir(parents=True, exist_ok=True)
    for _ in range(5):
        folder = parent / (prefix + uuid4().hex[:12])
        try:
            folder.mkdir()
        except FileExistsError:
            continue
        return folder
    raise FileExistsError(f"Could not allocate a unique run directory in {parent}")


def retry_io(
    action: Callable[[], object],
    *,
    description: str = "filesystem operation",
    retry_seconds: int = 60,
    max_minutes: int = 90,
    exceptions: tuple[Type[BaseException], ...] = (OSError, IOError),
) -> object:
    """Run action, retrying transient filesystem failures for max_minutes."""
    attempts = max(1, int((max_minutes * 60) / max(1, retry_seconds)))
    last_exc = None
    for attempt in range(1, attempts + 1):
        try:
            return action()
        except exceptions as exc:
            last_exc = exc
            if attempt >= attempts:
                break
            print(
                description
                + " failed ("
                + str(exc)
                + "); retrying in "
                + str(retry_seconds)
                + " seconds ["
                + str(attempt)
                + "/"
                + str(attempts)
                + "]"
            )
            time.sleep(retry_seconds)
    raise last_exc


def atomic_write_with_retry(
    final_path: str | Path,
    writer: Callable[[str], object],
    *,
    description: str = "atomic file write",
    retry_seconds: int = 60,
    max_minutes: int = 90,
    require_nonempty: bool = False,
) -> str:
    """Write final_path via same-folder temp path, retrying network IO failures."""
    final = Path(final_path)

    def _write_once() -> str:
        final.parent.mkdir(parents=True, exist_ok=True)
        # A unique sibling avoids collisions when two GUI actions or processes
        # publish the same cached artifact at the same time.
        tmp = final.with_name(final.name + ".tmp." + uuid4().hex)
        try:
            writer(str(tmp))
            if require_nonempty and (not tmp.exists() or tmp.stat().st_size <= 0):
                raise OSError("temporary output is empty: " + str(tmp))
            os.replace(str(tmp), str(final))
            if require_nonempty and (not final.exists() or final.stat().st_size <= 0):
                raise OSError("final output is empty: " + str(final))
            return str(final)
        finally:
            try:
                if tmp.exists():
                    tmp.unlink()
            except OSError:
                pass

    return str(
        retry_io(
            _write_once,
            description=description + ": " + str(final),
            retry_seconds=retry_seconds,
            max_minutes=max_minutes,
        )
    )
