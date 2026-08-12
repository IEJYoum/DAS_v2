"""Filesystem helpers for flaky network-backed project folders."""

from __future__ import annotations

import os
import time
from pathlib import Path
from typing import Callable, Type


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
        tmp = final.with_name(final.name + ".tmp")
        try:
            if tmp.exists():
                tmp.unlink()
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
