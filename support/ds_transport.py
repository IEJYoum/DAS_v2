"""
Tiny file-backed DS transport helpers.

This module intentionally stays small:
- create one DS session folder
- append JSONL event rows
- poll JSONL replies
"""

from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional


@dataclass
class DsSession:
    session_root: Path
    session_id: str
    session_dir: Path
    events_path: Path
    replies_path: Path
    session_path: Path
    started_at: str
    poll_interval_sec: float = 0.25
    next_event_index: int = 0
    next_prompt_index: int = 0
    consumed_reply_keys: set[tuple[str, str]] = field(default_factory=set)


def create_session(
    session_root: str | Path,
    *,
    session_id: Optional[str] = None,
    poll_interval_sec: float = 0.25,
    session_meta: Optional[dict[str, Any]] = None,
) -> DsSession:
    root = Path(session_root).expanduser().resolve()
    root.mkdir(parents=True, exist_ok=True)

    base_id = str(session_id).strip() if session_id is not None else _default_session_id()
    session_dir, resolved_id = _allocate_session_dir(root, base_id)
    events_path = session_dir / "ds_events.jsonl"
    replies_path = session_dir / "ds_replies.jsonl"
    session_path = session_dir / "ds_session.json"
    events_path.touch(exist_ok=True)
    replies_path.touch(exist_ok=True)

    session = DsSession(
        session_root=root,
        session_id=resolved_id,
        session_dir=session_dir,
        events_path=events_path,
        replies_path=replies_path,
        session_path=session_path,
        started_at=_utc_now_text(),
        poll_interval_sec=float(poll_interval_sec),
    )
    update_session_status(session, "active", extra=session_meta)
    write_event(
        session,
        "session",
        text="DS session started",
        event="started",
        session_id=session.session_id,
        started_at=session.started_at,
    )
    return session


def update_session_status(
    session: DsSession,
    status: str,
    *,
    extra: Optional[dict[str, Any]] = None,
) -> None:
    payload: dict[str, Any] = {
        "session_id": session.session_id,
        "started_at": session.started_at,
        "status": str(status),
    }
    if extra:
        for key, value in extra.items():
            normalized = _normalize_json_value(value)
            if normalized is not None:
                payload[str(key)] = normalized
    session.session_path.write_text(
        json.dumps(payload, ensure_ascii=True, indent=2) + "\n",
        encoding="utf-8",
    )


def next_prompt_id(session: DsSession) -> str:
    prompt_id = f"{session.session_id}-p{session.next_prompt_index}"
    session.next_prompt_index += 1
    return prompt_id


def write_event(
    session: DsSession,
    kind: str,
    *,
    text: Optional[str] = None,
    **fields: Any,
) -> dict[str, Any]:
    row: dict[str, Any] = {
        "index": session.next_event_index,
        "kind": str(kind),
        "mode": "ds",
    }
    if text is not None:
        row["text"] = str(text)
    for key, value in fields.items():
        normalized = _normalize_json_value(value)
        if normalized is not None:
            row[str(key)] = normalized
    with session.events_path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, ensure_ascii=True) + "\n")
    session.next_event_index += 1
    return row


def wait_for_reply(session: DsSession, prompt_id: str) -> str:
    wanted = str(prompt_id)
    while True:
        row = _find_reply_row(session, wanted)
        if row is not None:
            return str(row.get("text", ""))
        time.sleep(session.poll_interval_sec)


def _find_reply_row(session: DsSession, prompt_id: str) -> Optional[dict[str, Any]]:
    if not session.replies_path.exists():
        return None
    with session.replies_path.open("r", encoding="utf-8") as handle:
        for line_no, raw_line in enumerate(handle):
            line = raw_line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except Exception:
                continue
            if not isinstance(row, dict):
                continue
            key = _reply_key(row, line_no)
            if key in session.consumed_reply_keys:
                continue
            if str(row.get("prompt_id", "")) != prompt_id:
                continue
            session.consumed_reply_keys.add(key)
            return row
    return None


def _reply_key(row: dict[str, Any], line_no: int) -> tuple[str, str]:
    if "index" in row:
        return ("index", str(row.get("index")))
    return ("line", str(line_no))


def _allocate_session_dir(root: Path, base_id: str) -> tuple[Path, str]:
    stem = base_id if base_id else _default_session_id()
    suffix = 0
    while True:
        name = stem if suffix == 0 else f"{stem}_{suffix:02d}"
        candidate = root / name
        try:
            candidate.mkdir(parents=True, exist_ok=False)
            return candidate, name
        except FileExistsError:
            suffix += 1


def _default_session_id() -> str:
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    return f"{stamp}_p{os.getpid()}"


def _utc_now_text() -> str:
    return datetime.now(timezone.utc).isoformat()


def _normalize_json_value(value: Any) -> Any:
    if value is None:
        return None
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (str, bool, int, float)):
        return value
    if isinstance(value, dict):
        out: dict[str, Any] = {}
        for key, item in value.items():
            normalized = _normalize_json_value(item)
            if normalized is not None:
                out[str(key)] = normalized
        return out
    if isinstance(value, (list, tuple, set)):
        return [_normalize_json_value(item) for item in value]
    return str(value)
