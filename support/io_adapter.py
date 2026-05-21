"""
Pseudocode adapter for terminal/GUI input-output routing.

Goal:
- Keep legacy terminal behavior by default.
- Allow top-level callers to switch to GUI mode with minimal code edits.
- Maintain a lightweight session log separate from legacy LOG/VLOG lists.
"""

from __future__ import annotations

import importlib.util
import re
import sys
from typing import Any, Optional

import ds_transport

# Preserve original builtins so adapter input/print remain stable even when
# legacy contexts monkey-patch builtins.input/print.
_BASE_INPUT = input
_BASE_PRINT = print


# Runtime mode: "terminal" (default) or "gui".
_MODE = "terminal"

# GUI integration handles (set by init_gui).
_SCREEN = None
_FE = None  # frontend module (expects ginput/logprint)
_DS_SESSION = None
_DS_STAGE = None
_DS_PROGRESS_LAST = None

# Simple in-memory IO log.
_LOG: list[str] = []
_SUPPORT_CAPTURE: list[dict[str, Any]] = []
_PROMPT_STICKY_OPTIONS: dict[str, list[dict[str, str]]] = {}
_MAX_PROMPT_STICKY = 48
PROGRESS_TICK = 0
PROGRESS_MAX = 0
PROGRESS_PHASE = ""

if __spec__ is None and "__file__" in globals():
    __spec__ = importlib.util.spec_from_file_location(__name__, __file__)
    if __spec__ is not None:
        __loader__ = __spec__.loader


class UserAbortError(Exception):
    """Raised when a GUI session is closed and user declines terminal resume."""
    


def init_terminal() -> None:
    """
    PSEUDOCODE:
    - Reset adapter to terminal mode.
    - Keep existing _LOG unless caller explicitly clears it.
    """
    global _MODE, _SCREEN, _FE, _DS_SESSION, _DS_STAGE, _DS_PROGRESS_LAST, PROGRESS_TICK, PROGRESS_MAX, PROGRESS_PHASE
    _deactivate_ds_session("switched_to_terminal")
    _MODE = "terminal"
    _SCREEN = None
    _FE = None
    _DS_SESSION = None
    _DS_STAGE = None
    _DS_PROGRESS_LAST = None
    _PROMPT_STICKY_OPTIONS.clear()
    PROGRESS_TICK = 0
    PROGRESS_MAX = 0
    PROGRESS_PHASE = ""


def init_gui(screen: Any, frontend_module: Any) -> None:
    """
    PSEUDOCODE:
    - Switch adapter into GUI mode.
    - Store screen + frontend module for ginput/logprint calls.
    - frontend_module contract:
      - ginput(screen, log, prompt) -> tuple(value, updated_log) or [value, updated_log]
      - logprint(log, *parts) -> updated_log
    """
    global _MODE, _SCREEN, _FE, _DS_SESSION, _DS_STAGE, _DS_PROGRESS_LAST, PROGRESS_TICK, PROGRESS_MAX, PROGRESS_PHASE
    _deactivate_ds_session("switched_to_gui")
    _MODE = "gui"
    _SCREEN = screen
    _FE = frontend_module
    _DS_SESSION = None
    _DS_STAGE = None
    _DS_PROGRESS_LAST = None
    _PROMPT_STICKY_OPTIONS.clear()
    PROGRESS_TICK = 0
    PROGRESS_MAX = 0
    PROGRESS_PHASE = ""


def init_ds(
    session_root: str,
    *,
    session_id: Optional[str] = None,
    poll_interval_sec: float = 0.25,
    session_meta: Optional[dict[str, Any]] = None,
) -> None:
    """
    Switch adapter into DS mode backed by append-only session files.
    """
    global _MODE, _SCREEN, _FE, _DS_SESSION, _DS_STAGE, _DS_PROGRESS_LAST
    _deactivate_ds_session("replaced")
    _MODE = "ds"
    _SCREEN = None
    _FE = None
    _DS_STAGE = None
    _DS_PROGRESS_LAST = None
    _DS_SESSION = ds_transport.create_session(
        session_root,
        session_id=session_id,
        poll_interval_sec=poll_interval_sec,
        session_meta=session_meta,
    )


def close_ds(reason: str = "completed") -> None:
    """
    Close the active DS session, if any, and record a final status.
    """
    global _MODE, _SCREEN, _FE, _DS_SESSION, _DS_STAGE, _DS_PROGRESS_LAST
    _deactivate_ds_session(str(reason))
    _MODE = "terminal"
    _SCREEN = None
    _FE = None
    _DS_SESSION = None
    _DS_STAGE = None
    _DS_PROGRESS_LAST = None


def progress_fraction() -> float:
    if PROGRESS_MAX == 0:
        return 0.0
    return round(max(0.0, min(1.0, PROGRESS_TICK / PROGRESS_MAX)), 3)


def _send_progress() -> None:
    global _DS_PROGRESS_LAST
    if _MODE == "gui" and _FE is not None and hasattr(_FE, "set_progress"):
        _FE.set_progress(_SCREEN, progress_fraction(), PROGRESS_TICK, PROGRESS_MAX, PROGRESS_PHASE)
    if _MODE == "ds" and _DS_SESSION is not None:
        payload_key = (int(PROGRESS_TICK), int(PROGRESS_MAX), str(PROGRESS_PHASE))
        if payload_key == _DS_PROGRESS_LAST:
            return
        _DS_PROGRESS_LAST = payload_key
        ds_transport.write_event(
            _DS_SESSION,
            "progress",
            text=str(PROGRESS_PHASE or "progress"),
            progress_fraction=progress_fraction(),
            progress_tick=int(PROGRESS_TICK),
            progress_max=int(PROGRESS_MAX),
            progress_phase=str(PROGRESS_PHASE),
        )


def reset_progress(max_ticks: int, phase: str = "") -> None:
    global PROGRESS_TICK, PROGRESS_MAX, PROGRESS_PHASE
    PROGRESS_TICK = 0
    PROGRESS_MAX = int(max_ticks)
    PROGRESS_PHASE = str(phase)
    _send_progress()


def tick_progress(phase: str = "", inc: int = 1) -> None:
    global PROGRESS_TICK, PROGRESS_PHASE
    PROGRESS_TICK += int(inc)
    if str(phase).strip() != "":
        PROGRESS_PHASE = str(phase)
    _send_progress()


def clear_progress() -> None:
    global PROGRESS_TICK, PROGRESS_MAX, PROGRESS_PHASE
    PROGRESS_TICK = 0
    PROGRESS_MAX = 0
    PROGRESS_PHASE = ""
    _send_progress()


def iget(
    prompt: str,
    default: Optional[str] = None,
    *,
    prompt_meta: Optional[dict[str, Any]] = None,
) -> str:
    """
    Unified input.

    PSEUDOCODE:
    - If terminal mode:
      - call built-in input(prompt)
      - if blank and default is set, return default
      - append "prompt -> value" to _LOG
    - If GUI mode:
      - call _FE.ginput(_SCREEN, _LOG, prompt)
      - unpack value + updated log
      - normalize blank/default handling
      - return value as string
    """
    global _LOG
    context_lines = _build_prompt_context_lines()
    prompt_kind = _infer_prompt_kind(prompt, context_lines)
    context_text = "\n".join(context_lines)
    stage_hint = _stage_hint_for_prompt(prompt_kind)
    meta = dict(prompt_meta) if isinstance(prompt_meta, dict) else _build_prompt_meta(prompt, prompt_kind)

    if _MODE == "ds" and _DS_SESSION is not None:
        _set_ds_stage(stage_hint, text=prompt, source="prompt")
        prompt_id = ds_transport.next_prompt_id(_DS_SESSION)
        ds_transport.write_event(
            _DS_SESSION,
            "prompt",
            text=prompt,
            prompt_id=prompt_id,
            default=default,
            prompt_kind=prompt_kind,
            prompt_meta=meta,
            context_lines=context_lines or None,
            context_text=context_text or None,
            stage_hint=stage_hint,
        )
        dprint(f"DS waiting for reply: prompt_id={prompt_id} kind={prompt_kind} text={str(prompt).strip()[:120]}")
        raw_value = ds_transport.wait_for_reply(_DS_SESSION, prompt_id)
        dprint(f"DS received reply: prompt_id={prompt_id} raw_reply={str(raw_value)[:120]}")
        used_default = raw_value == "" and default is not None
        value = default if used_default else raw_value
        _append_log(prompt, value)
        _append_support_capture(
            "prompt_reply",
            prompt=prompt,
            reply=str(value),
            raw_reply=str(raw_value),
            default=default,
            default_used=used_default,
            prompt_id=prompt_id,
            prompt_kind=prompt_kind,
        )
        return str(value)

    if _MODE == "terminal":
        raw_value = _BASE_INPUT(prompt)
        used_default = raw_value == "" and default is not None
        value = default if used_default else raw_value
        _append_log(prompt, value)
        _append_support_capture(
            "prompt_reply",
            prompt=prompt,
            reply=str(value),
            raw_reply=str(raw_value),
            default=default,
            default_used=used_default,
            prompt_kind=prompt_kind,
        )
        return str(value)

    if _FE is None:
        # Defensive fallback if GUI was not fully initialized.
        raw_value = _BASE_INPUT(prompt)
        used_default = raw_value == "" and default is not None
        value = default if used_default else raw_value
        _append_log(prompt, value)
        _append_support_capture(
            "prompt_reply",
            prompt=prompt,
            reply=str(value),
            raw_reply=str(raw_value),
            default=default,
            default_used=used_default,
            prompt_kind=prompt_kind,
        )
        return str(value)

    try:
        if bool(getattr(_FE, "SUPPORTS_PROMPT_META", False)):
            result = _FE.ginput(_SCREEN, _LOG, prompt, prompt_meta=meta)
        else:
            result = _FE.ginput(_SCREEN, _LOG, prompt)
    except Exception as exc:
        disconnected_exc = getattr(_FE, "BrowserDisconnectedError", None)
        if disconnected_exc is not None and isinstance(exc, disconnected_exc):
            reason = str(getattr(exc, "reason", "")).strip().lower()
            if reason in {"window_closed", "pagehide", "closed", "window-close"}:
                _BASE_PRINT("")
                _BASE_PRINT("Python session ended. Restart from the terminal to continue.")
                #ch = _BASE_INPUT("Continue in terminal? (y): ").strip().lower()
                ch = 'n'
                if ch not in {"y", ""}:
                    raise UserAbortError("User declined terminal resume after browser window closed.")
                init_terminal()
                return iget(prompt, default=default)
            _BASE_PRINT("")
            _BASE_PRINT("Browser disconnected. Continuing in terminal mode.")
            init_terminal()
            return iget(prompt, default=default)
        raise
    try:
        value, updated_log = result[0], result[1]
    except Exception:
        # Fallback for alternate frontend return styles.
        value, updated_log = result, _LOG
    raw_value = value
    _LOG = list(updated_log) if updated_log is not None else _LOG
    used_default = value == "" and default is not None
    if used_default:
        value = default
    _append_log(prompt, value)
    _append_support_capture(
        "prompt_reply",
        prompt=prompt,
        reply=str(value),
        raw_reply=str(raw_value),
        default=default,
        default_used=used_default,
        prompt_kind=prompt_kind,
    )
    return str(value)


def iprint(*parts: Any, sep: str = " ", end: str = "\n") -> None:
    """
    Unified print.

    PSEUDOCODE:
    - If terminal mode: print normally.
    - If GUI mode:
      - pass text into _FE.logprint for display/log panel.
      - keep terminal print fallback for robustness.
    """
    global _LOG
    text = sep.join(str(p) for p in parts)

    if _MODE == "ds" and _DS_SESSION is not None:
        _append_log("print", text)
        _write_ds_output_event(text)
        _append_support_capture("print", text=text)
        return

    if _MODE == "terminal" or _FE is None:
        _BASE_PRINT(text, end=end)
        _append_log("print", text)
        _append_support_capture("print", text=text)
        return

    try:
        updated_log = _FE.logprint(_LOG, text)
        _LOG = list(updated_log) if updated_log is not None else _LOG
    except Exception:
        _BASE_PRINT(text, end=end)
    _append_log("print", text)
    _append_support_capture("print", text=text)


def dprint(*parts: Any, sep: str = " ", end: str = "\n") -> None:
    """
    Debug/technical print.

    In GUI mode this stays in terminal/stdout to keep browser focused on
    prompt-relevant interaction text.
    """
    text = sep.join(str(p) for p in parts)
    sys.stdout.write(text + end)
    sys.stdout.flush()
    if _MODE == "ds" and _DS_SESSION is not None:
        ds_transport.write_event(_DS_SESSION, "debug", text=text)
    _append_log("debug", text)
    _append_support_capture("debug", text=text)


def legacy_print(
    *parts: Any,
    sep: str = " ",
    end: str = "\n",
    file: Any = None,
    flush: bool = False,
    **kwargs: Any,
) -> None:
    """
    Print shim safe to use when legacy code monkey-patches builtins.print.
    """
    global _LOG
    text = sep.join(str(p) for p in parts)
    stream = file if file is not None else sys.stdout

    # Respect explicit file handles (e.g., StringIO in third-party libs) and
    # unknown kwargs by falling back to base print behavior.
    if file is not None or kwargs:
        _BASE_PRINT(*parts, sep=sep, end=end, file=file, flush=flush, **kwargs)
        return

    technical = _is_technical_line(text)

    if _MODE == "gui" and _FE is not None and not technical:
        try:
            updated_log = _FE.logprint(_LOG, text)
            _LOG = list(updated_log) if updated_log is not None else _LOG
        except Exception:
            pass

    # Keep technical lines in terminal; user-facing lines are browser-first.
    if _MODE == "terminal" or technical:
        stream.write(text + end)
        if flush and hasattr(stream, "flush"):
            stream.flush()

    if _MODE == "ds" and _DS_SESSION is not None:
        _write_ds_output_event(text, legacy=True, technical=technical)

    _append_log("print", text)
    _append_support_capture("print", text=text, legacy=True, technical=technical)


def get_log() -> list[str]:
    """
    Return a copy of adapter log for checkpointing/export.
    """
    return list(_LOG)


def get_support_capture() -> list[dict[str, Any]]:
    """
    Return a copy of structured prompt-visible support capture rows.
    """
    return [dict(row) for row in _SUPPORT_CAPTURE]


def clear_log() -> None:
    """
    Reset adapter log.
    """
    _LOG.clear()
    _SUPPORT_CAPTURE.clear()
    _PROMPT_STICKY_OPTIONS.clear()


def _deactivate_ds_session(reason: str) -> None:
    global _DS_SESSION, _DS_STAGE, _DS_PROGRESS_LAST
    if _DS_SESSION is None:
        return
    try:
        ds_transport.write_event(
            _DS_SESSION,
            "session",
            text=f"DS session {reason}",
            event=str(reason),
            session_id=_DS_SESSION.session_id,
        )
        ds_transport.update_session_status(_DS_SESSION, str(reason))
    except Exception:
        pass
    _DS_SESSION = None
    _DS_STAGE = None
    _DS_PROGRESS_LAST = None


def _append_log(context: str, value: Any) -> None:
    """
    Internal helper: append one text log event.
    """
    _LOG.append(f"{context} -> {value}")


def _append_support_capture(kind: str, **fields: Any) -> None:
    row: dict[str, Any] = {
        "index": len(_SUPPORT_CAPTURE),
        "kind": str(kind),
        "mode": _MODE,
    }
    for key, value in fields.items():
        if value is None:
            continue
        row[str(key)] = value
    _SUPPORT_CAPTURE.append(row)


def _write_ds_output_event(
    text: str,
    *,
    legacy: bool = False,
    technical: bool = False,
) -> None:
    if _DS_SESSION is None:
        return

    if _is_low_value_ds_line(text):
        return

    stage = _infer_stage_from_output(text)
    if stage:
        _set_ds_stage(stage, text=text, source="output")

    payload: dict[str, Any] = {
        "text": text,
    }
    if legacy:
        payload["legacy"] = True
        payload["technical"] = technical
    elif technical:
        payload["technical"] = True

    if _should_summarize_ds_output(text):
        line_count = max(len(str(text).splitlines()), 1)
        char_count = len(str(text))
        preview = _compact_context_text(text, max_chars=240)
        payload["text"] = f"[bulk output omitted: {line_count} lines, {char_count} chars] {preview}"
        payload["bulk_output"] = True
        payload["line_count"] = line_count
        payload["char_count"] = char_count
        payload["preview"] = preview

    ds_transport.write_event(_DS_SESSION, "output", **payload)


def _set_ds_stage(stage: str, *, text: str = "", source: str = "") -> None:
    global _DS_STAGE
    if _DS_SESSION is None:
        return
    stage = str(stage).strip()
    if not stage or stage == _DS_STAGE:
        return
    _DS_STAGE = stage
    ds_transport.write_event(
        _DS_SESSION,
        "stage",
        text=_compact_context_text(text, max_chars=180) or stage,
        stage=stage,
        source=source or None,
    )


def _build_prompt_context_lines(limit: int = 6, max_chars: int = 600) -> list[str]:
    """
    Build a compact recent prompt-visible context block for DS prompt rows.
    """
    rows: list[str] = []
    total_chars = 0

    for row in reversed(_SUPPORT_CAPTURE):
        kind = str(row.get("kind", ""))
        if kind == "debug":
            continue
        if kind == "print" and row.get("technical") is True:
            continue
        if kind == "print" and _is_low_value_ds_line(row.get("text", "")):
            continue

        line = _format_context_line(row)
        if not line:
            continue

        if rows and total_chars + len(line) > max_chars:
            break
        rows.append(line)
        total_chars += len(line)
        if len(rows) >= limit:
            break

    rows.reverse()
    return rows


def _format_context_line(row: dict[str, Any]) -> str:
    kind = str(row.get("kind", ""))
    if kind == "print":
        if _is_low_value_ds_line(row.get("text", "")):
            return ""
        return _compact_context_text(row.get("text", ""))
    if kind == "prompt_reply":
        prompt = _compact_context_text(row.get("prompt", ""))
        reply = _compact_context_text(row.get("reply", ""))
        if not prompt and not reply:
            return ""
        return f"{prompt} => {reply}"
    return ""


def _compact_context_text(value: Any, max_chars: int = 160) -> str:
    text = str(value).strip()
    if not text:
        return ""
    text = re.sub(r"\s+", " ", text)
    if len(text) <= max_chars:
        return text
    head = max_chars - 3
    return text[:head].rstrip() + "..."


def _build_prompt_meta(prompt: str, prompt_kind: str) -> dict[str, Any]:
    recent_lines = _collect_recent_print_lines_since_last_reply()
    done_cue = _has_done_cue(prompt, recent_lines)
    sticky_key = _normalize_prompt_key(prompt)
    sticky_allowed = bool(done_cue or prompt_kind in {"category_picker", "column_picker"})

    options: list[dict[str, str]] = []
    if prompt_kind == "yes_no":
        options = _build_yes_no_options()
    else:
        numeric_options = _extract_last_numeric_option_block(recent_lines)
        if numeric_options:
            options = _normalize_option_rows(numeric_options)
            if sticky_allowed and sticky_key:
                _PROMPT_STICKY_OPTIONS[sticky_key] = [dict(opt) for opt in options]
                _trim_prompt_sticky_cache()
        elif sticky_allowed and sticky_key:
            options = [dict(opt) for opt in _PROMPT_STICKY_OPTIONS.get(sticky_key, [])]

        if done_cue:
            options = _with_done_option(options)
        elif sticky_key and not sticky_allowed:
            _PROMPT_STICKY_OPTIONS.pop(sticky_key, None)

    meta: dict[str, Any] = {"kind": str(prompt_kind)}
    if options:
        meta["options"] = options
    if done_cue:
        meta["done_cue"] = True
    return meta


def _collect_recent_print_lines_since_last_reply() -> list[str]:
    last_reply_idx = -1
    for i in range(len(_SUPPORT_CAPTURE) - 1, -1, -1):
        if str(_SUPPORT_CAPTURE[i].get("kind", "")) == "prompt_reply":
            last_reply_idx = i
            break

    rows = _SUPPORT_CAPTURE[last_reply_idx + 1 :]
    lines: list[str] = []
    for row in rows:
        if str(row.get("kind", "")) != "print":
            continue
        if row.get("technical") is True:
            continue
        text = str(row.get("text", ""))
        if not text:
            continue
        for part in text.splitlines():
            lines.append(part)
    return lines


def _extract_last_numeric_option_block(lines: list[str]) -> list[dict[str, str]]:
    current: list[dict[str, str]] = []
    last: list[dict[str, str]] = []
    for raw in lines:
        parsed = _parse_numeric_option_line(raw)
        if parsed is not None:
            current.append(parsed)
            continue
        if current:
            last = current
            current = []
    if current:
        last = current
    return last


def _parse_numeric_option_line(line: str) -> Optional[dict[str, str]]:
    match = re.match(r"^\s*(\d+|x|y|n)\s*:\s*(.+?)\s*$", str(line or ""), flags=re.IGNORECASE)
    if not match:
        return None
    value = str(match.group(1)).strip()
    if value.isalpha():
        value = value.lower()
    label = str(match.group(2)).strip()
    if not value:
        return None
    return {"value": value, "label": label, "description": label}


def _normalize_option_rows(options: list[dict[str, str]]) -> list[dict[str, str]]:
    out: list[dict[str, str]] = []
    for option in options:
        value = str(option.get("value", "")).strip()
        if not value:
            continue
        label = str(option.get("label", value))
        description = str(option.get("description", label))
        out.append({"value": value, "label": label, "description": description})
    return out


def _build_yes_no_options() -> list[dict[str, str]]:
    return [
        {"value": "y", "label": "yes", "description": "Submit y"},
        {"value": "n", "label": "no", "description": "Submit n"},
    ]


def _with_done_option(options: list[dict[str, str]]) -> list[dict[str, str]]:
    out = [dict(opt) for opt in options]
    has_done = any(str(opt.get("label", "")).strip().lower() == "done" or str(opt.get("value", "")).strip().lower() in ("", "x") for opt in out)
    if not has_done:
        out.append({"value": "", "label": "done", "description": "Submit blank"})
    return out


def _has_done_cue(prompt: str, recent_lines: list[str]) -> bool:
    cue_pattern = re.compile(r"(when done|to continue)", re.IGNORECASE)
    if cue_pattern.search(str(prompt or "")):
        return True
    for line in recent_lines:
        if cue_pattern.search(str(line or "")):
            return True
    return False


def _normalize_prompt_key(prompt: str) -> str:
    text = re.sub(r"\s+", " ", str(prompt or "").strip().lower())
    if not text:
        return ""
    return text[:240]


def _trim_prompt_sticky_cache() -> None:
    if len(_PROMPT_STICKY_OPTIONS) <= _MAX_PROMPT_STICKY:
        return
    keys = list(_PROMPT_STICKY_OPTIONS.keys())
    for key in keys[: len(keys) - _MAX_PROMPT_STICKY]:
        _PROMPT_STICKY_OPTIONS.pop(key, None)


def _infer_prompt_kind(prompt: str, context_lines: list[str]) -> str:
    prompt_low = str(prompt).strip().lower()
    context_low = "\n".join(context_lines).lower()
    has_path_word = bool(re.search(r"\bpath\b", context_low))
    has_menu_context = _context_has_menu_lines(context_lines)

    if "int or range() to add" in prompt_low:
        return "category_picker"
    if (
        prompt_low.startswith("truth category")
        or prompt_low.startswith("repeat analysis on each unique value in")
        or prompt_low.startswith("category to ")
        or prompt_low.endswith(" category:")
    ):
        return "category_picker"
    if "sort x axis by" in prompt_low or "color boxes by" in prompt_low:
        return "column_picker"
    if prompt_low.startswith("number:") and has_menu_context:
        return "menu"
    if prompt_low.startswith("number") and has_menu_context:
        return "menu"
    if (
        prompt_low.startswith("confidence threshold")
        or prompt_low.startswith("training row cap")
        or prompt_low.startswith("iteration cap")
        or prompt_low.startswith("convergence tolerance")
        or prompt_low.startswith("n clusters")
        or prompt_low.startswith("number of iterations")
    ):
        return "free_text"
    if (
        prompt_low.startswith("output name")
        or prompt_low.startswith("prediction output name")
        or prompt_low.startswith("evaluation output stem")
        or prompt_low.startswith("unknown label")
    ):
        return "free_text"
    if prompt_low.startswith("model file") or "csv path" in prompt_low:
        return "path"
    if (
        "(y" in prompt_low
        or prompt_low.startswith("quit?")
        or prompt_low.startswith("done?")
        or prompt_low.startswith("change?")
    ):
        return "yes_no"
    if (
        prompt_low == ":"
        or prompt_low.startswith("path:")
        or ("folder" in context_low and not has_menu_context)
        or has_path_word
        or "output root" in context_low
    ):
        return "path"
    if prompt_low.startswith("number:") or (
        prompt_low.startswith("number") and has_menu_context
    ):
        return "menu"
    if "column" in prompt_low or "column" in context_low:
        return "column_picker"
    return "free_text"


def _context_has_menu_lines(context_lines: list[str]) -> bool:
    for line in context_lines:
        stripped = line.strip().lower()
        if re.match(r"^\d+\s*:", stripped):
            return True
        if "send non-int" in stripped:
            return True
    return False


def _stage_hint_for_prompt(prompt_kind: str) -> str:
    if prompt_kind == "menu":
        return "awaiting_menu_choice"
    if prompt_kind == "yes_no":
        return "awaiting_confirmation"
    if prompt_kind == "path":
        return "awaiting_path"
    if prompt_kind == "category_picker":
        return "awaiting_category_selection"
    if prompt_kind == "column_picker":
        return "awaiting_column_selection"
    return "awaiting_input"


def _infer_stage_from_output(text: str) -> str | None:
    low = str(text).strip().lower()
    if not low:
        return None
    if "send non-int when done" in low or "return to previous menu" in low:
        return "building_commands"
    if "running subcommand:" in low or "executing com" in low:
        return "executing_commands"
    if _should_summarize_ds_output(text):
        return "showing_bulk_output"
    return None


def _is_low_value_ds_line(text: Any) -> bool:
    raw = str(text)
    stripped = raw.strip()
    if not stripped:
        return True
    low = stripped.lower()
    if low == "not logged !!!":
        return True
    if low == "send non-int when done (return df)":
        return True
    if low.endswith("all index the same"):
        return True
    return False


def _should_summarize_ds_output(text: str) -> bool:
    raw = str(text)
    stripped = raw.strip()
    if not stripped:
        return False

    line_count = len(raw.splitlines())
    char_count = len(raw)
    if line_count <= 12 and char_count <= 1200:
        return False

    low = stripped.lower()
    for token in (
        "send non-int",
        "number:",
        "choose",
        "select",
        "path:",
        "output root folder",
        "new label",
        "category",
        "column to",
    ):
        if token in low:
            return False

    if _context_has_menu_lines(raw.splitlines()):
        return False

    return True


def _is_technical_line(text: str) -> bool:
    s = str(text).strip()
    if not s:
        return False
    low = s.lower()

    # Keep prompt/menu context in browser.
    for token in (
        "send non-int",
        "number:",
        "choose",
        "select",
        "repeat analysis on each unique value",
        "running subcommand:",
        "press enter on empty line",
        "path:",
        "output root folder",
        "new label",
        "category",
        "column to",
    ):
        if token in low:
            return False

    if low.startswith("traceback (most recent call last)"):
        return True
    if low.startswith("during:"):
        return True
    if low.startswith('file "') and ", line " in low:
        return True
    if low.startswith("<class '") and "error" in low:
        return True

    for token in (
        "typingerror:",
        "numba.core.errors",
        "numba",
        "pynndescent",
        "could not excecute",
        "exception in",
        "com into menu",
        "coms out of menu",
        "executing com",
        "all index the same",
        "devmode",
    ):
        if token in low:
            return True
    if "shape" in low and "menu" not in low:
        return True
    return False
