"""
Lightweight HTML frontend for New_DAS interactive I/O.

This module is designed to pair with `io_adapter.py`:
- `ginput(screen, log, prompt)` waits for user input from a browser page.
- `logprint(log, *parts)` appends to both the adapter log and browser log panel.

Typical usage:
    import io_adapter as io
    import frontend

    screen = frontend.start_server(open_browser=True)
    io.init_gui(screen, frontend)
"""

from __future__ import annotations

import importlib.util
import json
import threading
import webbrowser
import time
from dataclasses import dataclass, field
from html import escape
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any, Optional
from urllib.parse import parse_qs


DEFAULT_HOST = "127.0.0.1"
DEFAULT_PORT = 8765
MAX_SESSION_LOG_LINES = 5000
DEFAULT_RENDER_LOG_LIMIT = 800
DEFAULT_DISCONNECT_TIMEOUT_SEC = 8.0
DEFAULT_CLOSE_CONFIRM_SEC = 1.0

_ACTIVE_SESSION: "HtmlSession | None" = None
SUPPORTS_PROMPT_META = True

if __spec__ is None and "__file__" in globals():
    __spec__ = importlib.util.spec_from_file_location(__name__, __file__)
    if __spec__ is not None:
        __loader__ = __spec__.loader


class BrowserDisconnectedError(RuntimeError):
    """Raised when a GUI prompt is pending but browser polling has stopped."""

    def __init__(self, message: str, *, reason: str = "disconnected"):
        super().__init__(message)
        self.reason = str(reason)


@dataclass
class HtmlSession:
    host: str = DEFAULT_HOST
    port: int = DEFAULT_PORT
    title: str = "New_DAS Frontend"
    log: list[str] = field(default_factory=list)
    pending_prompt: Optional[str] = None
    pending_answer: Optional[str] = None
    lock: threading.Lock = field(default_factory=threading.Lock, repr=False)
    answer_event: threading.Event = field(default_factory=threading.Event, repr=False)
    server: Optional[ThreadingHTTPServer] = field(default=None, repr=False)
    thread: Optional[threading.Thread] = field(default=None, repr=False)
    last_state_poll: float = field(default_factory=time.monotonic)
    disconnect_timeout_sec: float = DEFAULT_DISCONNECT_TIMEOUT_SEC
    close_confirm_sec: float = DEFAULT_CLOSE_CONFIRM_SEC
    last_disconnect_signal: float = 0.0
    last_disconnect_reason: str = ""
    pending_prompt_meta: Optional[dict[str, Any]] = None
    progress_fraction: float = 0.0
    progress_tick: int = 0
    progress_max: int = 0
    progress_phase: str = ""

    @property
    def base_url(self) -> str:
        return f"http://{self.host}:{self.port}/"


def start_server(
    *,
    host: str = DEFAULT_HOST,
    port: int = DEFAULT_PORT,
    title: str = "New_DAS Frontend",
    open_browser: bool = False,
) -> HtmlSession:
    """
    Start a tiny local web server for prompt/response interaction.
    """
    session = HtmlSession(host=host, port=port, title=title)
    handler_cls = _build_handler(session)
    server = ThreadingHTTPServer((host, int(port)), handler_cls)
    session.port = int(server.server_address[1])
    session.server = server
    session.thread = threading.Thread(target=server.serve_forever, daemon=True)
    session.thread.start()

    global _ACTIVE_SESSION
    _ACTIVE_SESSION = session

    if open_browser:
        webbrowser.open(session.base_url)
    return session


def stop_server(screen: Any = None) -> None:
    """
    Stop the active server (or a provided session object).
    """
    session = _coerce_session(screen)
    if session is None or session.server is None:
        return

    try:
        session.server.shutdown()
        session.server.server_close()
    finally:
        session.server = None
        session.thread = None
        session.answer_event.set()
        if _ACTIVE_SESSION is session:
            _set_active_session(None)


def set_progress(screen: Any, fraction: float, tick: int, max_tick: int, phase: str) -> None:
    session = _coerce_session(screen)
    if session is None:
        return
    with session.lock:
        session.progress_fraction = float(fraction)
        session.progress_tick = int(tick)
        session.progress_max = int(max_tick)
        session.progress_phase = str(phase)


def ginput(screen: Any, log: Any, prompt: str, prompt_meta: Optional[dict[str, Any]] = None) -> list[Any]:
    """
    Browser-backed input endpoint expected by io_adapter.

    Returns:
        [value, updated_log]
    """
    session = _coerce_session(screen)
    if session is None:
        value = input(prompt)
        out_log = _ensure_log(log)
        out_log.append(f"{prompt} -> {value}")
        return [value, out_log]

    out_log = _ensure_log(log)
    with session.lock:
        session.pending_prompt = str(prompt)
        session.pending_prompt_meta = dict(prompt_meta) if isinstance(prompt_meta, dict) else None
        session.pending_answer = None
        session.answer_event.clear()
        _append_session_log(session, str(prompt))
    out_log.append(str(prompt))

    pause_logged = False
    while True:
        if session.answer_event.wait(timeout=0.25):
            break
        with session.lock:
            last_poll = float(session.last_state_poll)
            timed_out = (time.monotonic() - last_poll) > float(session.disconnect_timeout_sec)
            server_stopped = session.server is None
            close_signal = float(session.last_disconnect_signal)
            close_reason = str(session.last_disconnect_reason or "window_closed")
            close_confirm_sec = float(session.close_confirm_sec)
            close_confirmed = close_signal > last_poll and ((time.monotonic() - close_signal) >= close_confirm_sec)
        if close_confirmed:
            with session.lock:
                session.pending_prompt = None
                session.pending_prompt_meta = None
                session.pending_answer = None
                session.answer_event.clear()
                _append_session_log(session, "[browser window closed]")
            raise BrowserDisconnectedError("Browser window closed while awaiting input.", reason=close_reason)
        if server_stopped:
            with session.lock:
                session.pending_prompt = None
                session.pending_prompt_meta = None
                session.pending_answer = None
                session.answer_event.clear()
                _append_session_log(session, "[browser disconnected]")
            raise BrowserDisconnectedError("Browser polling stopped while awaiting input.", reason="server_stopped")
        if not timed_out:
            if pause_logged:
                pause_logged = False
            continue
        if not pause_logged:
            pause_logged = True

    with session.lock:
        answer = session.pending_answer if session.pending_answer is not None else ""
        session.pending_prompt = None
        session.pending_prompt_meta = None
        session.pending_answer = None
        session.answer_event.clear()
        _append_session_log(session, f"user: {answer}")

    out_log.append(f"user: {answer}")
    return [answer, out_log]


def logprint(log: Any, *parts: Any) -> list[str]:
    """
    Browser-backed print endpoint expected by io_adapter.

    Returns:
        updated_log
    """
    out_log = _ensure_log(log)
    session = _coerce_session(None)

    for part in parts:
        if isinstance(part, list):
            text = ", ".join(str(x) for x in part)
        else:
            text = str(part)
        out_log.append(text)
        if session is not None:
            with session.lock:
                _append_session_log(session, text)

    return out_log


def _coerce_session(screen: Any) -> HtmlSession | None:
    if isinstance(screen, HtmlSession):
        return screen
    return _ACTIVE_SESSION


def _ensure_log(log: Any) -> list[str]:
    if log is None:
        return []
    if isinstance(log, list):
        return log
    try:
        return [str(x) for x in list(log)]
    except Exception:
        return [str(log)]


def _set_active_session(session: HtmlSession | None) -> None:
    global _ACTIVE_SESSION
    _ACTIVE_SESSION = session


def _append_session_log(session: HtmlSession, text: str) -> None:
    session.log.append(str(text))
    if len(session.log) > MAX_SESSION_LOG_LINES:
        del session.log[: len(session.log) - MAX_SESSION_LOG_LINES]


def _build_handler(session: HtmlSession):
    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            if self.path.startswith("/state"):
                with session.lock:
                    session.last_state_poll = time.monotonic()
                    prompt_text = _safe(session.pending_prompt)
                    prompt_meta = dict(session.pending_prompt_meta) if isinstance(session.pending_prompt_meta, dict) else None
                    prompt_pending = bool(session.pending_prompt)
                    lines = list(session.log)
                self._json(
                    {
                        "title": session.title,
                        "prompt": prompt_text,
                        "prompt_meta": prompt_meta,
                        "prompt_pending": prompt_pending,
                        "backend_active": bool(session.server is not None),
                        "session_active": _ACTIVE_SESSION is session,
                        "log_count": len(lines),
                        "render_log_limit": DEFAULT_RENDER_LOG_LIMIT,
                        "log": lines,
                        "progress_fraction": float(session.progress_fraction),
                        "progress_tick": int(session.progress_tick),
                        "progress_max": int(session.progress_max),
                        "progress_phase": str(session.progress_phase),
                    }
                )
                return
            self._html(_render_index(session.title))

        def do_POST(self):
            if self.path.startswith("/disconnect"):
                payload = self._read_payload()
                reason = str(payload.get("reason", "window_closed")).strip() or "window_closed"
                with session.lock:
                    session.last_disconnect_signal = time.monotonic()
                    session.last_disconnect_reason = reason
                self._json({"ok": True})
                return

            if not self.path.startswith("/answer"):
                self._json({"ok": False, "error": "not found"}, code=404)
                return

            payload = self._read_payload()
            answer = str(payload.get("answer", ""))
            with session.lock:
                if not session.pending_prompt:
                    self._json({"ok": False, "error": "no pending prompt"}, code=409)
                    return
                session.pending_answer = answer
                session.answer_event.set()
            self._json({"ok": True})

        def _read_payload(self) -> dict[str, Any]:
            length = int(self.headers.get("Content-Length", "0"))
            raw = self.rfile.read(length) if length > 0 else b""
            body = raw.decode("utf-8", errors="replace")
            ctype = self.headers.get("Content-Type", "").lower()
            if "application/json" in ctype:
                try:
                    data = json.loads(body)
                    return data if isinstance(data, dict) else {}
                except Exception:
                    return {}
            parsed = parse_qs(body, keep_blank_values=True)
            out = {}
            for key, values in parsed.items():
                out[key] = values[0] if values else ""
            return out

        def _json(self, payload: dict[str, Any], code: int = 200):
            body = json.dumps(payload).encode("utf-8")
            self.send_response(code)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def _html(self, text: str, code: int = 200):
            body = text.encode("utf-8")
            self.send_response(code)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, fmt: str, *args):
            # Silence request logs by default.
            return

    return Handler


def _safe(value: Any) -> str:
    if value is None:
        return ""
    return str(value)


def _render_index(title: str) -> str:
    title = escape(title)
    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>{title}</title>
  <link rel="preconnect" href="https://fonts.googleapis.com" />
  <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin />
  <link href="https://fonts.googleapis.com/css2?family=IBM+Plex+Sans:wght@400;600;700&family=IBM+Plex+Mono:wght@400;500&display=swap" rel="stylesheet" />
  <style>
    :root {{
      --bg: #2b2f34;
      --panel: #ffffff;
      --line: #d4dde8;
      --line-soft: #e8eef5;
      --ink: #16202d;
      --muted: #536273;
      --accent: #145ea8;
      --accent-soft: #e8f1fb;
      --panel-height: 58vh;
    }}
    * {{ box-sizing: border-box; }}
    body {{
      margin: 0;
      font-family: "IBM Plex Sans", "Segoe UI", sans-serif;
      color: var(--ink);
      background: var(--bg);
    }}
    .wrap {{
      max-width: 1180px;
      margin: 0 auto;
      padding: 16px 18px;
    }}
    .card {{
      background: var(--panel);
      border: 1px solid var(--line);
      border-radius: 12px;
      padding: 14px 16px;
      box-shadow: 0 6px 22px rgba(15, 23, 42, 0.05);
    }}
    .topbar {{
      margin-bottom: 12px;
    }}
    .titleRow {{
      display: flex;
      flex-wrap: wrap;
      align-items: baseline;
      justify-content: space-between;
      gap: 8px 16px;
    }}
    h1 {{
      margin: 0;
      font-size: 19px;
      font-weight: 700;
      font-family: "IBM Plex Sans", "Segoe UI", sans-serif;
    }}
    .status {{
      color: var(--muted);
      font-size: 13px;
      white-space: nowrap;
      overflow: hidden;
      text-overflow: ellipsis;
    }}
    .workspace {{
      display: grid;
      grid-template-columns: minmax(260px, 0.92fr) minmax(380px, 1.4fr);
      gap: 12px;
      align-items: start;
      margin-bottom: 12px;
    }}
    .sectionTitle {{
      margin: 0 0 10px;
      font-size: 13px;
      text-transform: uppercase;
      letter-spacing: 0.05em;
      color: var(--muted);
      font-weight: 700;
      font-family: "IBM Plex Sans", "Segoe UI", sans-serif;
    }}
    .actionCard {{
      background: linear-gradient(180deg, #f5fbf8 0%, #ffffff 70%);
      border-color: #cfe3db;
    }}
    .actionMain {{
      height: var(--panel-height);
      display: flex;
      flex-direction: column;
      gap: 8px;
    }}
    .prompt {{
      margin: 0;
      min-height: 42px;
      max-height: calc(1.45em * 4 + 22px);
      flex: 0 0 auto;
      overflow: auto;
      border: 1px solid var(--line-soft);
      border-radius: 9px;
      padding: 10px 11px;
      background: #f4f9ff;
      font-family: "IBM Plex Mono", Consolas, "Courier New", monospace;
      font-size: 14px;
      line-height: 1.45;
      white-space: pre-wrap;
    }}
    .progressWrap {{
      flex: 0 0 auto;
      border: 1px solid #d8e8de;
      background: #eef8f2;
      border-radius: 9px;
      padding: 10px 11px;
      display: none;
      gap: 6px;
    }}
    .progressWrap.active {{
      display: grid;
    }}
    .progressPhase {{
      font-size: 13px;
      font-weight: 600;
      color: var(--ink);
      white-space: nowrap;
      overflow: hidden;
      text-overflow: ellipsis;
    }}
    .progressTrack {{
      width: 100%;
      height: 12px;
      background: #dbe8df;
      border-radius: 999px;
      overflow: hidden;
    }}
    .progressFill {{
      height: 100%;
      width: 0%;
      background: linear-gradient(90deg, #2f7dd1 0%, #42b883 100%);
      border-radius: 999px;
      transition: width 120ms ease;
    }}
    .progressText {{
      font-size: 12px;
      color: var(--muted);
      font-family: "IBM Plex Mono", Consolas, "Courier New", monospace;
    }}
    .optionHint {{
      padding: 0 2px 6px;
      margin-bottom: 2px;
      border-bottom: 1px dashed #d3e4d9;
      color: var(--muted);
      font-size: 12px;
    }}
    .optionsPanel {{
      flex: 1 1 auto;
      min-height: 0;
      overflow: auto;
      border: 1px solid #d8e8de;
      background: #eef8f2;
      border-radius: 9px;
      padding: 9px;
      display: flex;
      flex-direction: column;
      gap: 6px;
    }}
    .inputForm {{
      margin: 0;
    }}
    .row {{
      display: flex;
      gap: 8px;
      flex-wrap: wrap;
    }}
    input[type="text"] {{
      flex: 1;
      min-width: 220px;
      border: 1px solid var(--line);
      border-radius: 8px;
      padding: 10px 12px;
      font-size: 14px;
      font-family: "IBM Plex Mono", Consolas, "Courier New", monospace;
      background: #ffffff;
      color: var(--ink);
      outline: none;
      transition: border-color 120ms ease, box-shadow 120ms ease;
    }}
    input[type="text"]:focus {{
      border-color: #7aa8d7;
      box-shadow: 0 0 0 3px #dcecff;
    }}
    button {{
      border: 1px solid transparent;
      border-radius: 8px;
      padding: 10px 14px;
      cursor: pointer;
      font: 600 13px/1.2 "IBM Plex Sans", "Segoe UI", sans-serif;
      transition: background-color 100ms ease, border-color 100ms ease, color 100ms ease;
    }}
    .submitBtn {{
      background: var(--accent);
      color: #ffffff;
      border-color: #114f8d;
      min-width: 94px;
    }}
    .submitBtn:hover:enabled {{
      background: #0f548f;
    }}
    .optionBtn {{
      background: #e7f2ff;
      border-color: #bfd8ee;
      color: #1b4f7d;
      padding: 8px 11px;
      border-radius: 8px;
      font-weight: 600;
      text-align: left;
      width: 100%;
    }}
    .optionBtn:hover:enabled,
    .optionBtn:focus-visible:enabled {{
      background: #deecfa;
      border-color: #aac8e7;
    }}
    button {{
      outline: none;
    }}
    button:disabled {{
      background: #d1d8e1;
      border-color: #c5ced8;
      color: #667489;
      cursor: not-allowed;
    }}
    .logCard {{
      background: linear-gradient(180deg, #f4f8ff 0%, #fcfdff 72%);
      border-color: #cbdcf0;
    }}
    .log {{
      height: var(--panel-height);
      overflow: auto;
      border: 1px solid var(--line-soft);
      border-radius: 8px;
      padding: 10px;
      font-family: "IBM Plex Mono", Consolas, "Courier New", monospace;
      font-size: 13px;
      line-height: 1.4;
      background: #f8fbff;
      white-space: pre-wrap;
    }}
    .line {{
      margin-bottom: 4px;
      border-bottom: 1px dashed #e9eff7;
      padding-bottom: 4px;
    }}
    .inputCard {{
      background: linear-gradient(180deg, #f9fcff 0%, #ffffff 100%);
      border-color: #cdddee;
    }}
    @media (max-width: 920px) {{
      .workspace {{
        grid-template-columns: 1fr;
      }}
      .prompt {{
        min-height: 42px;
      }}
      .log,
      .actionMain {{
        height: 34vh;
      }}
      .status {{
        white-space: normal;
      }}
    }}
  </style>
</head>
<body>
  <div class="wrap">
    <div class="card topbar">
      <div class="titleRow">
        <h1>{title}</h1>
        <div id="status" class="status">Checking backend status...</div>
      </div>
    </div>

    <div class="workspace">
      <div class="card logCard">
        <div class="sectionTitle">Session Log</div>
        <div id="log" class="log"></div>
      </div>

      <div class="card actionCard">
        <div class="sectionTitle">Current Prompt</div>
        <div class="actionMain">
          <div id="progressWrap" class="progressWrap">
            <div id="progressPhase" class="progressPhase"></div>
            <div class="progressTrack"><div id="progressFill" class="progressFill"></div></div>
            <div id="progressText" class="progressText"></div>
          </div>
          <div id="prompt" class="prompt">Waiting for prompt...</div>
          <div id="options" class="optionsPanel"></div>
        </div>
      </div>
    </div>

    <div class="card inputCard">
      <form id="answerForm" class="inputForm">
        <div class="row">
          <input id="answer" type="text" autocomplete="off" placeholder="Type response here" />
          <button id="submitBtn" class="submitBtn" type="submit">Submit</button>
        </div>
      </form>
    </div>
  </div>

  <script>
    const promptEl = document.getElementById("prompt");
    const statusEl = document.getElementById("status");
    const logEl = document.getElementById("log");
    const optionsEl = document.getElementById("options");
    const progressWrap = document.getElementById("progressWrap");
    const progressPhase = document.getElementById("progressPhase");
    const progressFill = document.getElementById("progressFill");
    const progressText = document.getElementById("progressText");
    const form = document.getElementById("answerForm");
    const answer = document.getElementById("answer");
    const submitBtn = document.getElementById("submitBtn");
    let lastCount = -1;
    let lastPrompt = "";
    let lastPromptDisplay = "";
    let lastPromptPending = null;
    let lastOptionSignature = "";
    let seenBackend = false;

    function resizePromptBox(text) {{
      const promptText = String(text || "");
      const explicitLines = promptText.split(/\\r?\\n/);
      const boxWidth = Math.max(240, promptEl.clientWidth || promptEl.getBoundingClientRect().width || 320);
      const approxCharsPerLine = Math.max(18, Math.floor((boxWidth - 24) / 8));
      let lineCount = 0;
      for (const line of explicitLines) {{
        const textLine = String(line || "");
        lineCount += Math.max(1, Math.ceil(textLine.length / approxCharsPerLine));
      }}
      lineCount = Math.max(1, lineCount);
      const rows = Math.max(1, Math.min(4, lineCount));
      const lineHeight = 1.45 * 14;
      const pad = 22;
      promptEl.style.height = `${{Math.round(rows * lineHeight + pad)}}px`;
      promptEl.style.overflowY = lineCount > 4 ? "auto" : "hidden";
    }}

    function parseOptionLine(line) {{
      const optionPattern = /^\\s*(\\d+|x|y|n)\\s*:\\s*(.+?)\\s*$/i;
      const m = String(line || "").match(optionPattern);
      if (!m) {{
        return null;
      }}
      const value = /^[a-z]+$/i.test(m[1]) ? m[1].toLowerCase() : m[1];
      return {{
        value,
        label: m[2],
        description: m[2],
      }};
    }}

    function parsePromptOptions(promptText) {{
      const lines = String(promptText || "").split(/\\r?\\n/);
      const bodyLines = [];
      const options = [];
      for (const line of lines) {{
        const option = parseOptionLine(line);
        if (option) {{
          options.push(option);
        }} else {{
          bodyLines.push(line);
        }}
      }}
      return {{
        options,
        bodyText: bodyLines.join("\\n").trim(),
      }};
    }}

    function normalizeMetaOptions(rawOptions) {{
      if (!Array.isArray(rawOptions)) {{
        return [];
      }}
      const out = [];
      for (const raw of rawOptions) {{
        if (!raw || typeof raw !== "object") {{
          continue;
        }}
        const value = String(raw.value ?? "").trim();
        if (!value) {{
          continue;
        }}
        const label = String(raw.label ?? value);
        const description = String(raw.description ?? label);
        out.push({{ value, label, description }});
      }}
      return out;
    }}

    function isYesNoPrompt(promptText) {{
      return /\\(y\\)\\s*:?[\\s]*$/i.test(String(promptText || "").trim());
    }}

    function findLastUserLineIndex(logLines) {{
      for (let i = logLines.length - 1; i >= 0; i -= 1) {{
        if (/^\\s*user\\s*:/i.test(String(logLines[i] || ""))) {{
          return i;
        }}
      }}
      return -1;
    }}

    function getRecentLinesAfterLastUser(logLines) {{
      const start = findLastUserLineIndex(logLines) + 1;
      const recentEntries = logLines.slice(start);
      const out = [];
      for (const entry of recentEntries) {{
        const parts = String(entry || "").split(/\\r?\\n/);
        for (const part of parts) {{
          out.push(part);
        }}
      }}
      return out;
    }}

    function parseRecentLogNumericOptions(logLines) {{
      const recent = getRecentLinesAfterLastUser(logLines);
      let currentBlock = [];
      let lastBlock = [];
      for (const rawLine of recent) {{
        const option = parseOptionLine(rawLine);
        if (option) {{
          currentBlock.push(option);
          continue;
        }}
        if (currentBlock.length) {{
          lastBlock = currentBlock;
          currentBlock = [];
        }}
      }}
      if (currentBlock.length) {{
        lastBlock = currentBlock;
      }}
      return lastBlock;
    }}

    function hasDoneCue(promptText, logLines) {{
      const cuePattern = /(when done|to continue)/i;
      if (cuePattern.test(String(promptText || ""))) {{
        return true;
      }}
      const recent = getRecentLinesAfterLastUser(logLines);
      for (const line of recent) {{
        if (cuePattern.test(String(line || ""))) {{
          return true;
        }}
      }}
      return false;
    }}

    function buildYesNoOptions() {{
      return [
        {{ value: "y", label: "yes", description: "Submit y" }},
        {{ value: "n", label: "no", description: "Submit n" }},
      ];
    }}

    function withDoneOption(options) {{
      const out = Array.isArray(options) ? options.slice() : [];
      const hasDone = out.some(opt => String(opt.label || "").toLowerCase() === "done" || ["", "x"].includes(String(opt.value || "").toLowerCase()));
      if (!hasDone) {{
        out.push({{ value: "", label: "done", description: "Submit blank" }});
      }}
      return out;
    }}

    function renderOptionButtons(options, enabled) {{
      optionsEl.innerHTML = "";
      if (!options.length || !enabled) {{
        return;
      }}
      const optionHintEl = document.createElement("div");
      optionHintEl.className = "optionHint";
      const defaultHint = `${{options.length}} option(s) available.`;
      optionHintEl.textContent = defaultHint;
      optionsEl.appendChild(optionHintEl);
      for (const option of options) {{
        const btn = document.createElement("button");
        btn.type = "button";
        btn.className = "optionBtn";
        btn.textContent = `${{option.value}}: ${{option.label}}`;
        btn.title = option.description;
        btn.disabled = false;
        btn.dataset.value = option.value;
        btn.addEventListener("click", () => submitAnswer(option.value));
        btn.addEventListener("mouseenter", () => {{
          optionHintEl.textContent = option.description;
        }});
        btn.addEventListener("mouseleave", () => {{
          optionHintEl.textContent = defaultHint;
        }});
        btn.addEventListener("focus", () => {{
          optionHintEl.textContent = option.description;
        }});
        btn.addEventListener("blur", () => {{
          optionHintEl.textContent = defaultHint;
        }});
        optionsEl.appendChild(btn);
      }}
    }}

    function renderProgress(progressFraction, progressTick, progressMax, progressPhaseText) {{
      const active = Number(progressMax || 0) > 0;
      if (!active) {{
        progressWrap.classList.remove("active");
        progressPhase.textContent = "";
        progressFill.style.width = "0%";
        progressText.textContent = "";
        return;
      }}
      const pct = Math.max(0, Math.min(100, Math.round(Number(progressFraction || 0) * 100)));
      progressWrap.classList.add("active");
      progressPhase.textContent = String(progressPhaseText || "Working...");
      progressFill.style.width = `${{pct}}%`;
      progressText.textContent = `${{Number(progressTick || 0)}} / ${{Number(progressMax || 0)}}`;
    }}

    async function submitAnswer(value) {{
      if (submitBtn.disabled) {{
        return;
      }}
      const payload = new URLSearchParams();
      payload.set("answer", String(value));
      try {{
        await fetch("/answer", {{
          method: "POST",
          headers: {{ "Content-Type": "application/x-www-form-urlencoded" }},
          body: payload.toString(),
        }});
      }} finally {{
        answer.value = "";
        if (!submitBtn.disabled) {{
          answer.focus();
        }}
      }}
    }}

    async function loadState() {{
      try {{
        const res = await fetch("/state", {{ cache: "no-store" }});
        if (!res.ok) return;
        const data = await res.json();
        seenBackend = true;
        const backendActive = Boolean(data.backend_active);
        const sessionActive = Boolean(data.session_active);
        const promptPending = Boolean(data.prompt_pending);
        const promptText = data.prompt ? String(data.prompt) : "";
        const progressFraction = Number(data.progress_fraction || 0);
        const progressTick = Number(data.progress_tick || 0);
        const progressMax = Number(data.progress_max || 0);
        const progressPhaseText = data.progress_phase ? String(data.progress_phase) : "";
        const promptMeta = data.prompt_meta && typeof data.prompt_meta === "object" ? data.prompt_meta : null;
        const lines = (data.log || []).map(t => String(t));
        const parsed = parsePromptOptions(promptText);
        const wantsDoneButton = hasDoneCue(promptText, lines);
        let options = [];
        const metaOptions = normalizeMetaOptions(promptMeta ? promptMeta.options : null);
        if (metaOptions.length) {{
          options = metaOptions;
        }} else if (isYesNoPrompt(promptText)) {{
          options = buildYesNoOptions();
        }} else {{
          options = parseRecentLogNumericOptions(lines);
        }}
        if (!metaOptions.length && wantsDoneButton && !isYesNoPrompt(promptText)) {{
          options = withDoneOption(options);
        }}
        const optionSignature = options.map(opt => `${{opt.value}}:${{opt.label}}`).join("|");
        const promptPendingChanged = lastPromptPending !== promptPending;
        const promptDisplay = promptPending
          ? (parsed.bodyText || (parsed.options.length ? "Choose an option below, or type a response." : promptText || "Waiting for prompt..."))
          : "Waiting for prompt...";

        statusEl.textContent = `Backend: ${{backendActive ? "active" : "stopped"}} | Session: ${{sessionActive ? "active" : "inactive"}} | Prompt: ${{promptPending ? "waiting for input" : "idle"}}`;
        if (promptDisplay !== lastPromptDisplay) {{
          promptEl.textContent = promptDisplay;
          resizePromptBox(promptDisplay);
          lastPromptDisplay = promptDisplay;
        }}
        if (promptPending !== !submitBtn.disabled || lastPrompt !== promptText || promptPendingChanged) {{
          answer.disabled = !promptPending;
          submitBtn.disabled = !promptPending;
          if (promptPending) {{
            answer.focus();
          }} else {{
            answer.value = "";
          }}
          lastPrompt = promptText;
          lastPromptPending = promptPending;
        }}

        if (optionSignature !== lastOptionSignature || promptPendingChanged) {{
          renderOptionButtons(options, promptPending);
          lastOptionSignature = promptPending ? optionSignature : "";
        }}
        renderProgress(progressFraction, progressTick, progressMax, progressPhaseText);

        const cap = Number(data.render_log_limit || 800);
        const visible = lines.length > cap ? lines.slice(lines.length - cap) : lines;
        if (lines.length !== lastCount) {{
          logEl.innerHTML = visible.map(t => `<div class="line">${{escapeHtml(t)}}</div>`).join("");
          logEl.scrollTop = logEl.scrollHeight;
          lastCount = lines.length;
        }}
      }} catch (_err) {{
        if (seenBackend) {{
          statusEl.textContent = "Backend: reconnecting | Session: waiting | Prompt: idle";
          promptEl.textContent = "Waiting to hear from python backend...";
          resizePromptBox(promptEl.textContent);
          renderOptionButtons([], false);
          lastOptionSignature = "";
          answer.disabled = true;
          submitBtn.disabled = true;
          answer.value = "";
        }}
      }}
    }}

    function escapeHtml(text) {{
      return text
        .replaceAll("&", "&amp;")
        .replaceAll("<", "&lt;")
        .replaceAll(">", "&gt;");
    }}

    form.addEventListener("submit", async (ev) => {{
      ev.preventDefault();
      submitAnswer(answer.value);
    }});

    function notifyDisconnect(reason) {{
      try {{
        const payload = new URLSearchParams();
        payload.set("reason", String(reason || "window_closed"));
        navigator.sendBeacon("/disconnect", payload);
      }} catch (_err) {{
        // best effort only
      }}
    }}

    window.addEventListener("pagehide", () => {{
      notifyDisconnect("pagehide");
    }});

    setInterval(loadState, 300);
    loadState();
    answer.disabled = true;
    submitBtn.disabled = true;
  </script>
</body>
</html>
"""


if __name__ == "__main__":
    sess = start_server(open_browser=True)
    print(f"Frontend running at {sess.base_url}")
    try:
        while True:
            time.sleep(1.0)
    except KeyboardInterrupt:
        stop_server(sess)
