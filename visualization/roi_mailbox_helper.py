import csv
import http.server
import json
import os
import socketserver
import sys


ROI_MAILBOX_PREFIX = "ifa_roi_patch_"


def _is_native_path(path):
    """Reject foreign-OS paths so we never create phantom directory trees.

    On Windows: require a drive letter (C:\\, Z:/) or UNC (\\\\server\\share).
    On POSIX:   require an absolute POSIX path; reject drive letters and UNC.

    NOTE for future work — a better solution would translate paths between OSes
    using project_config.txt / project_config_posix.txt so a Linux-generated
    viewer works natively on Windows.  That risks fragility if the two configs
    point at different data versions.  Raise with the user before implementing.
    """
    p = str(path).strip()
    if not p:
        return False
    if os.name == "nt":
        # Windows: accept  Z:\...  Z:/...  \\server\share
        if len(p) >= 2 and p[1] == ":" and p[0].isalpha():
            return True
        if p.startswith("\\\\"):
            return True
        return False
    else:
        # POSIX: accept /... but reject drive letters and UNC
        if len(p) >= 2 and p[1] == ":" and p[0].isalpha():
            return False
        if p.startswith("\\\\"):
            return False
        return p.startswith("/")


def next_roi_mailbox_patch_path(mailbox_dir):
    mailbox_dir = os.path.abspath(os.path.normpath(str(mailbox_dir)))
    os.makedirs(mailbox_dir, exist_ok=True)
    existing = []
    try:
        for name in os.listdir(mailbox_dir):
            low = str(name).lower()
            if not low.endswith(".csv"):
                continue
            if not low.startswith("ifa_roi_patch"):
                continue
            existing.append(str(name))
    except Exception:
        existing = []
    next_n = 1
    i = 0
    while i < len(existing):
        stem = os.path.splitext(str(existing[i]))[0]
        if stem.startswith(ROI_MAILBOX_PREFIX):
            suffix = stem[len(ROI_MAILBOX_PREFIX):]
            try:
                next_n = max(next_n, int(suffix) + 1)
            except Exception:
                pass
        i += 1
    return os.path.join(mailbox_dir, f"{ROI_MAILBOX_PREFIX}{next_n:04d}.csv")


def read_threshold_table(path):
    """Return header and rows from a study thresholds CSV path."""
    if str(path).strip() == "":
        raise ValueError("path is required")
    if not os.path.isfile(path):
        return ["Markers"], []
    with open(path, "r", encoding="utf-8-sig", newline="") as f:
        rows = list(csv.reader(f))
    if len(rows) == 0:
        return ["Markers"], []
    header = list(rows[0])
    body = [list(row) for row in rows[1:]]
    if len(header) == 0:
        header = ["Markers"]
    return header, body


def threshold_marker_column(header):
    """Return the column index containing marker names in a threshold table header."""
    if not isinstance(header, list) or len(header) == 0:
        raise ValueError("header is empty")
    i = 0
    while i < len(header):
        if str(header[i]).strip().lower() == "markers":
            return i
        i += 1
    return 0


def write_threshold_table_atomic(path, header, rows):
    """Write a study thresholds table atomically and return the final path."""
    if str(path).strip() == "":
        raise ValueError("path is required")
    folder = os.path.dirname(os.path.abspath(path))
    if folder != "":
        os.makedirs(folder, exist_ok=True)
    temp_path = os.path.abspath(path) + ".tmp"
    with open(temp_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        writer.writerows(rows)
    os.replace(temp_path, path)
    return path


def update_threshold_table(path, thresholds):
    """Merge ROI->marker threshold values into a study thresholds table."""
    if str(path).strip() == "":
        raise ValueError("study_thresholds_path is required")
    if not isinstance(thresholds, dict) or len(thresholds) == 0:
        raise ValueError("thresholds must be a non-empty dict")

    header, rows = read_threshold_table(path)
    marker_col = threshold_marker_column(header)
    if len(header) == 0:
        header = ["Markers"]
    if marker_col == 0 and str(header[marker_col]).strip() == "":
        header[marker_col] = "Markers"
    while len(header) <= marker_col:
        header.append("")

    cleaned = {}
    for roi_id, marker_values in thresholds.items():
        roi_id = str(roi_id).strip()
        if roi_id == "" or not isinstance(marker_values, dict):
            continue
        for marker, threshold in marker_values.items():
            marker = str(marker).strip()
            if marker == "":
                continue
            try:
                value = float(threshold)
            except Exception:
                raise ValueError("threshold is not numeric for " + marker + ":" + roi_id)
            cleaned.setdefault(roi_id, {})[marker] = value
    if len(cleaned) == 0:
        raise ValueError("thresholds did not contain any numeric values")

    for roi_id in cleaned.keys():
        if roi_id not in header:
            header.append(roi_id)
    width = len(header)
    for row in rows:
        while len(row) < width:
            row.append("")

    row_by_marker = {}
    for row in rows:
        while len(row) < width:
            row.append("")
        key = str(row[marker_col]).strip().lower()
        if key != "" and key not in row_by_marker:
            row_by_marker[key] = row
    updated = []
    for roi_id, marker_values in cleaned.items():
        col_idx = header.index(roi_id)
        for marker, value in marker_values.items():
            key = str(marker).strip().lower()
            target_row = row_by_marker.get(key)
            if target_row is None:
                target_row = [""] * width
                target_row[marker_col] = str(marker).strip()
                rows.append(target_row)
                row_by_marker[key] = target_row
            while len(target_row) < width:
                target_row.append("")
            target_row[col_idx] = repr(value)
            updated.append(str(marker).strip() + ":" + roi_id)

    write_threshold_table_atomic(path, header, rows)
    return updated


class Handler(http.server.BaseHTTPRequestHandler):
    def _send_cors(self):
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")

    def do_OPTIONS(self):
        self.send_response(204)
        self._send_cors()
        self.end_headers()

    def do_GET(self):
        if self.path != "/health":
            self.send_response(404)
            self._send_cors()
            self.end_headers()
            return
        self.send_response(200)
        self._send_cors()
        self.send_header("Content-Type", "application/json")
        self.end_headers()
        self.wfile.write(b'{"ok":true,"service":"roi_mailbox_helper"}')

    def do_POST(self):
        if self.path not in ["/ifa_roi_patch", "/study_threshold"]:
            self.send_response(404)
            self._send_cors()
            self.end_headers()
            return
        try:
            n = int(self.headers.get("Content-Length", "0") or "0")
        except Exception:
            n = 0
        raw = self.rfile.read(n) if n > 0 else b""
        try:
            payload = json.loads(raw.decode("utf-8"))
        except Exception:
            self.send_response(400)
            self._send_cors()
            self.end_headers()
            return
        if self.path == "/study_threshold":
            self._handle_study_threshold(payload)
            return
        mailbox_dir = os.path.abspath(os.path.normpath(str(payload.get("mailbox_dir", "")).strip())) if str(payload.get("mailbox_dir", "")).strip() != "" else ""
        column = str(payload.get("column", "")).strip()
        assignments = list(payload.get("assignments", []))
        if mailbox_dir == "" or column == "" or len(assignments) == 0:
            self.send_response(400)
            self._send_cors()
            self.end_headers()
            return
        if not _is_native_path(mailbox_dir):
            self.send_response(400)
            self._send_cors()
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            body = json.dumps({"ok": False, "error": "mailbox_dir is a foreign-OS path"}).encode("utf-8")
            self.wfile.write(body)
            return
        try:
            final_path = next_roi_mailbox_patch_path(mailbox_dir)
            temp_path = final_path + ".tmp"
            with open(temp_path, "w", encoding="utf-8", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(["column", "index", "label"])
                i = 0
                while i < len(assignments):
                    row = assignments[i]
                    writer.writerow([
                        column,
                        str(row.get("index", "")),
                        str(row.get("label", "")),
                    ])
                    i += 1
            os.replace(temp_path, final_path)
            self.send_response(200)
            self._send_cors()
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            body = json.dumps({"ok": True, "service": "roi_mailbox_helper", "path": final_path}).encode("utf-8")
            self.wfile.write(body)
        except Exception:
            self.send_response(500)
            self._send_cors()
            self.end_headers()

    def _handle_study_threshold(self, payload):
        try:
            raw_path = str(payload.get("study_thresholds_path", "")).strip()
            path = os.path.abspath(os.path.normpath(raw_path)) if raw_path != "" else ""
            thresholds = payload.get("thresholds", {})
            if path == "" or not isinstance(thresholds, dict) or len(thresholds) == 0:
                self.send_response(400)
                self._send_cors()
                self.end_headers()
                return
            if not _is_native_path(raw_path):
                self.send_response(400)
                self._send_cors()
                self.send_header("Content-Type", "application/json")
                self.end_headers()
                body = json.dumps({"ok": False, "error": "study_thresholds_path is a foreign-OS path"}).encode("utf-8")
                self.wfile.write(body)
                return
            updated = update_threshold_table(path, thresholds)
            self.send_response(200)
            self._send_cors()
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            body = json.dumps({"ok": True, "service": "roi_mailbox_helper", "path": path, "updated_cells": updated}).encode("utf-8")
            self.wfile.write(body)
        except ValueError as exc:
            self.send_response(400)
            self._send_cors()
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            body = json.dumps({"ok": False, "error": str(exc)}).encode("utf-8")
            self.wfile.write(body)
        except Exception as exc:
            self.send_response(500)
            self._send_cors()
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            body = json.dumps({"ok": False, "error": str(exc)}).encode("utf-8")
            self.wfile.write(body)

    def log_message(self, format, *args):
        return


class ThreadingServer(socketserver.ThreadingMixIn, http.server.HTTPServer):
    daemon_threads = True


def main():
    port = 38765
    if len(sys.argv) > 1:
        try:
            port = int(sys.argv[1])
        except Exception:
            port = 38765
    server = ThreadingServer(("127.0.0.1", int(port)), Handler)
    try:
        server.serve_forever()
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
