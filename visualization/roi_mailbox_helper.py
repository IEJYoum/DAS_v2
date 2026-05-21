import csv
import http.server
import json
import os
import socketserver
import sys


ROI_MAILBOX_PREFIX = "ifa_roi_patch_"


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
        if self.path != "/ifa_roi_patch":
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
        mailbox_dir = os.path.abspath(os.path.normpath(str(payload.get("mailbox_dir", "")).strip())) if str(payload.get("mailbox_dir", "")).strip() != "" else ""
        column = str(payload.get("column", "")).strip()
        assignments = list(payload.get("assignments", []))
        if mailbox_dir == "" or column == "" or len(assignments) == 0:
            self.send_response(400)
            self._send_cors()
            self.end_headers()
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
