"""Run one adapter on one bundle in the adapter's own conda env and record provenance."""
from datetime import datetime
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

from harness.paths import ADAPTERS, BUNDLES, HARNESS_ROOT, METHODS, RUNS


def conda_exe():
    for candidate in (os.environ.get("CONDA_EXE"), shutil.which("conda"),
                      r"C:\ProgramData\anaconda3\Scripts\conda.exe"):
        if candidate and Path(candidate).exists():
            return candidate
    raise RuntimeError("conda not found; set CONDA_EXE")


def git_commit(folder):
    try:
        return subprocess.run(["git", "-C", str(folder), "rev-parse", "HEAD"], capture_output=True,
                              text=True, timeout=30).stdout.strip() or None
    except (OSError, subprocess.SubprocessError):
        return None


def run(label, adapter, bundle_id, config=None):
    config = config or {}
    spec = json.loads((ADAPTERS / adapter / "adapter.json").read_text(encoding="utf-8"))
    bundle_meta = (BUNDLES / bundle_id / "bundle.json").read_bytes()
    out = RUNS / f"{label}__{bundle_id}__{datetime.now():%Y%m%d_%H%M%S}"
    out.mkdir(parents=True)  # runs_root from harness_paths*.txt (the share)
    config_text = json.dumps(config, sort_keys=True)
    record = {"label": label, "adapter": adapter, "env": spec["env"], "bundle": bundle_id,
              "bundle_json_sha256": hashlib.sha256(bundle_meta).hexdigest(),
              "config": config, "config_sha256": hashlib.sha256(config_text.encode()).hexdigest(),
              "harness_commit": git_commit(HARNESS_ROOT),
              "upstream": spec.get("upstream"), "host": os.environ.get("COMPUTERNAME") or os.uname().nodename,
              "started": datetime.now().isoformat(timespec="seconds"), "status": "running"}
    if spec.get("upstream"):
        record["upstream_commit"] = git_commit(METHODS / spec["upstream"].split()[0])
    (out / "run.json").write_text(json.dumps(record, indent=2), encoding="utf-8")
    command = [conda_exe(), "run", "--no-capture-output", "-n", spec["env"], "python",
               str(ADAPTERS / adapter / "run.py"), "--bundle", bundle_id, "--out", str(out), "--config", config_text]
    print("running:", " ".join(command[:6]), "...", flush=True)
    with open(out / "log.txt", "w", encoding="utf-8") as log:
        env = {**os.environ, "PYTHONUNBUFFERED": "1"}  # live log.txt
        result = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, cwd=str(HARNESS_ROOT), env=env)
    record["finished"] = datetime.now().isoformat(timespec="seconds")
    record["status"] = "complete" if result.returncode == 0 and (out / "predictions.csv").is_file() else "failed"
    record["returncode"] = result.returncode
    (out / "run.json").write_text(json.dumps(record, indent=2), encoding="utf-8")
    print(record["status"], out, flush=True)
    if record["status"] != "complete":
        print((out / "log.txt").read_text(encoding="utf-8", errors="replace")[-3000:], file=sys.stderr)
    return out
