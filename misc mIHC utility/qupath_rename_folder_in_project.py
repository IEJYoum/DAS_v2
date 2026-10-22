"""
Patch a QuPath project after a folder containing its images was renamed.

Rewrites every file: URI under OLD_FOLDER so it points at the renamed folder,
in both project.qpproj (what QuPath actually reads) and data/*/server.json
(a mirror QuPath keeps per image). Only the "uri" string values are touched;
the rest of each file is left byte-for-byte as is.

Close the project in QuPath before running. If it stays open, QuPath will
write its in-memory (old) paths back over the patch on its next save.
"""
import os
import re
import shutil
from datetime import datetime
from urllib.parse import quote, unquote


PROJECT_DIR = r"Z:\Multiplex_IHC_studies\AP_mouse\2022_TLSworkup\3x Tumor Screen\3x_slides\WINDOWS_Qupath"
OLD_FOLDER = r"Z:\Multiplex_IHC_studies\AP_mouse\2022_TLSworkup\3x Tumor Screen\3x_slides\NR"  # full path, before rename
NEW_NAME = "NR_renamed"  # new name of that last folder only (same parent)
DRY_RUN = True  # report what would change; set False to write

URI_RE = re.compile(r'("uri"\s*:\s*")([^"]*)(")')


def path_parts(path):
    return [p for p in os.path.normpath(path).replace("\\", "/").split("/") if p]


def patch_uri(uri, old_parts, new_segment):
    """Return the patched URI, or None if it is not under OLD_FOLDER."""
    if not uri.startswith("file:/"):
        return None
    head, rest = "file:/", uri[len("file:/"):]
    segments = rest.split("/")
    n = len(old_parts)
    if len(segments) < n:
        return None
    decoded = [unquote(s) for s in segments[:n]]
    # Windows paths are case-insensitive
    if [s.casefold() for s in decoded] != [p.casefold() for p in old_parts]:
        return None
    segments[n - 1] = new_segment
    return head + "/".join(segments)


def patch_file(path, old_parts, new_segment, changes):
    with open(path, "r", encoding="utf-8") as f:
        text = f.read()

    def repl(m):
        new_uri = patch_uri(m.group(2), old_parts, new_segment)
        if new_uri is None:
            return m.group(0)
        changes.append((path, m.group(2), new_uri))
        return m.group(1) + new_uri + m.group(3)

    new_text = URI_RE.sub(repl, text)
    return new_text if new_text != text else None


def main():
    qpproj = os.path.join(PROJECT_DIR, "project.qpproj")
    if not os.path.isfile(qpproj):
        raise SystemExit(f"No project.qpproj in {PROJECT_DIR}")
    if any(c in NEW_NAME for c in "\\/"):
        raise SystemExit("NEW_NAME must be a single folder name, not a path")

    old_parts = path_parts(OLD_FOLDER)
    new_folder = os.path.join(os.path.dirname(os.path.normpath(OLD_FOLDER)), NEW_NAME)
    # Encode like Java's File.toURI (spaces -> %20); keep chars Java leaves alone
    new_segment = quote(NEW_NAME, safe="!$&'()*+,;=:@-._~")

    if not os.path.isdir(new_folder):
        print(f"WARNING: renamed folder does not exist (yet): {new_folder}")
    if os.path.isdir(OLD_FOLDER):
        print(f"WARNING: old folder still exists: {OLD_FOLDER}")

    targets = [qpproj]
    data_dir = os.path.join(PROJECT_DIR, "data")
    if os.path.isdir(data_dir):
        for entry in sorted(os.listdir(data_dir)):
            sj = os.path.join(data_dir, entry, "server.json")
            if os.path.isfile(sj):
                targets.append(sj)

    changes = []
    patched = {}
    for path in targets:
        new_text = patch_file(path, old_parts, new_segment, changes)
        if new_text is not None:
            patched[path] = new_text

    for path, old_uri, new_uri in changes:
        print(f"{os.path.relpath(path, PROJECT_DIR)}\n  {old_uri}\n  -> {new_uri}")

    missing = []
    for _, _, new_uri in changes:
        p = unquote(new_uri[len("file:/"):])
        if not p.lower().endswith(".qpproj") and not os.path.exists(p):
            missing.append(p)

    print(f"\n{len(changes)} URI(s) in {len(patched)} of {len(targets)} file(s) match {OLD_FOLDER}")
    if missing:
        print(f"{len(set(missing))} patched path(s) do not exist on disk, e.g. {missing[0]}")

    if DRY_RUN:
        print("DRY_RUN: nothing written")
        return
    if not patched:
        return

    backup_root = os.path.join(PROJECT_DIR, "uri_patch_backups", datetime.now().strftime("%Y%m%d_%H%M%S"))
    for path, new_text in patched.items():
        backup = os.path.join(backup_root, os.path.relpath(path, PROJECT_DIR))
        os.makedirs(os.path.dirname(backup), exist_ok=True)
        shutil.copy2(path, backup)
        with open(path, "w", encoding="utf-8", newline="") as f:
            f.write(new_text)
    print(f"Wrote {len(patched)} file(s); originals in {backup_root}")


if __name__ == "__main__":
    main()
