"""
Canonical launcher for New_DAS DS-backed runtime.
"""

from __future__ import annotations

import argparse
import os


def _configure_noninteractive_plot_backend() -> None:
    """
    Non-interactive launchers should save plots without opening blocking
    desktop windows.
    """
    os.environ.setdefault("MPLBACKEND", "Agg")
    try:
        import matplotlib

        matplotlib.use("Agg", force=True)
    except Exception:
        return


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Launch New_DAS in DS file-transport mode.",
    )
    parser.add_argument(
        "--data-folder",
        default=None,
        help="Optional project root / triplet folder override.",
    )
    parser.add_argument(
        "--figure-folder",
        default=None,
        help="Optional figure-folder override.",
    )
    parser.add_argument(
        "--build-folder",
        default=None,
        help="Optional legacy import/build source-folder override.",
    )
    parser.add_argument(
        "--stem",
        default=None,
        help="Optional startup stem override.",
    )
    parser.add_argument(
        "--session-root",
        default=None,
        help="Folder that will contain DS session subfolders. Defaults to app-data state.",
    )
    parser.add_argument(
        "--session-id",
        default=None,
        help="Optional DS session id override.",
    )
    parser.add_argument(
        "--poll-interval",
        type=float,
        default=0.25,
        help="Reply poll interval in seconds.",
    )
    return parser.parse_args()


def main() -> int:
    _configure_noninteractive_plot_backend()
    import controler

    args = parse_args()
    controler.load_legacy_ifa5()
    controler.main_ds(
        default_folder=args.data_folder,
        figure_folder=args.figure_folder,
        build_folder=args.build_folder,
        stem=args.stem,
        session_root=args.session_root,
        session_id=args.session_id,
        poll_interval_sec=float(args.poll_interval),
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
