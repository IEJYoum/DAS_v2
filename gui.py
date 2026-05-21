"""
Canonical launcher for New_DAS browser-backed runtime.
"""

from __future__ import annotations

import argparse
import os


def _configure_noninteractive_plot_backend() -> None:
    """
    Browser-backed runs should save plots without opening blocking desktop
    windows.
    """
    os.environ.setdefault("MPLBACKEND", "Agg")
    try:
        import matplotlib

        matplotlib.use("Agg", force=True)
    except Exception:
        return


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Launch New_DAS in HTML GUI mode.",
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
        "--port",
        type=int,
        default=8765,
        help="Frontend server port.",
    )
    parser.add_argument(
        "--no-open-browser",
        action="store_true",
        help="Start server without opening a browser window.",
    )
    return parser.parse_args()


def main() -> int:
    _configure_noninteractive_plot_backend()
    import controler

    args = parse_args()
    controler.load_legacy_ifa5()
    controler.main_html(
        default_folder=args.data_folder,
        figure_folder=args.figure_folder,
        build_folder=args.build_folder,
        stem=args.stem,
        open_browser=not args.no_open_browser,
        html_port=int(args.port),
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
