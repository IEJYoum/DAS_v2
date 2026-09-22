"""Standard terminal launcher for DAS_v2."""

from __future__ import annotations

import controler


def main() -> int:
    controler.main(use_html=False)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
