"""Small shared path rules for mIHC ROI registration outputs."""

from __future__ import annotations

from pathlib import Path


def trim_mihc_roi_output_root(output_root: str | Path, slide_names) -> Path:
    """Return the batch root when a user points inside a known slide output.

    The ROI engine owns the final ``slide/ROI##`` layout.  Accepting either
    of those generated folders as its batch root would otherwise duplicate
    the slide name or ROI directory.
    """
    path = Path(output_root).expanduser()
    known_slides = {str(name).strip().casefold() for name in list(slide_names or []) if str(name).strip()}
    candidates = []
    for ancestor in [path, *path.parents]:
        if ancestor.name.casefold() in known_slides:
            candidates.append(ancestor.parent)
    # Parents are visited from inner to outer, so the last match removes any
    # accidentally selected slide/ROI portion.
    return candidates[-1] if candidates else path
