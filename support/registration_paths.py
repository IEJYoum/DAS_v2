"""Small shared path rules for mIHC ROI registration outputs."""

from __future__ import annotations

from pathlib import Path


REG_DAS = "Reg_DAS"


def trim_mihc_roi_output_root(output_root: str | Path, slide_names) -> Path:
    """Return the batch root when a user points inside ``slide/Reg_DAS``.

    The ROI engine owns the final ``slide/Reg_DAS/ROI##`` layout.  Accepting
    that final location as its batch root would otherwise duplicate it.
    """
    path = Path(output_root).expanduser()
    known_slides = {str(name).strip().casefold() for name in list(slide_names or []) if str(name).strip()}
    candidates = []
    for ancestor in [path, *path.parents]:
        if ancestor.name.casefold() != REG_DAS.casefold():
            continue
        if ancestor.parent.name.casefold() in known_slides:
            candidates.append(ancestor.parent.parent)
    # Parents are visited from inner to outer, so the last match removes any
    # already-duplicated slide/Reg_DAS portion as well.
    return candidates[-1] if candidates else path
