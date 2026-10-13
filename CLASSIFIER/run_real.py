"""Small AH08 mechanics run. Labels are prepared upstream in one column."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from CLASSIFIER import prepare_real, run
from CLASSIFIER.config import CONFIG

OBS_CSV = r"Z:\Multiplex_IHC_studies\AlexGuimaraes\D10\Analysis\IY\DAS\subset_001\AH_08.1_obs - Copy.csv"
LABEL_COLUMN = "Celltype: CLASSIFIER"
IMAGE_ROOT = Path(r"Z:\Multiplex_IHC_studies\AlexGuimaraes\D10\Slides\ColorDecon")
DATA_DIR = Path(CONFIG["output_root"]) / "AH08_real_inputs" / "dataset"
SCENES = [{"scene_id": slide + "ROI01", "slide": slide, "split": split,
           "folder": str(IMAGE_ROOT / slide / "Processed" / "ROI01")}
          for slide, split in [("BTK153", "train"), ("BTK162", "val"), ("BTK172", "test")]]
PREP_KWARGS = {"um_per_px": 0.5, "tile_px": 256, "tiles_per_scene": 8,
               "seed": 7, "cache_images": True}
RUN_KWARGS = {"marker_groups": {"NUC": ["H2B"], "IMMUNE": ["CD45"], "EPI": ["PANCK"]},
              "steps": 100, "batch_size": 2, "debug_scenes": 1,
              "validate_every": 25, "target_um_per_px": PREP_KWARGS["um_per_px"]}


def main(**overrides):
    manifest = DATA_DIR / "manifest.csv"
    if not manifest.is_file():
        manifest = prepare_real.main(OBS_CSV, LABEL_COLUMN, SCENES, DATA_DIR, **PREP_KWARGS)
    kwargs = {**RUN_KWARGS, "manifest": str(manifest), **overrides}
    return run.main(**kwargs)


if __name__ == "__main__":
    main()
