"""Simple editable job file for standalone PhenoBIC runs.

Activate phenobic-clean, edit SOURCE_GLOB and RUN_NAME below, then run:

    python run_phenobic_kwargs.py

This calls run_phenobic_standalone.main() with keyword arguments.  It does not
modify a DAS triplet or the source ROI folders.
"""

from pathlib import Path
import runpy


# ---------------------------------------------------------------------------
# Edit these settings for a run.

MODE = "batch_masks"  # "batch_masks" for DAS ROI folders; "masks" for staged input.

# Batch mode: one ROI is valid, or use ROI* for an entire same-panel batch.
SOURCE_GLOB = r""  # e.g. r"Z:\Multiplex_IHC_studies\AlexGuimaraes\D10\Slides\ColorDecon\BTK153\Processed\ROI*"
RUN_NAME = "phenobic_batch"  # A new name for this staging/output run.

# Masks mode only: exact PhenoBIC input layout, not a raw DAS ROI folder.
MASKS_BASE_DIR = r""
CHANNELS = []  # e.g. ["CD45", "CD3"]; ignored by batch_masks (all markers are found).

WRITE_DEBUG = True
DEBUG_CHANNELS = []  # [] means every marker.
NUM_CELLS_BATCH = 4000
BUFFER_RATIO = 0.1
LOWER_PERCENTILE = 10.0
UPPER_PERCENTILE = 90.0


HERE = Path(__file__).resolve().parent
WORKSPACE_ROOT = HERE.parents[1]
PHENOBIC_ROOT = WORKSPACE_ROOT / "phenobic"
LAUNCHER_PATH = HERE / "run_phenobic_standalone.py"


def main():
    launcher = runpy.run_path(str(LAUNCHER_PATH))
    mode = str(MODE).strip().lower()
    if mode == "batch_masks":
        if str(SOURCE_GLOB).strip() == "":
            raise ValueError("Set SOURCE_GLOB before running this job.")
        run_root = PHENOBIC_ROOT / "temp_outputs" / RUN_NAME
        output = launcher["main"](
            mode="batch_masks",
            batch_masks_kwargs={
                "source_glob": SOURCE_GLOB,
                "staging_dir": PHENOBIC_ROOT / "temp_inputs" / RUN_NAME,
                "out_dir_path": run_root,
                "debug_dir": run_root / "debug",
                "write_debug": WRITE_DEBUG,
                "debug_channels": DEBUG_CHANNELS,
                "num_cells_batch": NUM_CELLS_BATCH,
                "buffer_ratio": BUFFER_RATIO,
                "lower_percentile": LOWER_PERCENTILE,
                "upper_percentile": UPPER_PERCENTILE,
            },
        )
    elif mode == "masks":
        if str(MASKS_BASE_DIR).strip() == "" or len(CHANNELS) == 0:
            raise ValueError("Set MASKS_BASE_DIR and CHANNELS for masks mode.")
        run_root = PHENOBIC_ROOT / "temp_outputs" / RUN_NAME
        output = launcher["main"](
            mode="masks",
            masks_kwargs={
                "base_dir": MASKS_BASE_DIR,
                "channels": CHANNELS,
                "out_dir_path": run_root,
                "debug_dir": run_root / "debug",
                "write_debug": WRITE_DEBUG,
                "debug_channels": DEBUG_CHANNELS,
                "num_cells_batch": NUM_CELLS_BATCH,
                "buffer_ratio": BUFFER_RATIO,
                "lower_percentile": LOWER_PERCENTILE,
                "upper_percentile": UPPER_PERCENTILE,
            },
        )
    else:
        raise ValueError("MODE must be 'batch_masks' or 'masks'.")
    print("[PhenoBIC] completed:", output)


if __name__ == "__main__":
    main()
