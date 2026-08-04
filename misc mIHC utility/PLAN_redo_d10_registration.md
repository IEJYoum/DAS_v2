# Codex Plan: redo_d10_registration.py

Create a new script `redo_d10_registration.py` in `DAS_v2/misc mIHC utility/` that
automatically re-registers failed slides from the D10 dataset using the
translation engine from `realign_mihc_test.py`, producing output compatible
with the existing legacy pipeline.

## Context

The legacy pipeline sometimes fails to register specific marker channels.
These failures end up in `Redo_ROI*` folders with a `nonreg_` prefix.
A MATLAB manual-registration script exists but is slow and interactive.
This script replaces that workflow with automated translation-only registration.

### Folder layout (read-only inputs)

```
Z:\...\D10\Slides\Run\
├── BTK162\
│   ├── Redo_ROI01\
│   │   ├── NUCLEI_KB_AG_KPC_BTK162_D10_C06R2_HEM_ROI01.tif   (fixed)
│   │   └── nonreg_KB_AG_KPC_BTK162_D10_C03R2_CD3_ROI01.tif   (moving)
│   ├── Redo_ROI02\
│   │   ├── NUCLEI_...HEM_ROI02.tif
│   │   └── nonreg_...CD206_ROI02.tif  (etc., 1-5 nonreg per folder)
│   └── Registered_Regions\
│       ├── ROI01\
│       │   ├── NUCLEI_...HEM_ROI01.tif          (cropped to final size)
│       │   ├── reg_KB_AG_KPC_..._HE_ROI01.tif   (H&E reference)
│       │   ├── reg_KB_AG_KPC_..._CSF1R_ROI01.tif (other successfully-registered markers)
│       │   └── reg_nonreg_KB_AG_KPC_..._CD3_ROI01.tif  (manually registered — skip)
│       └── ROI02\ ...
├── GEM702\ ...
├── SAL392\ ...
├── SAL452\ ...
├── SAL692\ ...
└── Registration_Check\          (excluded from crawl)
    └── Reg_IY\                  (our output root — created by this script)
```

### Image characteristics (verified)

- Redo inputs: RGB uint8, 3 channels, **JPEG-compressed, tiled** TIFF
  (requires imagecodecs package for reading)
- Registered_Regions outputs: RGB uint8, 3 channels, **PACKBITS-compressed,
  strip-based** TIFF
- Redo images are LARGER than Registered_Regions images
  (e.g., 5984×5983 vs 3983×3983)
- Both files in a Redo folder are always the same shape
- The Registered_Regions versions are axis-aligned crops of the Redo versions
  (determined by XML annotations with a 1000-pixel buffer in the original
  pipeline; however the Redo TIFFs have NO metadata recording their SVS
  origin offset, so the XML coordinates cannot be mapped directly to Redo
  TIFF coordinates — crop recovery must use pixel matching instead)

### Output structure

```
Registration_Check\Reg_IY\
└── BTK162\
    └── Registered_Regions\
        └── ROI02\
            ├── reg_nonreg_KB_AG_KPC_BTK162_D10_C01R1_CD206_ROI02.tif
            ├── reg_nonreg_KB_AG_KPC_BTK162_D10_C01R2_MHCII_ROI02.tif
            ├── reg_nonreg_KB_AG_KPC_BTK162_D10_C03R1_F480_ROI02.tif
            ├── reg_KB_AG_KPC_BTK162_D10_C00R1_HE_ROI02.tif   (copied from original)
            └── redo_debug.txt
```

---

## Implementation spec

### 1. Discovery (crawl + skip logic)

```python
RUN_ROOT = Path(r"Z:\Multiplex_IHC_studies\AlexGuimaraes\D10\Slides\Run")
OUTPUT_ROOT = RUN_ROOT / "Registration_Check" / "Reg_IY"
```

Add `--run-root` and `--output-root` CLI args (argparse) so paths are not
hardcoded. Also add `--dry-run` flag that prints the manifest without
registering.

**Crawl logic:**

```
for each slide_dir in RUN_ROOT (skip "Registration_Check"):
    for each redo_dir matching Redo_*:
        roi = redo_dir.name[len("Redo_"):]          # e.g., "ROI01"
        nuclei = the one file matching NUCLEI_*.tif  # fixed reference
        nonreg_files = all files matching nonreg_*.tif

        for each nonreg file:
            # Skip if already registered in original location
            check {slide}/Registered_Regions/{roi}/ for:
              - reg_{nonreg_stem}.tif   (preserving original case)
              - reg_NONREG_{body}.tif   (legacy uppercase variant)
            # Skip if already registered in our output
            check OUTPUT_ROOT/{slide}/Registered_Regions/{roi}/ for same

            if not skipped: add to work list
```

Where `nonreg_stem` = the nonreg filename without extension (e.g.,
`nonreg_KB_AG_KPC_BTK162_D10_C03R2_CD3_ROI01`) and `body` = stem with
`nonreg_` prefix stripped.

**Verified inventory:**
- 29 total nonreg files across 6 slides, 13 ROIs
- 1 already manually registered (BTK162/ROI01 CD3)
- 28 to register

### 2. Crop coordinate recovery

The Redo TIFFs are pre-extracted crops from the SVS. They contain NO metadata
recording their SVS origin offset. The XML annotation files exist but their
vertex coordinates are in SVS-global pixel space, which cannot be mapped to
Redo TIFF coordinates without knowing the extraction offset. Therefore, use
FFT cross-correlation to find where the (smaller) Registered_Regions NUCLEI
appears within the (larger) Redo NUCLEI.

Validated on BTK162/ROI01: recovered offset (999, 999), consistent with the
MATLAB 1000px buffer. Pixel values differ slightly due to JPEG vs PACKBITS
recompression but the geometric match is correct.

For each ROI that has work to do:

1. Read the Redo NUCLEI (full size, e.g., 5984×5983)
2. Read the Registered_Regions NUCLEI (cropped, e.g., 3983×3983)
3. Find the crop offset using FFT cross-correlation on one channel (red):

```python
from scipy.signal import fftconvolve

def find_crop_offset(redo_channel, cropped_channel):
    """Return (row_offset, col_offset) where cropped appears in redo.

    Uses a center patch from the cropped image as template to avoid
    edge artifacts. Operates on a single grayscale channel.
    """
    patch_size = min(256, cropped_channel.shape[0] // 4,
                     cropped_channel.shape[1] // 4)
    cy = cropped_channel.shape[0] // 2
    cx = cropped_channel.shape[1] // 2
    patch = cropped_channel[cy:cy+patch_size, cx:cx+patch_size].astype(np.float32)
    patch -= patch.mean()
    redo_f = redo_channel.astype(np.float32)
    cc = fftconvolve(redo_f, patch[::-1, ::-1], mode='valid')
    peak = np.unravel_index(np.argmax(cc), cc.shape)
    row_offset = peak[0] - cy
    col_offset = peak[1] - cx
    return int(row_offset), int(col_offset)
```

4. **Sanity check**: verify `row_offset >= 0`, `col_offset >= 0`, and that
   `row_offset + crop_h <= redo_h` and `col_offset + crop_w <= redo_w`.
   If not, warn and skip the ROI.

5. Store `(row_offset, col_offset, crop_h, crop_w)` — reuse for all nonreg
   files in the same ROI (same crop applies to all markers).

### 3. Registration engine (translation-only)

Import from `realign_mihc_test`:

```python
import realign_mihc_test
```

Functions needed:
- `realign_mihc_test.rgb_to_k_channel(image)` → 2D grayscale
- `realign_mihc_test.fit_translation_scaled(fixed, moving, image_scale,
  warning_dir=..., context=...)` → `(dy, dx)`
- `realign_mihc_test.shift_image(image, dy, dx, fill_value, out_shape)` → shifted

**Important**: Before calling the engine, explicitly snapshot and restore
the engine globals that affect translation behavior, to isolate from any
future changes to `realign_mihc_test.py` defaults:

```python
# Freeze engine settings for D10 redo runs
_SAVED_FIT_SCALES = list(realign_mihc_test.FIT_SCALES)
_SAVED_SEARCH_RADIUS = realign_mihc_test.INITIAL_SEARCH_RADIUS_FULL_PIXELS

def register_pair(fixed_k, moving_k, warning_dir, context):
    """Find translation (dy, dx) using the realign_mihc_test engine."""
    realign_mihc_test.FIT_SCALES[:] = _SAVED_FIT_SCALES
    realign_mihc_test.INITIAL_SEARCH_RADIUS_FULL_PIXELS = _SAVED_SEARCH_RADIUS
    return realign_mihc_test.fit_translation_scaled(
        fixed_k, moving_k, 1.0,
        warning_dir=warning_dir, context=context,
    )
```

For each nonreg file:

1. Read redo NUCLEI as RGB (cache per ROI — already loaded for crop recovery)
2. Read nonreg file as RGB
3. Extract K-channel from both:
   ```python
   fixed_k = realign_mihc_test.rgb_to_k_channel(redo_nuclei)
   moving_k = realign_mihc_test.rgb_to_k_channel(nonreg_rgb)
   ```
4. Find translation:
   ```python
   dy, dx = register_pair(fixed_k, moving_k,
                           warning_dir=out_dir, context=nonreg_stem)
   ```
5. Apply shift to each RGB channel independently, with white fill (255)
   and output shape matching the redo NUCLEI:
   ```python
   shifted = np.empty_like(nonreg_rgb)
   for ch in range(3):
       shifted[:, :, ch] = realign_mihc_test.shift_image(
           nonreg_rgb[:, :, ch], dy, dx, fill_value=255,
           out_shape=redo_nuclei.shape[:2]
       )
   ```
6. Crop to final dimensions:
   ```python
   r, c, h, w = crop_row, crop_col, crop_h, crop_w
   output_rgb = shifted[r:r+h, c:c+w, :]
   ```

### 4. Save output

**Output path:**
```python
out_dir = OUTPUT_ROOT / slide_name / "Registered_Regions" / roi_name
out_dir.mkdir(parents=True, exist_ok=True)
out_path = out_dir / ("reg_" + nonreg_stem + ".tif")
```

Note: `nonreg_stem` preserves the original `nonreg_` prefix and original
case (lowercase), so the output filename is `reg_nonreg_KB_AG_KPC_...tif`.

**TIFF format** — match existing registered output files exactly:
```python
import tifffile
tifffile.imwrite(
    str(out_path),
    output_rgb,
    photometric='rgb',
    compression='packbits',
)
```

- Photometric: RGB
- Compression: PACKBITS
- dtype: uint8
- Strip-based (default, not tiled)

**Copy H&E for overlay checks:**

After registering all nonreg files for a given ROI, check if
`{slide}/Registered_Regions/{roi}/reg_*_HE_*.tif` exists. If so, copy it
to the output directory (shutil.copy2). This is for running the overlay
visualization script on the Reg_IY folder. All 13 ROIs have HE files
(verified), so this should always succeed.

Do NOT copy the NUCLEI/HEM file. It already exists in the original
`Registered_Regions/` folder and does not need to be duplicated.

### 5. Debug output

Write `redo_debug.txt` per ROI in the output Registered_Regions/{ROI}/ folder:

```
slide       BTK162
roi         ROI02
crop_offset row=999 col=999
crop_size   3983x3983
timestamp   2026-08-04T10:30:00-07:00

file                                                              dy    dx    status
nonreg_KB_AG_KPC_BTK162_D10_C01R1_CD206_ROI02.tif               -12    45    ok
nonreg_KB_AG_KPC_BTK162_D10_C01R2_MHCII_ROI02.tif                 3   -21    ok
nonreg_KB_AG_KPC_BTK162_D10_C03R1_F480_ROI02.tif                  8     7    ok
```

### 6. Error handling

- If `fit_translation_scaled` raises (e.g., zero-variance correlation), catch
  the exception, print a warning, write `status=FAILED` + exception message
  in the debug txt, and continue to the next file. Do NOT abort the run.
- If crop offset recovery fails (offset out of bounds or negative), print a
  warning and skip the entire ROI.
- Use the existing `_retry_io` pattern from `realign_mihc_test` for network
  drive reads/writes (CIFS EIO transient errors). Import `_retry_io`,
  `_TRANSIENT_ERRNOS`, `IO_RETRY_COUNT`, `IO_RETRY_WAIT_SECONDS`.

### 7. CLI interface

```
python redo_d10_registration.py [--run-root PATH] [--output-root PATH] [--dry-run]
```

- `--run-root`: defaults to the D10 RUN_ROOT constant
- `--output-root`: defaults to RUN_ROOT / "Registration_Check" / "Reg_IY"
- `--dry-run`: print manifest (what would be registered, what skipped, crop
  coords per ROI) without actually registering or writing files

---

## What this script does NOT do

- No affine/rotation/scale — translation only
- No single-channel OME-TIFF output
- No multichannel OME-TIFF output
- No overlay PNGs (user will run overlay script separately)
- No K-channel debug images
- Does NOT modify, move, or delete any original files
- Does NOT write to Registered_Regions — only to Registration_Check/Reg_IY

## Files to read (for reference, do not modify)

- `realign_mihc_test.py`: import `rgb_to_k_channel`, `fit_translation_scaled`,
  `shift_image`, `_retry_io`, `_TRANSIENT_ERRNOS`, `IO_RETRY_COUNT`,
  `IO_RETRY_WAIT_SECONDS`, `FIT_SCALES`, `INITIAL_SEARCH_RADIUS_FULL_PIXELS`
- `triage_d10_registration.py`: reference only for folder structure understanding

## Files to create

- `redo_d10_registration.py` — the new script (single file, self-contained
  except for realign_mihc_test imports)

## Suggested build order

1. Build dry-run/discovery first: manifest targets, skips, output paths,
   crop recovery per ROI. Validate that the manifest shows 28 to register,
   1 skipped.
2. Validate crop math on BTK162/ROI01: recovered offset should be ~(999,999),
   crop dimensions should match Registered_Regions NUCLEI (3983×3983 for
   BTK162/ROI01).
3. Add registration + write for one ROI (e.g., SAL692/ROI01 which has only
   1 nonreg file — fast to test).
4. Expand to full batch with per-file failure logging.

## Test checklist

1. `--dry-run` prints correct manifest: 28 files to register, 1 skipped
   (BTK162/ROI01 CD3)
2. BTK162/ROI01 is fully skipped (CD3 already manually registered)
3. Crop offset recovery produces sensible values (~999,999 for BTK162)
4. Output TIFF matches format: RGB uint8, PACKBITS, strip-based
5. Output dimensions match the Registered_Regions NUCLEI dimensions
6. Output path is Registration_Check/Reg_IY/{SLIDE}/Registered_Regions/{ROI}/
7. Output filename: reg_nonreg_{body}.tif (lowercase, original case preserved)
8. H&E file copied to output directory for each ROI
9. Original files are never modified or deleted
10. Network I/O errors (CIFS EIO) trigger retry, not crash
11. A failed registration writes FAILED status in debug txt and continues
