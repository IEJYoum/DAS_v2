# Nimbus Marker Positivity

Nimbus is a separate pretrained image model. It estimates per-cell marker
scores; it does not replace DAS's CNN lineage classifier or directly assign
biological cell types.

## First Use

1. Install Nimbus-Inference in a separate Python environment.
2. Run `Prepare Data -> Nimbus marker positivity`.
3. Supply either a completed recipe feature-extraction `result.json`, or an
   explicit source JSON in the format below.
4. First choose `preview_only=y`. Read the generated `result.json` and verify
   scene names, selected marker files, mask role, labels, and shapes.
5. Rerun with `preview_only=n`, a valid `nimbus_python`, and a known image
   magnification. The output score table is additive; DAS does not change the
   current triplet yet.

## Explicit Source JSON

All image paths must be individual, single-channel 2D TIFFs. The first bridge
does not silently unpack OME-TIFF channels or copy/rename input images.

```json
{
  "scenes": [
    {
      "slide_scene": "Slide1_ROI01",
      "mask": "C:/project/Segmentation/Slide1_ROI01_cells.tif",
      "mask_role": "cell",
      "markers": {
        "CD3": "C:/project/registered/Slide1_ROI01_CD3.tif",
        "CD45": "C:/project/registered/Slide1_ROI01_CD45.tif"
      }
    }
  ]
}
```

Every selected scene must contain exactly the same selected marker names. Each
image must share the mask's pixel grid. `mask_role` is provenance, not a guess:
use `cell` only when the selected mask labels are the namespace you intend to
join later, normally DAS `obs.seg_label` paired with `obs.slide_scene`.

## Outputs

Each invocation creates `nimbus/nimbus_<stamp>/` with:

- `request.json`: resolved sources and requested settings.
- `worker.log`: Nimbus process output.
- `nimbus_scores_long.csv`: strict long-form scores with
  `slide_scene, mask_label, marker, score`.
- `result.json`: completed/failed status and provenance.

The bridge rejects missing, duplicate, extra, non-finite, or scene-mismatched
result rows. It never joins scores by dataframe row order or bare cell ID.

`checkpoint=latest` is convenient for a first run but can change when Nimbus
downloads weights. The worker records the resolved checkpoint path and SHA-256;
after validation, set the recorded checkpoint name explicitly for reproducible
runs.
