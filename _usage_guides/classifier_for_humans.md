# Piloting the classifier: the short version

The classifier looks at stain images plus cell masks and calls every cell
IMMUNE, EPI, UNCLASSIFIED or ARTIFACT. You give it images, masks and (for
training) labels; it gives back a table of calls and pictures to check them.

The GUI (`python gui.py`) and the terminal (`python cli.py`) show the same menus
and ask the same questions. Everything below works in either.

## 1. Choose a project folder

A project folder is where your recipes, results and remembered settings live.
Pick one before anything else: **Startup → set project folder**.

| You are... | Project folder |
| --- | --- |
| Training or testing on the shared setup (Windows or Linux) | `\\rdsdcw.ohsu.edu\Coussens-secure\Multiplex_IHC_studies\Isaac_Youm\CLASSIFIER_PROJECT` on Windows, `/mnt/rdscoussens/Multiplex_IHC_studies/Isaac_Youm/CLASSIFIER_PROJECT` on Linux |
| Playing on the workstation that keeps data on `D:` | Any folder on `D:`, e.g. `D:\CLASSIFIER_PLAYGROUND`. Add a `project_config.txt` there containing `classifier_output_root=D:\CLASSIFIER_PLAYGROUND\runs` |
| Starting on another lab's data | A new folder next to their data, set up like the `D:` one |

On the shared setup, type the UNC path (`\\rdsdcw...`), not `Z:`. Linux cannot
read drive letters, and paths you type get saved into recipes.

## 2. Run it

**Startup → CNN cell classification**, then choose a mode:

- **predict**: run an existing model and score it. No training. Needs a model file (`best.pt`).
- **prepare**: load the inputs and draw previews, no training. A good first
  check on new data. It copies images, so it uses real disk space.
- **train_test**: train a new model, pick the best version on validation ROIs,
  score it on test ROIs.
- **tune_test**: the same, but starting from an existing model.

It then asks, in order:

1. **Where the labels are**: a CSV file, folder or glob, or a saved DAS triplet.
   Save any edits to the triplet first; it reads the saved files, not what is open.
2. **Which column holds the cell type** and **which column holds the mask ID**
   (the number each cell has in the mask image, usually `ObjectNumber`).
3. **What each label means**: map each label value to IMMUNE, EPI,
   UNCLASSIFIED or ARTIFACT, or ignore it. Ignored or blank cells are not used
   for training; they are never counted as UNCLASSIFIED.
4. **Which ROIs, and their role**: `train` (learn from), `val` (choose the best
   version), `test` (final score), `reserved` (leave alone). A whole ROI
   always stays in one role. With six or more ROIs it can split them for you.
5. **Where the images and masks are**, as a folder plus a filename pattern
   per marker, e.g. `{scene_id}/*CD45*.tif`. It shows the first match so you
   can see whether the pattern is right.
6. **Which markers go into each signal**: exactly one nuclear stain; one or
   more immune stains (e.g. CD45, CD3); one or more epithelial stains (e.g.
   PanCK, Ecad).
7. **Microns per pixel** of the images. Look this up; do not guess.
8. **Model file** (predict/tune only), **output folder**, and for training the
   number of epochs, batch size and learning rate.

## 3. Save it as a recipe

After a run you are happy with: **Recipes → save recording**. This writes the
answers as a JSON file in the project's `recipes/` folder. Rerunning the
recipe repeats the run exactly, with no questions. To run a recipe on the
Linux machine, from your DAS_v2 folder there:

```bash
python run_recipe.py --project /mnt/rdscoussens/Multiplex_IHC_studies/Isaac_Youm/CLASSIFIER_PROJECT --run RECIPE_NAME
```

Recipes can also be written by an agent and dropped into `recipes/`. Every
recipe run leaves a record in `recipe_runs/`.

## 4. Look at the results

Each run makes a new folder under the output folder. Open, in this order:

1. `debug_summary.txt`: what ran, on which data, the scores, and where every file is.
2. `debug/*__scaling.png`: raw versus scaled stains. If these look wrong,
   nothing downstream will be right.
3. The test (or validation) ROI images: truth, prediction, and errors side by side.
4. `test/errors*.png`: every wrongly called cell, most confident first. Many
   of these turn out to be label mistakes.
5. `predictions.csv`: one row per cell, with the call and a certainty score.

Scores measure agreement with your labels, not truth. If the labels came from
gating, a high score means "agrees with gating".

## 5. Another lab's data: what you need

- One single-plane TIFF per marker per ROI. Multichannel stacks are not
  read yet; they need a small adapter written first.
- One integer mask image per ROI, where each cell's pixels hold its ID.
- For training: a table with one row per cell, its mask ID, ROI name and cell type.
- A nuclear stain plus at least one immune and one epithelial stain.
- The real microns per pixel.

Run **prepare** first and look at the scaling images before training.

## 6. The training-data store

Curated public datasets are being collected in
`Multiplex_IHC_studies/Isaac_Youm/CLASSIFIER_DATA` (see its README). Once a
dataset there is converted, it has its own `manifest.csv`, which the
classifier reads through **Advanced: preset or JSON import**, or as a
`manifest` source in a recipe. None are converted yet.

## When it stops with an error

It stops on purpose rather than guessing. The usual causes:

- A filename pattern matches no file, or more than one, for some ROI.
- A label value has no mapping.
- A mask ID in the label table is not in the mask image.
- A stain is blank or constant.
- A path uses a drive letter on Linux.

The message names the ROI and file. Fix the input or the answer, then run again.
