# PinPoint Elevation — Run Book

This directory implements the elevation plan in
[`../PROJECT1_PINPOINT_ELEVATION.md`](../PROJECT1_PINPOINT_ELEVATION.md):
AUC/EER/AP/ECE metrics, the faithfulness + localization evaluation (the new core
contribution), true cross-dataset generalization, and the optional DINOv2
backbone ablation.

Everything is built on a single shared module, **`pinpoint_core.py`**, which
re-uses the exact PinPoint architecture (with the ResNet inplace-ops XAI patch)
plus a metadata-carrying dataset and a checkpoint loader that **auto-detects
`NUM_FRAMES`/`NUM_MFCC` from the saved weights** — so you never have to remember
whether the model was trained at 30 or 64 frames.

---

## 0. One-time setup — edit the PATHS block

Open **`pinpoint_core.py`** and edit the `PATHS (EDIT ME)` block in
`class CoreConfig` to point at your Kaggle inputs:

| Field | What it is | Example |
|---|---|---|
| `DATA_DIRECTORY` | merged preprocessed tensors dir | `/kaggle/input/new-model-unified-pre-processing/preprocessed_data` |
| `METADATA_PATH` | `unified_metadata.json` | `…/preprocessed_data/unified_metadata.json` |
| `MODEL_PATH` | trained checkpoint `best_pinpoint_model_antisocial.pth` | `/kaggle/input/<your-ckpt-dataset>/best_pinpoint_model_antisocial.pth` |
| `LAVDF_METADATA_PATH` | original LAV-DF `metadata.json` (for `fake_periods` + source tagging) | `/kaggle/input/localized-audio-visual-deepfake-dataset-lav-df/LAV-DF/metadata.json` |
| `OUTPUT_DIR` | where results are written | `/kaggle/working/elevation_outputs` |

> These are the **only** paths you need to edit. Every script reads them from
> `CoreConfig`. The scripts take CLI flags but no paths.

**Add this `elevation/` folder as a Kaggle "utility script" / dataset, or just
upload the files**, then `cd` into it before running so the local imports
(`import pinpoint_core`, `import metrics_utils`) resolve.

**Dependencies** (all already present on a standard Kaggle GPU image):
`torch`, `torchvision`, `numpy`, `scikit-learn`, `matplotlib`, `tqdm`.
The DINOv2 variant additionally needs internet (torch.hub) or a cached DINOv2
weights dataset.

---

## 1. Run order

You said the **checkpoint + merged data are available on Kaggle**, so you can run
the no-retraining scripts directly. Recommended order:

### Tier-0 — detection metrics (no GPU strictly needed, fast)
```bash
python eval_metrics.py
```
Outputs → `OUTPUT_DIR/metrics/`:
`metrics.json`, `roc_curve.png`, `pr_curve.png`, `reliability_diagram.png`,
and `test_predictions.npz` (raw probs/labels/source/manip-type for re-plotting).
Console prints AUC, AP, EER, Acc@0.5/@EER, macro-F1, Brier, ECE, **per-source
AUC** and **per-manipulation-type AUC**.

### Tier-1a — attention localization vs LAV-DF ground truth (1× T4)
```bash
python localization_eval.py            # all LAV-DF fake test clips with GT
# python localization_eval.py --max-clips 1000   # quick subset
```
Outputs → `OUTPUT_DIR/localization/localization.json`: mean per-clip frame AUC/AP,
pointing-game accuracy (+ random baseline), AP@tIoU{0.5,0.75}.

### Tier-1b — faithfulness (deletion/insertion) (1× T4)
```bash
python faithfulness_eval.py --n-clips 300 --ig-steps 20
```
Outputs → `OUTPUT_DIR/faithfulness/faithfulness.json`: deletion AUC + insertion
AUC for **attention vs integrated_gradients vs grad_cam vs random** on the same
model/frames. (Lower deletion, higher insertion = more faithful.)

### Tier-2 — cross-dataset generalization (P100, two runs)
```bash
python crossdataset_train.py --train-on lavdf       --test-on fakeavceleb --epochs 8
python crossdataset_train.py --train-on fakeavceleb --test-on lavdf       --epochs 8
```
Outputs → `OUTPUT_DIR/crossdataset_<a>_to_<b>/{model.pth,crossdataset.json}`:
in-domain dev AUC + **cross-dataset test AUC/AP/EER**.

### Optional — DINOv2-S backbone ablation (1× T4; needs torch.hub internet)
```bash
python dino_adapter_variant.py --epochs 8
```
Outputs → `OUTPUT_DIR/dino_adapter/{model.pth,dino_adapter.json}`: AUC/AP/EER of
the frozen-DINOv2 + adapter variant on the merged set, for the ResNet-vs-DINOv2
comparison.

> If you ever need to regenerate the checkpoint, run the original
> `../PinPoint.py` first; it saves `best_pinpoint_model_antisocial.pth`, which is
> what `MODEL_PATH` should point to.

---

## 2. How outputs map to the target results table (plan §2.5)

| Plan block | Produced by | Key fields |
|---|---|---|
| Main detection (AUC/AP/EER/ECE, per-category AUC) | `eval_metrics.py` | `metrics.json` → `auc/ap/eer/ece`, `per_source_auc`, `per_manip_auc` |
| **Localization** (tIoU, AP@tIoU, pointing-game) | `localization_eval.py` | `localization.json` |
| **Faithfulness** (deletion/insertion vs IG/Grad-CAM) | `faithfulness_eval.py` | `faithfulness.json` |
| Cross-dataset (AUC/EER FAVC↔LAVDF) | `crossdataset_train.py` ×2 | `crossdataset.json` |
| Backbone ablation (ResNet vs DINOv2) | `dino_adapter_variant.py` | `dino_adapter.json` (+ ResNet from `eval_metrics.py`) |
| Ablations re-emitted with AUC | `eval_metrics.py` with `MODEL_PATH` pointed at each ablation `.pth` | one `metrics/` run per checkpoint |

To **re-emit the no-sync / no-gate / no-curriculum ablations with AUC/EER**,
just point `CoreConfig.MODEL_PATH` at each ablation checkpoint and re-run
`eval_metrics.py` (use a separate `OUTPUT_DIR` per run, or move the `metrics/`
folder between runs). The ablation training scripts already live in the repo
root (`no loss.py`, `no gate - model.py`, `no curr-model.py`).

---

## 3. Assumptions & robustness notes (read if a script returns 0 samples)

- **Source tagging** (`lavdf` vs `fakeavceleb`) uses, in order: an explicit
  `source`/`dataset` field in the unified metadata → membership in the original
  LAV-DF `metadata.json` → a filename heuristic (LAV-DF basenames are purely
  numeric). If your merged metadata already carries a source field, it wins.
  If cross-dataset splits come back empty, your metadata isn't separable by
  source and you'll need to add a `source` field in preprocessing.
- **Localization needs `fake_periods`** (per-segment seconds) and a **duration**
  per clip. These come from the unified metadata if preprocessing kept them
  (it copies the original LAV-DF fields), otherwise from the
  `LAVDF_METADATA_PATH` join by filename. Clips without usable GT are skipped and
  counted in `skipped_no_duration`. FakeAVCeleb (whole-clip fakes, no segments)
  is intentionally excluded from localization.
- **Frame↔time mapping**: preprocessing samples `NUM_FRAMES` evenly across the
  whole clip, so video frame `v` sits at `t_v = v/(NUM_FRAMES-1) · duration`.
  LAV-DF is treated as **25 fps** (`CoreConfig.LAVDF_FPS`) when duration must be
  derived from a frame count.
- **The suspicion signal** measures attention falling *outside* the synchronized
  diagonal band the model was trained on (`center=i/MFCC_FRAMES_PER_VIDEO_FRAME`,
  half-width `SYNC_LOSS_BANDWIDTH`) — i.e. it reads the very mechanism the sync
  loss supervised, which is what makes the localization claim "faithful by
  construction".
- **`NUM_FRAMES`** for the eval scripts is auto-detected from the checkpoint; the
  training scripts (`crossdataset_train.py`, `dino_adapter_variant.py`) fix it to
  30 (the PinPoint default) when training from scratch.
