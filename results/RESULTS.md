# PinPoint — Consolidated Results (Source of Truth)

> **Purpose.** This folder is the single trusted record of every PinPoint number
> to cite when writing the paper draft. Anything not in this folder should be
> treated as scratch. Each result carries a **trust tag** and its **provenance**
> (which run / file produced it), because the raw numbers elsewhere can drift.
>
> Last consolidated: **2026-06-26**.

### Trust legend
| Tag | Meaning |
|---|---|
| ✅ **TRUSTED** | Cite directly. Reproducible from a saved run. |
| ⚠️ **INVALID** | Run completed but the experiment is degenerate/biased — **do not cite as a result**. Kept only to document the failure and why. |
| ⬜ **NOT RUN** | Planned, not yet executed (or crashed before finishing). |

### ⚠️ Two different test sets — do not conflate
- **Merged test set** (LAV-DF + FakeAVCeleb), **n = 26,097** (6,906 real / 19,191 fake). Used by the *original* full run and the three ablations.
- **LAV-DF-only test set**, **n = 1,550** (405 real / 1,145 fake). Used by the *elevation* (Tier-0) re-evaluation, the DINOv2 variant, localization, and faithfulness.

The original 97.47 % accuracy (merged) and the elevation AUC 0.9968 (LAV-DF only) are **not on the same data** and must not be presented as a before/after of the same number.

---

## 1. Main detection — original full run ✅ TRUSTED
*Merged test set, n = 26,097. Provenance: `../full-main-run-results.txt`.*

| Metric | Value |
|---|---|
| Accuracy | 0.9747 (25,436 / 26,097) |
| Macro F1 | 0.9680 |
| Fake F1 / Real F1 | 0.9825 / 0.9535 |
| Real Precision / Recall | 0.93 / 0.98 |
| Fake Precision / Recall | 0.99 / 0.97 |

Confusion matrix: Real→Real 6,767 · Real→Fake 139 · Fake→Real 522 · Fake→Fake 18,669.

> No AUC/EER/AP was computed on the merged set (thresholded metrics only). For
> threshold-free metrics use the elevation Tier-0 numbers in §3 (LAV-DF only).

---

## 2. Mechanism ablations ✅ TRUSTED
*Merged test set, n = 26,097. Provenance: `../no-sync-results.txt`, `../no-gate-results.txt`, `../no-curr-results.txt`.*

| Variant | Accuracy | Macro F1 | Fake F1 | Real F1 |
|---|---|---|---|---|
| **Full PinPoint** (ref, §1) | **0.9747** | **0.9680** | 0.9825 | 0.9535 |
| − Synchronization loss | 0.7547 | 0.7342 | 0.8067 | 0.6616 |
| − Gate | 0.5835 | 0.5813 | 0.6115 | 0.5510 |
| − Curriculum | 0.8144 | 0.7864 | 0.8639 | 0.7088 |

**Interpretation (safe to claim):** removing the gate (−39 pts) and the sync loss
(−22 pts) cause large drops → both components do substantial work for *detection*.
This is the strongest mechanistic evidence in the project.

---

## 3. Elevation Tier-0 — threshold-free detection metrics ✅ TRUSTED
*Same trained model as §1, re-evaluated on **LAV-DF-only** test, n = 1,550 (405 real / 1,145 fake). Provenance: elevation `eval_metrics.py` → `metrics.json` (Kaggle run).*

| Metric | Value |
|---|---|
| AUC | 0.9968 |
| AP | 0.9988 |
| EER | 0.0178 (threshold 0.8306) |
| Accuracy @0.5 / @EER | 0.9806 / 0.9819 |
| Macro F1 @0.5 / @EER | 0.9751 / 0.9769 |
| Fake F1 / Real F1 @0.5 | 0.9869 / 0.9632 |
| Brier | 0.0156 |
| **ECE** | **0.0139** (well calibrated) |

**Per-manipulation AUC** (each fake subset vs the same 405 real clips):

| Manipulation | AUC | AP | n_fake |
|---|---|---|---|
| audio_only | 0.9979 | 0.9973 | 383 |
| both | 0.9970 | 0.9971 | 382 |
| video_only | 0.9954 | 0.9954 | 380 |

> Near-ceiling and roughly equal across manipulation types, so this table does not
> by itself show *where* sync helps. Per-source: only `lavdf` present (= overall).

---

## 4. DINOv2-S backbone variant ✅ TRUSTED (but underperforms)
*Optional backbone ablation. Frozen DINOv2-S + adapter + existing fusion, trained 8 epochs. Same LAV-DF test set as §3 (n = 1,550). Provenance: elevation `dino_adapter_variant.py` → `dino_adapter.json`.*

| Metric | Value |
|---|---|
| Backbone | dinov2_vits14 @ 224px |
| Epochs / wall-clock | 8 / 136.1 min |
| Best val AUC | 0.8991 |
| Test AUC | 0.9101 |
| Test AP | 0.9609 |
| Test EER | 0.1615 |
| Test Acc @0.5 | 0.8581 |

**Interpretation:** directly comparable to §3 on the same test set — the
frozen-DINOv2 adapter (0.910 AUC) is **well below** the ResNet model (0.997 AUC)
at 8 epochs. Report as an honest negative / "needs more training," not a win.

---

## 5. Tier-1 Localization — attention vs LAV-DF GT ⚠️ NEGATIVE RESULT (TRUSTED run, null finding)
*LAV-DF fake clips, n = 1,145 (0 skipped for missing duration). Provenance: elevation `localization_eval.py` → `localization.json`.*

| Metric | Value | Baseline / chance |
|---|---|---|
| Mean per-clip frame AUC | 0.5348 | 0.5 (chance) |
| Mean per-clip frame AP | 0.2034 | ≈ fake-frame base rate |
| Pointing-game accuracy | 0.0847 | random **0.0893** |
| AP @tIoU=0.5 | 0.0029 | ≈ 0 |
| AP @tIoU=0.75 | 0.0002 | ≈ 0 |

**Interpretation (honest):** the audio→video attention does **not** temporally
localize the manipulated frames — frame AUC is at chance and pointing-game is at/below
random. The model detects clips near-perfectly (§3) without exposing a localizable
explanation. This is a genuine negative result. *Caveat:* pointing-game *below*
random hints at a possible GT-alignment/off-by-one issue that was never re-verified
(see §6 of the plan); treat the magnitude as approximate, but the "no localization"
conclusion stands.

---

## 6. Tier-1 Faithfulness (deletion/insertion) ⚠️ INVALID — DO NOT CITE
*n = 300 clips. Provenance: elevation `faithfulness_eval.py` **v1** → `faithfulness.json`.*

| Method | Deletion AUC | Insertion AUC | ins − del |
|---|---|---|---|
| attention | 0.9902 | 0.9902 | −0.0000 |
| integrated_gradients | 0.9880 | 0.9918 | +0.0038 |
| grad_cam | 0.9908 | 0.9888 | −0.0020 |
| random | 0.9900 | 0.9906 | +0.0006 |

**Why this is INVALID:** every method (incl. `random`) sits at ~0.99 for both
curves → the test measured nothing. Two causes: (a) the classifier is
audio-centric (`pooled = audio_feat.mean(1)`), but v1 perturbed only **video**;
(b) the all-**zero** baseline is out-of-distribution but still reads "fake", so
`p_fake` never moved. **No faithfulness conclusion can be drawn from this run.**

**Status of the fix:** `faithfulness_eval.py` was rewritten (v2: in-distribution
mean baseline + modality-aware video/audio tracks + a baseline-saturation sanity
check). The v2 run **⬜ has NOT completed** (Kaggle session crashed). Until v2
produces a table with `mean p_fake @ fully-baselined < 0.5`, there is **no trusted
faithfulness result**.

---

## 7. Earlier exploratory XAI ⚠️ WEAK — use as descriptor only
*Provenance: paper draft dump (`paper/v2/TECHNICAL_DUMP.md`), earlier run.*

- Attribution sparsity (Gini): video **0.631**, audio **0.928**.
- Inter-technique consistency (Spearman): **0.514**.
- TCAV lip-sync concept scores (main run): video_extractor.projection 0.975,
  audio_extractor.gru 1.000, gated layer-0 0.500, gated layer-1 1.000,
  classification_head 1.000.

> Gini is only a sparsity descriptor (not evidence of correctness). The 0.514
> consistency is weak and should **not** be used as a faithfulness claim — that was
> exactly the gap the Tier-1 work was meant to close.

---

## 8. Not yet run ⬜
| Result | Script | Status |
|---|---|---|
| Faithfulness v2 (valid deletion/insertion) | `faithfulness_eval.py` (fixed) | crashed mid-run; rerun pending |
| Cross-dataset FAVC→LAVDF / LAVDF→FAVC | `crossdataset_train.py` | not run |
| Ablations re-emitted with AUC/EER | `eval_metrics.py` per ablation ckpt | not run |

---

## 9. One-paragraph summary for the paper

PinPoint is a strong **in-domain detector with good calibration**: AUC **0.9968** /
EER **0.0178** / ECE **0.0139** on LAV-DF (§3), 97.47 % accuracy on the merged
LAV-DF+FakeAVCeleb set (§1), and large ablation drops (−gate −39 pts, −sync −22 pts,
§2) that establish the gate and synchronization loss as load-bearing. The originally
intended headline contribution — a **faithful, localizable** attention explanation —
is **not supported by current evidence**: attention does not localize manipulated
frames (§5, at chance) and the faithfulness test is invalid pending a rerun (§6).
Cross-dataset generalization (§8) is not yet measured. **Recommended framing:**
lead with detection + calibration + the mechanism ablations; report localization as
an honest negative; do not claim faithfulness until the v2 run lands.
