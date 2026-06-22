# Project 1 — PinPoint Elevation Plan
### Objective 1: Audio–Visual Fusion + Explainable AI

> **Scope of this document.** Grounded assessment of the *current* PinPoint (code, results, paper draft `paper/v2/main.tex`), then the exact elevation: what to keep, what to add, what to run, what numbers to gather, and how it fits the thesis. All current numbers below are read directly from `full-main-run-results.txt`, `paper/v2/TECHNICAL_DUMP.md`, and the ablation result files — nothing invented.

---

## 1. Current situation (grounded)

**What PinPoint is.** A synchronization-aware audio–visual deepfake detector. ResNet-18 (visual, early layers frozen, in-place ops patched for XAI) + Conv1D→LayerNorm→GRU (audio, 13-dim MFCC) → 3 gated cross-attention blocks (8 heads, d=256; audio=query, video=key/value, sigmoid gate) → fake logit + 11-class offset head (−5…+5 frames) + the audio-to-video attention map. Trained with `L = 1.0·BCE + 0.5·L_offset + 3.0·L_sync`, where `L_sync = 1.0·MSE + 0.5·diag + 0.2·smooth`, over a 3-stage curriculum (mask video → sync focus → full fusion), 15 epochs, batch 4, AdamW 1e-4, P100.

**Data.** A *unified* preprocessed benchmark merging LAV-DF + FakeAVCeleb: 78,703 train / 31,501 val / 26,097 test (test: 6,906 real, 19,191 fake; ratio 0.36).

**Headline results (test, n=26,097).**
| Metric | Value |
|---|---|
| Accuracy | 97.47% (25,436/26,097) |
| Macro F1 | 0.9680 |
| Fake F1 / Real F1 | 0.9825 / 0.9535 |
| Real P/R | 0.93 / 0.98 |
| Fake P/R | 0.99 / 0.97 |
| Confusion | Real→Real 6767, Real→Fake 139; Fake→Real 522, Fake→Fake 18669 |

**Ablations (accuracy).** No-sync 75.47% · No-gate 58.35% · No-curriculum 81.44%. These are *strong* — they prove the gate and the sync loss are doing real work, not decoration.

**XAI evidence (current).** Seven post-hoc methods (IG, LRP, SHAP, LIME, Grad-CAM, attention rollout, counterfactuals) + TCAV. Quantitative side: video attribution Gini 0.631, audio Gini 0.928, inter-technique consistency Spearman **0.514**. TCAV lip-sync concept scores reported per layer (e.g., 0.975–1.000 at deeper layers).

**Paper draft.** `paper/v2/main.tex` — Elsevier `elsarticle`, title *"PinPoint: Synchronization-Aware and Explainable Audio–Visual Deepfake Detection."* Target venue: **Ain Shams Engineering Journal** (Q1 Elsevier). The draft already makes the right conceptual move ("make the final attention map part of the supervised target").

### Honest critique — what blocks a strong publication

1. **No AUC, no EER, no AP anywhere.** The whole evaluation is thresholded accuracy/F1 on a 1:2.8 imbalanced test set. Every competitive AV paper (AVFF: 99.1 AUC; CMALDD: 99.6 AUC) reports AUC. *Without AUC/EER, PinPoint cannot be compared to the field at all.* This is the single most important fix and costs almost nothing (you already have the logits).
2. **No true cross-dataset generalization.** Training and testing both happen on the *merged* LAV-DF+FakeAVCeleb pool. So the 97.47% is in-distribution. There is no "train on A, test on B" number — reviewers will read the merge as hiding generalization weakness.
3. **The XAI is the claimed contribution but it is not validated.** Showing seven heatmaps is 2020-era practice. The one quantitative number (consistency Spearman 0.514) is *weak and unmotivated* — 0.51 agreement between methods is not evidence of faithfulness; it could equally mean the methods disagree. There is **no faithfulness test** (deletion/insertion) and **no localization test** against ground truth, even though LAV-DF *ships temporal-localization labels* you are not using.
4. **Accuracy framing loses.** 97.47% accuracy / 0.968 AUC-equivalent is below SOTA. If the paper is pitched as "accurate AND explainable," the accuracy half invites rejection.

### Your framing vs. the stronger framing

- **Your stated contribution:** "gated attention + multiple XAI techniques to validate the responses."
- **The problem with it:** "we applied many XAI methods" is not a contribution — it is an experimental section. Reviewers ask *"so what — are the explanations correct?"* and the draft cannot answer.
- **The stronger story (recommended):** **PinPoint produces a faithful-by-construction explanation.** Because the synchronization loss *supervises the attention map itself*, the map is not a post-hoc guess — it is the mechanism the model was trained to use. The contribution becomes: *"we make the detector's explanation part of its objective, then prove that explanation is (a) faithful — perturbing the highlighted region changes the decision — and (b) correct — it localizes the actually-manipulated frames in LAV-DF."* The seven XAI methods then play a supporting role: they all corroborate the same synchronization evidence. This converts "we drew heatmaps" into "we are the first AV detector whose explanation is supervised and quantitatively validated against manipulation ground truth." That is a genuine, defensible contribution that does **not** depend on beating AVFF on accuracy.

Both framings keep your gated attention + sync loss. The difference is what you *claim* and *measure*.

---

## 2. The elevation — what to do

### 2.1 Tier-0 fixes (mandatory, cheap, no retraining)

These use the existing trained model + saved logits.

- **Add AUC, AP (average precision), and EER** on the test set, plus a ROC and PR curve. You have the logits; this is a 1-cell script.
- **Per-category breakdown.** FakeAVCeleb has RR/FR/RF/FF categories and LAV-DF has its own. Report AUC per manipulation type so the paper shows *where* sync helps (it should help most on RF/FF where lip-sync is broken, least on FR where only the face is swapped). This turns one number into an insight.
- **Calibration:** report ECE + a reliability diagram. Forensic tools need calibrated confidence; this is cheap and reviewers like it.

### 2.2 Tier-1: the faithfulness + localization evaluation (the new core contribution)

This is what makes the paper publishable above a mid journal.

- **Temporal localization against LAV-DF GT.** LAV-DF provides per-segment manipulation timestamps. For each fake LAV-DF clip, take the diagonal energy of the audio→video attention map as a per-frame "manipulation suspicion" signal and compute **temporal IoU / AP@tIoU** against the GT fake segments. Claim: PinPoint's *attention* localizes manipulated frames without ever being trained on localization labels (the sync loss only saw real samples). This is a strong, novel, quantitative result.
- **Faithfulness via deletion/insertion (Captum has these).** Rank video frames / audio steps by attention weight; delete top-k and measure the drop in fake-probability (deletion AUC), and insert top-k into a baseline and measure the rise (insertion AUC). Compare the *attention* explanation against IG/SHAP/Grad-CAM on the **same** model. Hypothesis to verify: the sync-supervised attention has higher deletion/insertion AUC than post-hoc methods → "the supervised explanation is more faithful than post-hoc ones."
- **Replace the weak "consistency Spearman 0.514"** with: (i) localization AP@tIoU, (ii) deletion/insertion AUC per method, (iii) a pointing-game accuracy (does the attention peak fall inside the GT manipulated region/mouth box). Keep Gini only as a sparsity descriptor, not as evidence of correctness.

### 2.3 Tier-2: close the accuracy gap *just enough*, and add real generalization

You do **not** need to beat AVFF. You need to (a) not be embarrassingly behind and (b) show generalization.

- **One encoder-upgrade ablation (optional but high-value).** Add a variant where ResNet-18 is replaced by **frozen DINOv2-S features + a small trainable adapter**, audio by **frozen Whisper/AV-HuBERT features**. Reuse the cached-feature pipeline that already exists in `deepshield/CMAR` (DINOv2-S + Whisper-tiny features are already extracted for FakeAVCeleb). Train only the fusion + heads on Kaggle. Expected: AUC moves up toward 0.98+, and — crucially — the sync loss + faithfulness story now rides on a SOTA-class backbone. Report both ResNet and DINOv2 variants so the *mechanism* (sync supervision) is shown to be backbone-agnostic.
- **True cross-dataset split.** Stop using only the merged pool. Add: **train on FakeAVCeleb → test on LAV-DF**, and **train on LAV-DF → test on FakeAVCeleb**. Report AUC/EER. This is the honest generalization number; it will be lower than 97% and that is fine — it sets up Project 4.

### 2.4 What to run on Kaggle (concrete)

1. `eval_metrics.py` — load saved test logits/labels → AUC, AP, EER, ROC/PR curves, ECE, per-category AUC. *(no GPU needed)*
2. `localization_eval.py` — for LAV-DF fake clips, extract attention diagonal → tIoU/AP@tIoU vs GT segments + pointing-game. *(1× T4)*
3. `faithfulness_eval.py` — Captum deletion/insertion for attention vs IG/SHAP/Grad-CAM on a 500-clip subset. *(1× T4, slow methods on subset only)*
4. `crossdataset_train.py` — two runs: FAVC→LAV-DF and LAV-DF→FAVC (reuse existing training code, change splits). *(P100, ~existing cost)*
5. *(optional)* `dino_adapter_variant.py` — frozen DINOv2-S + Whisper features + adapter + existing fusion/sync. *(T4, cheap, features cached)*

### 2.5 Results to gather (target table shape)

| Block | Metrics | Status |
|---|---|---|
| Main detection | Acc, **AUC, AP, EER**, macro-F1, per-category AUC, ECE | add AUC/EER/AP/ECE |
| Ablations | same metrics for no-sync / no-gate / no-curr | re-emit with AUC |
| **Localization** | tIoU, AP@tIoU, pointing-game (LAV-DF) | **new** |
| **Faithfulness** | deletion AUC, insertion AUC (attention vs IG/SHAP/Grad-CAM) | **new** |
| Cross-dataset | AUC/EER for FAVC→LAVDF, LAVDF→FAVC | **new** |
| Backbone ablation | ResNet vs DINOv2-adapter (AUC + faithfulness) | optional |

---

## 3. Narrative integration

PinPoint is **Pillar 1** and it must *establish the synchronization signal and prove it is meaningful*. The elevated framing — supervised, faithful, localization-validated attention — is what every later paper reuses:
- **Project 2 (PIN-Lite)** compresses this detector and asks whether the faithful attention survives compression — only meaningful because Project 1 proved the attention is faithful in the first place.
- **Project 3 (CertAV)** already uses the same frozen-feature AV detector family; the DINOv2-adapter variant here is the bridge to it.
- **Project 4 (generalization)** reuses the synchronization score as the *generator-agnostic* signal; the cross-dataset numbers you add in §2.3 are the baseline Project 4 improves on.

So the single change that pays off four times: **make the attention a supervised, validated explanation, and report AUC/EER + cross-dataset.**

---

## 4. Venue

- **Realistic & already scoped:** *Ain Shams Engineering Journal* (Q1, Elsevier) — the draft fits it. Acceptable, but it is a generalist engineering Q1, not a forensics flagship.
- **Recommended step-up once faithfulness + localization land:** **IEEE Transactions on Multimedia (TMM)** or **IEEE TIFS** (both Q1, forensics/multimedia home), or **ACM MM** if you want a conference. The localization-validated-explanation result is exactly what TIFS/TMM reward and Ain Shams does not specifically prize.
- **Bar to clear for the step-up:** AUC within ~1–2 points of AV SOTA on at least one in-domain benchmark **and** the faithfulness/localization tables. Recommendation: prepare the manuscript so it can go to TMM/TIFS; fall back to Ain Shams only if the localization result is weak.

---

## 5. Risk assessment

| Risk | Likelihood | Mitigation |
|---|---|---|
| Attention does **not** localize manipulated frames well (low AP@tIoU) | medium | The *faithfulness* result (deletion/insertion) is independent and likely positive given the sync supervision; lead with that. Localization becomes a secondary, honestly-reported result. |
| DINOv2-adapter variant underperforms or eats time | low–med | It is optional. The ResNet model already works; the variant is upside, not a dependency. Features are cached from CertAV. |
| Cross-dataset AUC is low | low (it's expected) | Frame it as the motivation for Project 4, not a failure. |
| Reviewer says "still below AVFF accuracy" | medium | The contribution is explicitly faithfulness+localization, not SOTA accuracy. Report the Pareto position honestly. |

**Most likely failure mode:** spending effort chasing accuracy parity with AVFF. **Don't.** The win is the supervised-faithful-explanation evaluation, which no AV competitor has.
