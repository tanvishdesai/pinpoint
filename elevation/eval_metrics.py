"""
eval_metrics.py  --  Tier-0 of the elevation plan.
==================================================
Loads the trained PinPoint checkpoint, runs the test set, and emits the metrics
the field actually compares on but the original paper lacked:

    * ROC-AUC, Average Precision (AP), Equal Error Rate (EER)
    * Accuracy / macro-F1 at the 0.5 and EER thresholds
    * Brier score + Expected Calibration Error (ECE) + reliability diagram
    * ROC and PR curves (PNG)
    * Per-category AUC: by source dataset (lavdf / fakeavceleb) and by
      manipulation type (audio_only / video_only / both)

It also dumps the raw per-sample probabilities/labels/metadata to
``<OUTPUT_DIR>/test_predictions.npz`` so the other elevation scripts (and any
re-plotting) never need to re-run inference.

No GPU strictly required, but a GPU makes the single inference pass fast.

USAGE (Kaggle):
    1. Edit the PATHS block in pinpoint_core.CoreConfig (data, metadata, model).
    2. Run:  python eval_metrics.py
"""

import os
import json
import argparse
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
from torch.utils.data import DataLoader
from sklearn.metrics import f1_score, roc_curve, precision_recall_curve

from pinpoint_core import CoreConfig, EvalDataset, make_collate, load_model, set_seed
import metrics_utils as M


@torch.no_grad()
def run_inference(model, loader, dataset, device):
    """Return probs, labels, and per-sample metadata aligned by row."""
    model.eval()
    probs, labels, meta = [], [], []
    for batch in loader:
        if batch is None:
            continue
        video = batch["video"].to(device)
        audio = batch["audio"].to(device)
        vmask = batch["video_mask"].to(device)
        with torch.cuda.amp.autocast(enabled=(device != "cpu")):
            logits, _, _ = model(video, audio, vmask)
        p = torch.sigmoid(logits).squeeze(1).float().cpu().numpy()
        probs.extend(p.tolist())
        labels.extend(batch["label"].numpy().astype(int).tolist())
        for idx in batch["index"].numpy().tolist():
            s = dataset.samples[idx]
            meta.append({"source": s["source"], "manip_type": s["manip_type"],
                         "is_fake": bool(s["is_fake"]), "file": s["file"]})
    return np.array(probs), np.array(labels), meta


def plot_roc_pr(labels, probs, out_dir):
    fpr, tpr, _ = roc_curve(labels, probs)
    auc, ap = M.auc_ap(labels, probs)
    plt.figure(figsize=(5, 5))
    plt.plot(fpr, tpr, label=f"AUC = {auc:.4f}")
    plt.plot([0, 1], [0, 1], "k--", alpha=0.4)
    plt.xlabel("False Positive Rate"); plt.ylabel("True Positive Rate")
    plt.title("ROC -- PinPoint test set"); plt.legend(loc="lower right")
    plt.tight_layout(); plt.savefig(os.path.join(out_dir, "roc_curve.png"), dpi=200); plt.close()

    prec, rec, _ = precision_recall_curve(labels, probs)
    plt.figure(figsize=(5, 5))
    plt.plot(rec, prec, label=f"AP = {ap:.4f}")
    plt.xlabel("Recall"); plt.ylabel("Precision")
    plt.title("Precision-Recall -- PinPoint test set"); plt.legend(loc="lower left")
    plt.tight_layout(); plt.savefig(os.path.join(out_dir, "pr_curve.png"), dpi=200); plt.close()


def plot_reliability(rows, ece_value, out_dir):
    centers = [r[0] for r in rows]
    accs = [r[1] for r in rows]
    plt.figure(figsize=(5, 5))
    plt.plot([0, 1], [0, 1], "k--", alpha=0.4, label="perfect")
    plt.bar(centers, [a if a == a else 0 for a in accs], width=1.0 / len(rows) * 0.9,
            alpha=0.7, edgecolor="k", label="accuracy")
    plt.xlabel("Confidence"); plt.ylabel("Accuracy")
    plt.title(f"Reliability diagram (ECE = {ece_value:.4f})"); plt.legend(loc="upper left")
    plt.tight_layout(); plt.savefig(os.path.join(out_dir, "reliability_diagram.png"), dpi=200); plt.close()


def per_category_auc(labels, probs, meta, key):
    """AUC of {fakes in category C} vs {all reals}, for each category C of `key`."""
    labels = np.asarray(labels); probs = np.asarray(probs)
    real_mask = labels == 0
    cats = sorted({m[key] for m in meta if m["is_fake"]})
    out = {}
    for c in cats:
        cat_fake = np.array([m["is_fake"] and m[key] == c for m in meta])
        sel = real_mask | cat_fake
        y = labels[sel]; s = probs[sel]
        auc, ap = M.auc_ap(y, s)
        out[c] = {"auc": auc, "ap": ap, "n_fake": int(cat_fake.sum()), "n_real": int(real_mask.sum())}
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="test")
    ap.add_argument("--max-samples", type=int, default=None)
    ap.add_argument("--ece-bins", type=int, default=15)
    args = ap.parse_args()

    set_seed(42)
    cfg = CoreConfig()
    out_dir = os.path.join(cfg.OUTPUT_DIR, "metrics")
    os.makedirs(out_dir, exist_ok=True)

    model, cfg = load_model(cfg)
    ds = EvalDataset(cfg, split=args.split, max_samples=args.max_samples)
    loader = DataLoader(ds, batch_size=8, shuffle=False,
                        collate_fn=make_collate(cfg.NUM_FRAMES), num_workers=2)

    probs, labels, meta = run_inference(model, loader, ds, cfg.DEVICE)
    print(f"Collected {len(labels)} predictions "
          f"({int((labels==0).sum())} real / {int((labels==1).sum())} fake)")

    # --- core metrics ---
    auc, ap_score = M.auc_ap(labels, probs)
    eer_value, eer_thr = M.eer(labels, probs)
    ece_value, rel_rows = M.ece(labels, probs, n_bins=args.ece_bins)
    preds_05 = (probs >= 0.5).astype(int)
    preds_eer = (probs >= eer_thr).astype(int)
    results = {
        "n": int(len(labels)),
        "n_real": int((labels == 0).sum()),
        "n_fake": int((labels == 1).sum()),
        "auc": auc, "ap": ap_score, "eer": eer_value, "eer_threshold": eer_thr,
        "acc@0.5": M.accuracy_at(labels, probs, 0.5),
        "acc@eer": M.accuracy_at(labels, probs, eer_thr),
        "macro_f1@0.5": float(f1_score(labels, preds_05, average="macro", zero_division=0)),
        "macro_f1@eer": float(f1_score(labels, preds_eer, average="macro", zero_division=0)),
        "fake_f1@0.5": float(f1_score(labels, preds_05, pos_label=1, zero_division=0)),
        "real_f1@0.5": float(f1_score(labels, preds_05, pos_label=0, zero_division=0)),
        "brier": M.brier(labels, probs),
        "ece": ece_value,
        "per_source_auc": per_category_auc(labels, probs, meta, "source"),
        "per_manip_auc": per_category_auc(labels, probs, meta, "manip_type"),
    }

    # --- plots ---
    plot_roc_pr(labels, probs, out_dir)
    plot_reliability(rel_rows, ece_value, out_dir)

    # --- dumps ---
    np.savez_compressed(os.path.join(out_dir, "test_predictions.npz"),
                        probs=probs, labels=labels,
                        sources=np.array([m["source"] for m in meta]),
                        manip_types=np.array([m["manip_type"] for m in meta]),
                        files=np.array([m["file"] for m in meta]))
    with open(os.path.join(out_dir, "metrics.json"), "w") as f:
        json.dump(results, f, indent=2)

    # --- report ---
    print("\n" + "=" * 60)
    print("PINPOINT TEST METRICS (elevation Tier-0)")
    print("=" * 60)
    print(f"  AUC            : {auc:.4f}")
    print(f"  AP             : {ap_score:.4f}")
    print(f"  EER            : {eer_value:.4f}  (thr={eer_thr:.4f})")
    print(f"  Acc @0.5 / @EER: {results['acc@0.5']:.4f} / {results['acc@eer']:.4f}")
    print(f"  Macro-F1 @0.5  : {results['macro_f1@0.5']:.4f}")
    print(f"  Brier          : {results['brier']:.4f}")
    print(f"  ECE            : {ece_value:.4f}")
    print("\n  Per-source AUC:")
    for k, v in results["per_source_auc"].items():
        print(f"    {k:14s} AUC={v['auc']:.4f} AP={v['ap']:.4f} (fake={v['n_fake']}, real={v['n_real']})")
    print("  Per-manipulation AUC:")
    for k, v in results["per_manip_auc"].items():
        print(f"    {k:14s} AUC={v['auc']:.4f} AP={v['ap']:.4f} (fake={v['n_fake']}, real={v['n_real']})")
    print(f"\nSaved metrics.json, curves and test_predictions.npz to: {out_dir}")
    print("=" * 60)


if __name__ == "__main__":
    main()
