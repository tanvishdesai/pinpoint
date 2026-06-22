"""
metrics_utils.py
================
Small, dependency-light metric helpers shared by the elevation scripts.
Only needs numpy + scikit-learn (already used by PinPoint.py).
"""

import numpy as np
from sklearn.metrics import roc_auc_score, average_precision_score, roc_curve


def auc_ap(labels, scores):
    """ROC-AUC and Average Precision. Returns (nan, nan) if a class is absent."""
    labels = np.asarray(labels)
    scores = np.asarray(scores)
    if len(np.unique(labels)) < 2:
        return float("nan"), float("nan")
    return float(roc_auc_score(labels, scores)), float(average_precision_score(labels, scores))


def eer(labels, scores):
    """Equal Error Rate and the threshold at which FPR == FNR."""
    labels = np.asarray(labels)
    scores = np.asarray(scores)
    if len(np.unique(labels)) < 2:
        return float("nan"), float("nan")
    fpr, tpr, thr = roc_curve(labels, scores)
    fnr = 1 - tpr
    idx = int(np.nanargmin(np.abs(fnr - fpr)))
    return float((fpr[idx] + fnr[idx]) / 2.0), float(thr[idx])


def ece(labels, probs, n_bins=15):
    """Expected Calibration Error (equal-width confidence bins).

    Uses confidence = max(p, 1-p) and accuracy of the argmax (0.5-threshold)
    prediction inside each bin, the standard top-label ECE.
    """
    labels = np.asarray(labels).astype(int)
    probs = np.asarray(probs, dtype=float)
    preds = (probs >= 0.5).astype(int)
    conf = np.where(preds == 1, probs, 1 - probs)
    correct = (preds == labels).astype(float)
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    total = len(labels)
    e = 0.0
    rows = []
    for i in range(n_bins):
        lo, hi = bins[i], bins[i + 1]
        m = (conf > lo) & (conf <= hi) if i > 0 else (conf >= lo) & (conf <= hi)
        if m.sum() == 0:
            rows.append((0.5 * (lo + hi), np.nan, np.nan, 0))
            continue
        acc = correct[m].mean()
        avg_conf = conf[m].mean()
        e += (m.sum() / total) * abs(acc - avg_conf)
        rows.append((0.5 * (lo + hi), acc, avg_conf, int(m.sum())))
    return float(e), rows


def brier(labels, probs):
    labels = np.asarray(labels, dtype=float)
    probs = np.asarray(probs, dtype=float)
    return float(np.mean((probs - labels) ** 2))


def accuracy_at(labels, probs, threshold=0.5):
    labels = np.asarray(labels).astype(int)
    preds = (np.asarray(probs) >= threshold).astype(int)
    return float((preds == labels).mean())


# ---------------------------------------------------------------------------
# Temporal localization metrics (segment-level)
# ---------------------------------------------------------------------------
def segment_iou(a, b):
    """1-D temporal IoU between two [start, end] segments."""
    inter = max(0.0, min(a[1], b[1]) - max(a[0], b[0]))
    union = (a[1] - a[0]) + (b[1] - b[0]) - inter
    return inter / union if union > 0 else 0.0


def average_precision_at_tiou(preds, gts, tiou_threshold):
    """Temporal-action-detection style AP at a single tIoU threshold.

    preds : list of (clip_id, [start, end], score)
    gts   : dict {clip_id: [[start, end], ...]}
    """
    if not preds:
        return 0.0
    preds = sorted(preds, key=lambda x: -x[2])
    npos = sum(len(v) for v in gts.values())
    if npos == 0:
        return 0.0
    matched = {cid: [False] * len(segs) for cid, segs in gts.items()}
    tp = np.zeros(len(preds))
    fp = np.zeros(len(preds))
    for i, (cid, seg, _score) in enumerate(preds):
        gt_segs = gts.get(cid, [])
        best_iou, best_j = 0.0, -1
        for j, g in enumerate(gt_segs):
            iou = segment_iou(seg, g)
            if iou > best_iou:
                best_iou, best_j = iou, j
        if best_iou >= tiou_threshold and best_j >= 0 and not matched[cid][best_j]:
            tp[i] = 1
            matched[cid][best_j] = True
        else:
            fp[i] = 1
    tp_cum = np.cumsum(tp)
    fp_cum = np.cumsum(fp)
    rec = tp_cum / (npos + 1e-12)
    prec = tp_cum / np.maximum(tp_cum + fp_cum, 1e-12)
    # VOC-style all-points AP
    mrec = np.concatenate([[0.0], rec, [1.0]])
    mprec = np.concatenate([[0.0], prec, [0.0]])
    for k in range(len(mprec) - 1, 0, -1):
        mprec[k - 1] = max(mprec[k - 1], mprec[k])
    idx = np.where(mrec[1:] != mrec[:-1])[0]
    return float(np.sum((mrec[idx + 1] - mrec[idx]) * mprec[idx + 1]))
