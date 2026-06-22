"""
faithfulness_eval.py  --  Tier-1 (new core contribution, part B).
=================================================================
Quantifies whether PinPoint's sync-supervised *attention* is a more FAITHFUL
explanation than standard post-hoc methods, using deletion / insertion curves on
video frames (Petsiuk et al.).

For each clip we rank the 30 video frames by an importance score, then:
  * DELETION : progressively replace the most-important frames with a baseline
               and watch the predicted-class probability fall. Faithful
               explanation -> probability drops fast -> LOW deletion AUC.
  * INSERTION: progressively restore the most-important frames into a baseline
               clip and watch the probability rise. Faithful -> HIGH insertion AUC.

Importance methods compared on the SAME frames / SAME model:
  * attention            (sum of audio->video attention received per frame)
  * integrated_gradients (|IG| of p_fake w.r.t. the video input, per frame)
  * grad_cam             (Grad-CAM energy at the last ResNet block, per frame)
  * random               (control)

Hypothesis (from the elevation plan): the supervised attention is at least as
faithful as IG/Grad-CAM -> "the explanation is the mechanism, not a guess".

USAGE (Kaggle, 1x T4):
    python faithfulness_eval.py --n-clips 300 --ig-steps 20
"""

import os
import json
import argparse
import contextlib
import numpy as np
import torch
from torch.utils.data import DataLoader

from pinpoint_core import CoreConfig, EvalDataset, make_collate, load_model, set_seed


@contextlib.contextmanager
def cudnn_disabled():
    """Lets the GRU run its backward pass in eval() mode (no dropout noise)."""
    prev = torch.backends.cudnn.enabled
    torch.backends.cudnn.enabled = False
    try:
        yield
    finally:
        torch.backends.cudnn.enabled = prev


@torch.no_grad()
def p_fake(model, video, audio, vmask):
    logits, _, _ = model(video, audio, vmask)
    return torch.sigmoid(logits).squeeze(1)


# ----------------------------- importances --------------------------------
@torch.no_grad()
def imp_attention(model, video, audio, vmask):
    _, _, attn = model(video, audio, vmask)
    A = attn[0].float().cpu().numpy()        # [Ta, Tv]
    return A.sum(axis=0)                      # per-frame attention received


def imp_integrated_gradients(model, video, audio, vmask, steps):
    baseline = torch.zeros_like(video)
    diff = video - baseline
    grad_sum = torch.zeros_like(video)
    with cudnn_disabled():
        for alpha in torch.linspace(0, 1, steps, device=video.device):
            x = (baseline + alpha * diff).clone().requires_grad_(True)
            model.zero_grad(set_to_none=True)
            logits, _, _ = model(x, audio, vmask)
            torch.sigmoid(logits).sum().backward()
            grad_sum += x.grad.detach()
    ig = (diff * (grad_sum / steps)).detach()   # [1, Tv, C, H, W]
    return ig[0].abs().sum(dim=(1, 2, 3)).cpu().numpy()


def imp_grad_cam(model, video, audio, vmask):
    feats, grads = [], []
    target = model.video_extractor.feature_extractor[7]
    h1 = target.register_forward_hook(lambda m, i, o: feats.append(o))
    h2 = target.register_full_backward_hook(lambda m, gi, go: grads.append(go[0]))
    try:
        with cudnn_disabled():
            x = video.clone().requires_grad_(True)
            model.zero_grad(set_to_none=True)
            logits, _, _ = model(x, audio, vmask)
            torch.sigmoid(logits).sum().backward()
        f = feats[0].detach()                    # [Tv, C, h, w]
        g = grads[0].detach()                    # [Tv, C, h, w]
        weights = g.mean(dim=(2, 3), keepdim=True)
        cam = torch.relu((weights * f).sum(dim=1))   # [Tv, h, w]
        return cam.flatten(1).mean(dim=1).cpu().numpy()
    finally:
        h1.remove(); h2.remove()


IMPORTANCE_FNS = {
    "attention": lambda m, v, a, mk, ig_steps: imp_attention(m, v, a, mk),
    "integrated_gradients": lambda m, v, a, mk, ig_steps: imp_integrated_gradients(m, v, a, mk, ig_steps),
    "grad_cam": lambda m, v, a, mk, ig_steps: imp_grad_cam(m, v, a, mk),
    "random": lambda m, v, a, mk, ig_steps: np.random.rand(v.shape[1]),
}


# --------------------------- deletion / insertion -------------------------
@torch.no_grad()
def _scores_for_variants(model, variants, audio, vmask, chunk=16):
    """variants: [K, Tv, C, H, W] -> p_fake for each, in one (chunked) pass."""
    out = []
    K = variants.shape[0]
    for i in range(0, K, chunk):
        vb = variants[i:i + chunk]
        b = vb.shape[0]
        a = audio.repeat(b, 1, 1)
        mk = vmask.repeat(b, 1)
        logits, _, _ = model(vb, a, mk)
        out.append(torch.sigmoid(logits).squeeze(1).float().cpu().numpy())
    return np.concatenate(out)


def deletion_insertion_curves(model, video, audio, vmask, order, baseline, pred_is_fake):
    """order: frame indices most->least important. Returns (del_auc, ins_auc)."""
    Tv = video.shape[1]
    # deletion variants: k frames (most important first) set to baseline
    del_variants = []
    cur = video.clone()
    del_variants.append(cur.clone())
    for k in range(Tv):
        cur = cur.clone()
        cur[0, order[k]] = baseline[0, order[k]]
        del_variants.append(cur.clone())
    del_variants = torch.cat(del_variants, dim=0)

    # insertion variants: start from baseline, restore k most important frames
    ins_variants = []
    cur = baseline.clone()
    ins_variants.append(cur.clone())
    for k in range(Tv):
        cur = cur.clone()
        cur[0, order[k]] = video[0, order[k]]
        ins_variants.append(cur.clone())
    ins_variants = torch.cat(ins_variants, dim=0)

    del_p = _scores_for_variants(model, del_variants, audio, vmask)
    ins_p = _scores_for_variants(model, ins_variants, audio, vmask)
    # align to predicted class probability
    if not pred_is_fake:
        del_p = 1.0 - del_p
        ins_p = 1.0 - ins_p
    xs = np.linspace(0, 1, Tv + 1)
    return float(np.trapz(del_p, xs)), float(np.trapz(ins_p, xs))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-clips", type=int, default=300)
    ap.add_argument("--ig-steps", type=int, default=20)
    ap.add_argument("--methods", nargs="+",
                    default=["attention", "integrated_gradients", "grad_cam", "random"])
    args = ap.parse_args()

    set_seed(42)
    cfg = CoreConfig()
    out_dir = os.path.join(cfg.OUTPUT_DIR, "faithfulness")
    os.makedirs(out_dir, exist_ok=True)

    model, cfg = load_model(cfg)
    ds = EvalDataset(cfg, split="test", max_samples=args.n_clips)
    loader = DataLoader(ds, batch_size=1, shuffle=False,
                        collate_fn=make_collate(cfg.NUM_FRAMES), num_workers=2)

    agg = {m: {"del": [], "ins": []} for m in args.methods}
    n = 0
    for batch in loader:
        if batch is None:
            continue
        video = batch["video"].to(cfg.DEVICE)
        audio = batch["audio"].to(cfg.DEVICE)
        vmask = batch["video_mask"].to(cfg.DEVICE)
        baseline = torch.zeros_like(video)
        pred_is_fake = bool(p_fake(model, video, audio, vmask).item() >= 0.5)

        for m in args.methods:
            imp = IMPORTANCE_FNS[m](model, video, audio, vmask, args.ig_steps)
            order = np.argsort(-np.asarray(imp))      # most important first
            del_auc, ins_auc = deletion_insertion_curves(
                model, video, audio, vmask, order, baseline, pred_is_fake)
            agg[m]["del"].append(del_auc)
            agg[m]["ins"].append(ins_auc)
        n += 1

    results = {"n_clips": n, "methods": {}}
    for m in args.methods:
        d = np.array(agg[m]["del"]); i = np.array(agg[m]["ins"])
        results["methods"][m] = {
            "deletion_auc": float(d.mean()) if len(d) else float("nan"),
            "insertion_auc": float(i.mean()) if len(i) else float("nan"),
            "faithfulness": float(i.mean() - d.mean()) if len(d) else float("nan"),
        }
    with open(os.path.join(out_dir, "faithfulness.json"), "w") as f:
        json.dump(results, f, indent=2)

    print("\n" + "=" * 64)
    print("PINPOINT FAITHFULNESS  (deletion / insertion, n=%d clips)" % n)
    print("=" * 64)
    print(f"  {'method':22s} {'deletion AUC':>13s} {'insertion AUC':>14s} {'ins-del':>9s}")
    print("  " + "-" * 60)
    for m in args.methods:
        r = results["methods"][m]
        print(f"  {m:22s} {r['deletion_auc']:13.4f} {r['insertion_auc']:14.4f} {r['faithfulness']:9.4f}")
    print("\n  Lower deletion AUC = better; higher insertion AUC = better.")
    print(f"  Saved faithfulness.json to: {out_dir}")
    print("=" * 64)


if __name__ == "__main__":
    main()
