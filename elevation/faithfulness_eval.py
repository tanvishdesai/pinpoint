"""
faithfulness_eval.py  --  Tier-1 (new core contribution, part B). [v2 - FIXED]
=============================================================================
Quantifies whether PinPoint's sync-supervised *attention* is a more FAITHFUL
explanation than standard post-hoc methods, using deletion / insertion curves
(Petsiuk et al.).

WHY v2 EXISTS
-------------
v1 perturbed only VIDEO frames with a ZERO baseline and reported ~0.99 deletion
AND insertion AUC for EVERY method incl. random -> the test was DEGENERATE:
  * PinPoint's classifier reads `pooled = audio_feat.mean(1)` (audio-centric),
    so for audio-driven fakes, zeroing video frames does not move p_fake.
  * an all-zero video is out-of-distribution but still reads "fake", pinning
    both curves at ~0.99.
Both failure modes are fixed here:
  1. IN-DISTRIBUTION baseline: replace a unit with the per-clip MEAN frame /
     MEAN MFCC step (not zeros), so the baselined state is a plausible,
     low-information input.
  2. MODALITY-AWARE: each fake clip is routed to the track matching the
     modality that was actually manipulated:
        video_only / both -> VIDEO track  (attention_v, IG_v, Grad-CAM, random)
        audio_only / both -> AUDIO track  (attention_a, IG_a, random)
     so every explanation is tested against the modality the model uses for
     that clip.
  3. SANITY: prints mean p_fake at the fully-baselined state per track. If the
     baseline truly neutralises the evidence this should fall WELL BELOW 0.5;
     if it does not, the test is still degenerate and the number is invalid.

METRIC (computed on clips that are actually fake AND predicted fake):
  * DELETION : remove the most-important units (-> baseline) and watch p_fake
               fall. Faithful -> fast drop -> LOW deletion AUC.
  * INSERTION: start from the baselined clip and restore the most-important
               units, watch p_fake rise. Faithful -> HIGH insertion AUC.
  * faithfulness = insertion_auc - deletion_auc   (higher is better)

Hypothesis: sync-supervised attention >= IG / Grad-CAM (and >> random).

USAGE (Kaggle, 1x T4):
    python faithfulness_eval.py --n-clips 300 --ig-steps 20
    python faithfulness_eval.py --n-clips 300 --track audio        # audio only
    python faithfulness_eval.py --n-clips 300 --baseline zero      # reproduce v1 baseline
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
    """Lets the GRU run its backward pass in eval() mode (needed for IG/Grad-CAM)."""
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


# ----------------------------- baselines ----------------------------------
def make_baseline(x, kind):
    """In-distribution baseline for a [1, T, ...] tensor.

    'mean' : replace every unit with the per-clip temporal mean (low-information
             but in-distribution -> the recommended deletion/insertion baseline).
    'zero' : all zeros (v1 behaviour, kept only for reproducing the degenerate
             result).
    """
    if kind == "zero":
        return torch.zeros_like(x)
    # 'mean': broadcast the temporal mean back over the time axis
    return x.mean(dim=1, keepdim=True).expand_as(x).clone()


# ----------------------------- importances --------------------------------
# All importance fns return a 1-D numpy array, one score per perturbable UNIT
# along the time axis of the modality being tested (video frames or audio steps).

@torch.no_grad()
def imp_attention_video(model, video, audio, vmask):
    _, _, attn = model(video, audio, vmask)
    A = attn[0].float().cpu().numpy()          # [Ta, Tv]
    return A.sum(axis=0)                        # per video frame (sum over audio)


@torch.no_grad()
def imp_attention_audio(model, video, audio, vmask):
    _, _, attn = model(video, audio, vmask)
    A = attn[0].float().cpu().numpy()          # [Ta, Tv]
    return A.sum(axis=1)                        # per audio step (sum over video)


def imp_ig_video(model, video, audio, vmask, steps):
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


def imp_ig_audio(model, video, audio, vmask, steps):
    baseline = torch.zeros_like(audio)
    diff = audio - baseline
    grad_sum = torch.zeros_like(audio)
    with cudnn_disabled():
        for alpha in torch.linspace(0, 1, steps, device=audio.device):
            x = (baseline + alpha * diff).clone().requires_grad_(True)
            model.zero_grad(set_to_none=True)
            logits, _, _ = model(video, x, vmask)
            torch.sigmoid(logits).sum().backward()
            grad_sum += x.grad.detach()
    ig = (diff * (grad_sum / steps)).detach()   # [1, Ta, num_mfcc]
    return ig[0].abs().sum(dim=1).cpu().numpy()  # per audio step


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


# method-name -> importance fn, grouped by which modality's units it ranks.
VIDEO_METHODS = {
    "attention": lambda m, v, a, mk, ig: imp_attention_video(m, v, a, mk),
    "integrated_gradients": lambda m, v, a, mk, ig: imp_ig_video(m, v, a, mk, ig),
    "grad_cam": lambda m, v, a, mk, ig: imp_grad_cam(m, v, a, mk),
    "random": lambda m, v, a, mk, ig: np.random.rand(v.shape[1]),
}
AUDIO_METHODS = {
    "attention": lambda m, v, a, mk, ig: imp_attention_audio(m, v, a, mk),
    "integrated_gradients": lambda m, v, a, mk, ig: imp_ig_audio(m, v, a, mk, ig),
    "random": lambda m, v, a, mk, ig: np.random.rand(a.shape[1]),
}


# --------------------------- deletion / insertion -------------------------
@torch.no_grad()
def _scores_video_varies(model, video_variants, audio, vmask, chunk=16):
    out, K = [], video_variants.shape[0]
    for i in range(0, K, chunk):
        vb = video_variants[i:i + chunk]; b = vb.shape[0]
        logits, _, _ = model(vb, audio.repeat(b, 1, 1), vmask.repeat(b, 1))
        out.append(torch.sigmoid(logits).squeeze(1).float().cpu().numpy())
    return np.concatenate(out)


@torch.no_grad()
def _scores_audio_varies(model, video, audio_variants, vmask, chunk=16):
    out, K = [], audio_variants.shape[0]
    for i in range(0, K, chunk):
        ab = audio_variants[i:i + chunk]; b = ab.shape[0]
        logits, _, _ = model(video.repeat(b, 1, 1, 1, 1), ab, vmask.repeat(b, 1))
        out.append(torch.sigmoid(logits).squeeze(1).float().cpu().numpy())
    return np.concatenate(out)


def _build_variants(full, base, order):
    """full/base: [1, T, ...]. Returns (deletion[T+1,...], insertion[T+1,...])."""
    T = full.shape[1]
    del_list = [full.clone()]
    cur = full.clone()
    for k in range(T):
        cur = cur.clone(); cur[0, order[k]] = base[0, order[k]]; del_list.append(cur.clone())
    ins_list = [base.clone()]
    cur = base.clone()
    for k in range(T):
        cur = cur.clone(); cur[0, order[k]] = full[0, order[k]]; ins_list.append(cur.clone())
    return torch.cat(del_list, 0), torch.cat(ins_list, 0)


_TRAPZ = getattr(np, "trapezoid", None) or np.trapz  # np.trapz removed in NumPy 2.x


def del_ins_curves(model, video, audio, vmask, order, modality, baseline_kind):
    """Returns (del_auc, ins_auc, baseline_p) tracking p_fake (clip is fake)."""
    if modality == "video":
        base = make_baseline(video, baseline_kind)
        del_v, ins_v = _build_variants(video, base, order)
        del_p = _scores_video_varies(model, del_v, audio, vmask)
        ins_p = _scores_video_varies(model, ins_v, audio, vmask)
    else:
        base = make_baseline(audio, baseline_kind)
        del_a, ins_a = _build_variants(audio, base, order)
        del_p = _scores_audio_varies(model, video, del_a, vmask)
        ins_p = _scores_audio_varies(model, video, ins_a, vmask)
    T = order.shape[0]
    xs = np.linspace(0, 1, T + 1)
    # baseline_p = p_fake at the fully-baselined state (last deletion variant).
    return float(_TRAPZ(del_p, xs)), float(_TRAPZ(ins_p, xs)), float(del_p[-1])


# --------------------------------- driver ---------------------------------
def _route_tracks(manip_type, want):
    """Which tracks a fake clip contributes to, given its manipulation type."""
    tracks = []
    if want in ("video", "both") and manip_type in ("video_only", "both", "fake_unknown"):
        tracks.append("video")
    if want in ("audio", "both") and manip_type in ("audio_only", "both", "fake_unknown"):
        tracks.append("audio")
    return tracks


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-clips", type=int, default=300,
                    help="max fake clips to scan (per the dataset's own sampling)")
    ap.add_argument("--ig-steps", type=int, default=20)
    ap.add_argument("--track", choices=["video", "audio", "both"], default="both")
    ap.add_argument("--baseline", choices=["mean", "zero"], default="mean",
                    help="deletion/insertion baseline; 'mean' is in-distribution (recommended)")
    ap.add_argument("--require-pred-fake", action="store_true", default=True,
                    help="only score clips the model predicts fake (default on)")
    args = ap.parse_args()

    set_seed(42)
    cfg = CoreConfig()
    out_dir = os.path.join(cfg.OUTPUT_DIR, "faithfulness")
    os.makedirs(out_dir, exist_ok=True)

    model, cfg = load_model(cfg)
    # fake_only=True: faithfulness of "remove the manipulation evidence" only makes
    # sense on clips that contain a manipulation.
    ds = EvalDataset(cfg, split="test", fake_only=True, max_samples=args.n_clips)
    loader = DataLoader(ds, batch_size=1, shuffle=False,
                        collate_fn=make_collate(cfg.NUM_FRAMES), num_workers=2)

    method_sets = {"video": VIDEO_METHODS, "audio": AUDIO_METHODS}
    agg = {trk: {m: {"del": [], "ins": []} for m in method_sets[trk]}
           for trk in ("video", "audio")}
    base_p = {"video": [], "audio": []}          # saturation sanity per track
    n_clip = {"video": 0, "audio": 0}
    skipped_pred_real = 0
    n_scanned = 0

    for batch in loader:
        if batch is None:
            continue
        idx = int(batch["index"][0].item())
        info = ds.samples[idx]
        manip = info.get("manip_type", "fake_unknown")
        tracks = _route_tracks(manip, args.track)
        if not tracks:
            continue

        video = batch["video"].to(cfg.DEVICE)
        audio = batch["audio"].to(cfg.DEVICE)
        vmask = batch["video_mask"].to(cfg.DEVICE)

        if args.require_pred_fake and p_fake(model, video, audio, vmask).item() < 0.5:
            skipped_pred_real += 1
            continue
        n_scanned += 1

        for trk in tracks:
            recorded_base = False
            for m, fn in method_sets[trk].items():
                imp = np.asarray(fn(model, video, audio, vmask, args.ig_steps), dtype=np.float64)
                order = np.argsort(-imp)          # most important first
                del_auc, ins_auc, bp = del_ins_curves(
                    model, video, audio, vmask, order, trk, args.baseline)
                agg[trk][m]["del"].append(del_auc)
                agg[trk][m]["ins"].append(ins_auc)
                if not recorded_base:             # identical state across methods
                    base_p[trk].append(bp)
                    recorded_base = True
            n_clip[trk] += 1

    # ------------------------------ summarise ------------------------------
    results = {"n_scanned": n_scanned, "skipped_pred_real": skipped_pred_real,
               "baseline": args.baseline, "tracks": {}}
    for trk in ("video", "audio"):
        if n_clip[trk] == 0:
            continue
        trk_res = {"n_clips": n_clip[trk],
                   "baseline_p_fake_mean": float(np.mean(base_p[trk])),
                   "methods": {}}
        for m in method_sets[trk]:
            d = np.array(agg[trk][m]["del"]); i = np.array(agg[trk][m]["ins"])
            trk_res["methods"][m] = {
                "deletion_auc": float(d.mean()),
                "insertion_auc": float(i.mean()),
                "faithfulness": float(i.mean() - d.mean()),
            }
        results["tracks"][trk] = trk_res

    with open(os.path.join(out_dir, "faithfulness.json"), "w") as f:
        json.dump(results, f, indent=2)

    # ------------------------------- report --------------------------------
    print("\n" + "=" * 70)
    print("PINPOINT FAITHFULNESS  (deletion / insertion, baseline=%s)" % args.baseline)
    print("  scanned fake+pred-fake clips: %d   (skipped pred-real: %d)"
          % (n_scanned, skipped_pred_real))
    print("=" * 70)
    for trk in ("video", "audio"):
        if trk not in results["tracks"]:
            continue
        r = results["tracks"][trk]
        bp = r["baseline_p_fake_mean"]
        flag = "  <-- OK (evidence removed)" if bp < 0.5 else "  <-- WARNING: baseline still reads fake -> test invalid"
        print(f"\n  [{trk.upper()} track]  n={r['n_clips']}   "
              f"mean p_fake @ fully-baselined = {bp:.4f}{flag}")
        print(f"    {'method':22s} {'deletion AUC':>13s} {'insertion AUC':>14s} {'ins-del':>9s}")
        print("    " + "-" * 60)
        for m, mr in r["methods"].items():
            print(f"    {m:22s} {mr['deletion_auc']:13.4f} "
                  f"{mr['insertion_auc']:14.4f} {mr['faithfulness']:9.4f}")
    print("\n  Lower deletion AUC = better; higher insertion AUC = better;")
    print("  faithfulness = insertion - deletion (higher = more faithful).")
    print("  A valid test needs 'mean p_fake @ fully-baselined' < 0.5.")
    print(f"\n  Saved faithfulness.json to: {out_dir}")
    print("=" * 70)


if __name__ == "__main__":
    main()
