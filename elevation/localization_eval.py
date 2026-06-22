"""
localization_eval.py  --  Tier-1 (new core contribution, part A).
=================================================================
Tests whether PinPoint's *audio->video cross-attention* localizes the actually
manipulated frames in LAV-DF -- WITHOUT ever having been trained on localization
labels (the synchronization loss only ever saw real, synced samples).

Signal
------
From the last cross-attention map A (shape [T_audio, T_video]) we derive a
per-video-frame "manipulation suspicion" by measuring, for each audio step, how
much of its attention lies OUTSIDE the synchronized diagonal band the model was
trained to follow (center = i / MFCC_FRAMES_PER_VIDEO_FRAME, half-width =
SYNC_LOSS_BANDWIDTH). A synced/genuine frame keeps attention on the diagonal
(low suspicion); a manipulated frame breaks synchronization (high suspicion).

A video frame v sits at clip-time  t_v = v/(T_video-1) * duration  because the
preprocessing sampled frames evenly across the whole clip. We compare t_v against
the LAV-DF ``fake_periods`` (seconds) ground truth.

Metrics
-------
  * mean per-clip frame-level AUC / AP   (does suspicion rank manipulated frames
                                          above genuine frames within a clip?)
  * pointing-game accuracy               (does the peak-suspicion frame fall in a
                                          GT manipulated segment?)
  * AP@tIoU {0.5, 0.75} + mean best-tIoU (segment-level temporal detection)
  * random baselines for sanity.

USAGE (Kaggle, 1x T4):
    python localization_eval.py            # all LAV-DF fake test clips w/ GT
    python localization_eval.py --max-clips 1000
"""

import os
import json
import argparse
import numpy as np
import torch
from torch.utils.data import DataLoader

from pinpoint_core import (CoreConfig, EvalDataset, make_collate, load_model,
                           set_seed, _basename)
import metrics_utils as M


def clip_duration(sample, lavdf_index, fps):
    if sample.get("duration"):
        return float(sample["duration"])
    ref = lavdf_index.get(_basename(sample["file"])) if lavdf_index else None
    if ref:
        if ref.get("duration"):
            return float(ref["duration"])
        vf = ref.get("video_frames")
        if vf:
            return float(vf) / fps
    if sample.get("video_frames"):
        return float(sample["video_frames"]) / fps
    return None


def attention_suspicion(attn_map, r, bw):
    """attn_map: [T_audio, T_video] (rows sum to 1). Returns per-video-frame
    suspicion in [0, 1] (nan-filled by interpolation where unmapped)."""
    A = attn_map
    Ta, Tv = A.shape
    acc = np.zeros(Tv)
    cnt = np.zeros(Tv)
    for i in range(Ta):
        center = int(i / r)
        if center >= Tv:
            continue
        lo = max(0, center - bw)
        hi = min(Tv, center + bw + 1)
        diag = float(A[i, lo:hi].sum())
        acc[center] += (1.0 - diag)
        cnt[center] += 1
    susp = np.where(cnt > 0, acc / np.maximum(cnt, 1), np.nan)
    # interpolate any unmapped frames
    if np.isnan(susp).any():
        idx = np.arange(Tv)
        good = ~np.isnan(susp)
        if good.sum() == 0:
            susp = np.zeros(Tv)
        else:
            susp = np.interp(idx, idx[good], susp[good])
    return susp


def frame_times(Tv, duration):
    return np.array([v / max(Tv - 1, 1) * duration for v in range(Tv)])


def frame_labels(times, fake_periods):
    y = np.zeros(len(times), dtype=int)
    for v, t in enumerate(times):
        for s, e in fake_periods:
            if s <= t <= e:
                y[v] = 1
                break
    return y


def proposals_from_suspicion(susp, times, duration, Tv, thresholds):
    """Multi-threshold contiguous-run proposals -> list of ([s,e], score) in seconds."""
    rng = susp.max() - susp.min()
    norm = (susp - susp.min()) / rng if rng > 1e-9 else np.zeros_like(susp)
    half = (duration / max(Tv - 1, 1)) / 2.0
    props = []
    for th in thresholds:
        above = norm >= th
        v = 0
        while v < Tv:
            if above[v]:
                start = v
                while v < Tv and above[v]:
                    v += 1
                end = v - 1
                seg = [max(0.0, times[start] - half), min(duration, times[end] + half)]
                props.append((seg, float(susp[start:end + 1].mean())))
            else:
                v += 1
    return props


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-clips", type=int, default=None)
    ap.add_argument("--tiou", type=float, nargs="+", default=[0.5, 0.75])
    args = ap.parse_args()

    set_seed(42)
    cfg = CoreConfig()
    out_dir = os.path.join(cfg.OUTPUT_DIR, "localization")
    os.makedirs(out_dir, exist_ok=True)

    model, cfg = load_model(cfg)
    r = cfg.MFCC_FRAMES_PER_VIDEO_FRAME
    bw = cfg.SYNC_LOSS_BANDWIDTH

    # LAV-DF fakes with usable segment GT only.
    ds = EvalDataset(cfg, split="test", source_filter="lavdf", fake_only=True,
                     require_fake_periods=True, max_samples=args.max_clips)
    if len(ds) == 0:
        print("No LAV-DF fake clips with fake_periods found. "
              "Check LAVDF_METADATA_PATH / that the merged metadata keeps fake_periods.")
        return
    loader = DataLoader(ds, batch_size=1, shuffle=False,
                        collate_fn=make_collate(cfg.NUM_FRAMES), num_workers=2)

    per_clip_auc, per_clip_ap = [], []
    pointing_hits, pointing_total = 0, 0
    rand_pointing = []
    all_preds, all_gts = [], []
    skipped_no_duration = 0

    model.eval()
    for batch in loader:
        if batch is None:
            continue
        idx = int(batch["index"][0])
        sample = ds.samples[idx]
        dur = clip_duration(sample, ds.lavdf_index, cfg.LAVDF_FPS)
        if not dur or dur <= 0 or not sample["fake_periods"]:
            skipped_no_duration += 1
            continue

        with torch.no_grad():
            video = batch["video"].to(cfg.DEVICE)
            audio = batch["audio"].to(cfg.DEVICE)
            vmask = batch["video_mask"].to(cfg.DEVICE)
            _, _, attn = model(video, audio, vmask)
        A = attn[0].float().cpu().numpy()
        Tv = A.shape[1]

        susp = attention_suspicion(A, r, bw)
        times = frame_times(Tv, dur)
        y = frame_labels(times, sample["fake_periods"])
        if y.sum() == 0 or y.sum() == Tv:
            # no within-clip contrast for frame-AUC; still usable for pointing/tIoU
            pass
        else:
            auc, ap_ = M.auc_ap(y, susp)
            if auc == auc:
                per_clip_auc.append(auc)
                per_clip_ap.append(ap_)

        # pointing game
        peak = int(np.argmax(susp))
        pointing_total += 1
        if y[peak] == 1:
            pointing_hits += 1
        rand_pointing.append(y.mean())  # expected hit rate of a random frame

        # segment proposals for AP@tIoU
        cid = sample["file"]
        props = proposals_from_suspicion(susp, times, dur, Tv,
                                         thresholds=np.linspace(0.3, 0.7, 5))
        for seg, score in props:
            all_preds.append((cid, seg, score))
        all_gts.append((cid, [list(p) for p in sample["fake_periods"]]))

    gts_dict = {cid: segs for cid, segs in all_gts}
    ap_at_tiou = {f"AP@{t}": M.average_precision_at_tiou(all_preds, gts_dict, t)
                  for t in args.tiou}

    results = {
        "n_clips": pointing_total,
        "skipped_no_duration": skipped_no_duration,
        "mean_per_clip_frame_auc": float(np.mean(per_clip_auc)) if per_clip_auc else float("nan"),
        "mean_per_clip_frame_ap": float(np.mean(per_clip_ap)) if per_clip_ap else float("nan"),
        "n_clips_frame_auc": len(per_clip_auc),
        "pointing_game_acc": pointing_hits / pointing_total if pointing_total else float("nan"),
        "pointing_game_random": float(np.mean(rand_pointing)) if rand_pointing else float("nan"),
        **ap_at_tiou,
    }
    with open(os.path.join(out_dir, "localization.json"), "w") as f:
        json.dump(results, f, indent=2)

    print("\n" + "=" * 60)
    print("PINPOINT ATTENTION LOCALIZATION vs LAV-DF GT (Tier-1)")
    print("=" * 60)
    print(f"  clips evaluated            : {results['n_clips']} "
          f"(skipped {skipped_no_duration} w/o duration)")
    print(f"  mean per-clip frame AUC    : {results['mean_per_clip_frame_auc']:.4f} "
          f"(over {results['n_clips_frame_auc']} mixed clips)")
    print(f"  mean per-clip frame AP     : {results['mean_per_clip_frame_ap']:.4f}")
    print(f"  pointing-game accuracy     : {results['pointing_game_acc']:.4f} "
          f"(random {results['pointing_game_random']:.4f})")
    for t in args.tiou:
        print(f"  AP@tIoU={t:<4}              : {results[f'AP@{t}']:.4f}")
    print(f"\nSaved localization.json to: {out_dir}")
    print("=" * 60)


if __name__ == "__main__":
    main()
