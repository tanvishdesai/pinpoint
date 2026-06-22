"""
crossdataset_train.py  --  Tier-2 (honest generalization).
==========================================================
Trains PinPoint on ONE source dataset and tests on the OTHER, to report the
true cross-dataset number the merged-pool 97% accuracy hides:

    --train-on lavdf        --test-on fakeavceleb
    --train-on fakeavceleb  --test-on lavdf

The architecture, synchronization loss and curriculum are identical to
PinPoint.py (reused from pinpoint_core); only the data splits change. The model
is trained from scratch (ImageNet-pretrained ResNet backbone) and evaluated with
AUC / AP / EER on the held-out dataset's test split.

USAGE (Kaggle, P100):
    python crossdataset_train.py --train-on lavdf --test-on fakeavceleb --epochs 8
    python crossdataset_train.py --train-on fakeavceleb --test-on lavdf --epochs 8
"""

import os
import json
import time
import random
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
from torch.cuda.amp import GradScaler, autocast
from tqdm import tqdm

from pinpoint_core import (CoreConfig, PinpointTransformer, SynchronizationLoss,
                           load_lavdf_index, source_of, set_seed)
import metrics_utils as M


# --------------------------- training dataset -----------------------------
class CrossTrainDataset(Dataset):
    """Source-filtered dataset with anti-shortcut augmentation (train) and the
    real-sample temporal-offset injection that drives the offset head + the
    synced-mask for the synchronization loss."""

    def __init__(self, config, split, source_filter, augment, lavdf_index):
        self.config = config
        self.split = split
        self.augment = augment
        with open(config.METADATA_PATH) as f:
            all_meta = json.load(f)
        self.samples = []
        for item in all_meta:
            if item.get("split") != split:
                continue
            if source_filter is not None and source_of(item, lavdf_index) != source_filter:
                continue
            label = item.get("label") or ("fake" if item.get("n_fakes", 0) else "real")
            v = os.path.join(config.DATA_DIRECTORY, item["preprocessed_video_path"])
            a = os.path.join(config.DATA_DIRECTORY, item["preprocessed_audio_path"])
            if os.path.exists(v) and os.path.exists(a):
                self.samples.append({"video_path": v, "audio_path": a,
                                     "is_fake": label == "fake"})
        self.normalize = transforms.Normalize(mean=config.NORM_MEAN, std=config.NORM_STD)
        print(f"[CrossTrain] {split}/{source_filter}: {len(self.samples)} samples")

    def __len__(self):
        return len(self.samples)

    def _fix_audio(self, audio):
        nm = self.config.NUM_MFCC
        if audio.ndim == 3:
            if audio.shape[0] == nm:
                audio = audio.mean(dim=1).transpose(0, 1)
            elif audio.shape[1] == nm:
                audio = audio.mean(dim=0).transpose(0, 1)
            elif audio.shape[2] == nm:
                audio = audio.mean(dim=1)
            else:
                return None
        elif audio.ndim == 2 and audio.shape[0] == nm and audio.shape[1] != nm:
            audio = audio.transpose(0, 1)
        elif audio.ndim != 2:
            return None
        return audio

    def __getitem__(self, idx):
        try:
            info = self.samples[idx]
            video = torch.load(info["video_path"]).to(torch.float32) / 255.0
            audio = self._fix_audio(torch.load(info["audio_path"]))
            if audio is None:
                return None
            video = self.normalize(video)
            is_fake = info["is_fake"]
            offset_label = self.config.MAX_OFFSET
            if self.split == "train" and not is_fake and random.random() < 0.5:
                off = random.randint(-self.config.MAX_OFFSET, self.config.MAX_OFFSET)
                if off != 0:
                    offset_label += off
                    ao = off * self.config.MFCC_FRAMES_PER_VIDEO_FRAME
                    if ao > 0:
                        audio = audio[ao:]
                    elif ao < 0:
                        audio = audio[:ao]
            return {"video": video, "audio": audio, "label": 1.0 if is_fake else 0.0,
                    "is_fake": is_fake, "offset_label": offset_label}
        except Exception as e:
            print(f"[CrossTrain] skip {idx}: {e}")
            return None


def train_collate(batch, num_frames):
    batch = [b for b in batch if b is not None]
    if not batch:
        return None
    max_a = max(b["audio"].shape[0] for b in batch)
    V, A, Mk, L, O, Fk = [], [], [], [], [], []
    for it in batch:
        v = it["video"]; n = v.shape[0]
        if n > num_frames:
            v = v[torch.linspace(0, n - 1, num_frames, dtype=torch.long)]
        elif n < num_frames:
            v = torch.cat([v, torch.zeros(num_frames - n, *v.shape[1:], dtype=v.dtype)], 0)
        V.append(v)
        Mk.append(torch.zeros(num_frames, dtype=torch.bool))
        a = it["audio"]; A.append(F.pad(a, (0, 0, 0, max_a - a.shape[0])))
        L.append(it["label"]); O.append(it["offset_label"]); Fk.append(it["is_fake"])
    return {"video": torch.stack(V), "audio": torch.stack(A), "video_mask": torch.stack(Mk),
            "label": torch.tensor(L, dtype=torch.float32),
            "offset_label": torch.tensor(O, dtype=torch.long),
            "is_fake": torch.tensor(Fk, dtype=torch.bool)}


def _mask_frames(video, lo=0.5, hi=0.8):
    out = video.clone()
    b, t = out.shape[0], out.shape[1]
    for i in range(b):
        k = int(t * random.uniform(lo, hi))
        out[i, torch.randperm(t)[:k]] = 0
    return out


@torch.no_grad()
def evaluate_auc(model, loader, device):
    model.eval()
    probs, labels = [], []
    for batch in tqdm(loader, desc="eval", leave=False):
        if batch is None:
            continue
        with autocast():
            logits, _, _ = model(batch["video"].to(device), batch["audio"].to(device),
                                 batch["video_mask"].to(device))
        probs.extend(torch.sigmoid(logits).squeeze(1).float().cpu().numpy().tolist())
        labels.extend(batch["label"].numpy().astype(int).tolist())
    auc, ap = M.auc_ap(labels, probs)
    eer, thr = M.eer(labels, probs)
    acc = M.accuracy_at(labels, probs, 0.5)
    return {"auc": auc, "ap": ap, "eer": eer, "acc@0.5": acc,
            "n": len(labels), "n_fake": int(np.sum(labels))}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train-on", required=True, choices=["lavdf", "fakeavceleb"])
    ap.add_argument("--test-on", required=True, choices=["lavdf", "fakeavceleb"])
    ap.add_argument("--epochs", type=int, default=8)
    ap.add_argument("--batch-size", type=int, default=4)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--phase1", type=int, default=1, help="video-mask epochs")
    ap.add_argument("--phase2", type=int, default=2, help="sync-focus epochs")
    args = ap.parse_args()

    set_seed(42)
    cfg = CoreConfig()
    cfg.NUM_FRAMES = 30  # training from scratch: fix the positional-encoder size
    out_dir = os.path.join(cfg.OUTPUT_DIR, f"crossdataset_{args.train_on}_to_{args.test_on}")
    os.makedirs(out_dir, exist_ok=True)
    device = cfg.DEVICE
    lavdf_index = load_lavdf_index(cfg.LAVDF_METADATA_PATH)

    collate = lambda b: train_collate(b, cfg.NUM_FRAMES)
    train_ds = CrossTrainDataset(cfg, "train", args.train_on, augment=True, lavdf_index=lavdf_index)
    # validation = same-source dev split; test = OTHER source test split
    val_ds = CrossTrainDataset(cfg, "dev", args.train_on, augment=False, lavdf_index=lavdf_index)
    test_ds = CrossTrainDataset(cfg, "test", args.test_on, augment=False, lavdf_index=lavdf_index)
    if len(train_ds) == 0 or len(test_ds) == 0:
        print("Empty train or test split for the requested sources -- check source tagging.")
        return

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                              collate_fn=collate, num_workers=2, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=8, shuffle=False, collate_fn=collate, num_workers=2)
    test_loader = DataLoader(test_ds, batch_size=8, shuffle=False, collate_fn=collate, num_workers=2)

    model = PinpointTransformer(cfg).to(device)
    n_real = sum(1 for s in train_ds.samples if not s["is_fake"])
    n_fake = sum(1 for s in train_ds.samples if s["is_fake"])
    pos_weight = torch.tensor([n_real / max(n_fake, 1)], device=device) if n_fake and n_real else None
    loss_fns = {"cls": nn.BCEWithLogitsLoss(pos_weight=pos_weight),
                "off": nn.CrossEntropyLoss(),
                "attn": SynchronizationLoss(cfg)}
    optim = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=cfg.WEIGHT_DECAY if hasattr(cfg, "WEIGHT_DECAY") else 1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(optim, T_max=args.epochs * max(len(train_loader), 1))
    scaler = GradScaler()

    best_auc, best_path = -1.0, os.path.join(out_dir, "model.pth")
    t0 = time.time()
    for epoch in range(args.epochs):
        phase = 1 if epoch < args.phase1 else (2 if epoch < args.phase1 + args.phase2 else 3)
        model.train()
        pbar = tqdm(train_loader, desc=f"E{epoch+1} P{phase}", leave=False)
        for batch in pbar:
            if batch is None:
                continue
            video = batch["video"].to(device); audio = batch["audio"].to(device)
            vmask = batch["video_mask"].to(device)
            yl = batch["label"].to(device); ol = batch["offset_label"].to(device)
            fk = batch["is_fake"].to(device)
            if phase == 1:
                video = _mask_frames(video)
            optim.zero_grad(set_to_none=True)
            with autocast():
                logits, off_logits, attn = model(video, audio, vmask)
                l_cls = loss_fns["cls"](logits.squeeze(1), yl)
                real_mask = ~fk
                l_off = loss_fns["off"](off_logits[real_mask], ol[real_mask]) if real_mask.sum() > 0 else torch.tensor(0.0, device=device)
                if phase == 2:
                    l_off = l_off * 0.1
                synced = (ol == cfg.MAX_OFFSET) & real_mask
                l_attn = loss_fns["attn"](attn, synced)
                loss = cfg.W_CLASSIFICATION * l_cls + cfg.W_OFFSET * l_off + cfg.W_ATTENTION * l_attn
            if not torch.isfinite(loss):
                continue
            scaler.scale(loss).backward()
            scaler.unscale_(optim)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optim); scaler.update(); sched.step()
            pbar.set_postfix({"loss": f"{loss.item():.3f}"})

        val = evaluate_auc(model, val_loader, device) if len(val_ds) else {"auc": float("nan")}
        print(f"Epoch {epoch+1}: in-domain val AUC = {val['auc']:.4f}")
        if val["auc"] == val["auc"] and val["auc"] > best_auc:
            best_auc = val["auc"]
            torch.save(model.state_dict(), best_path)

    if os.path.exists(best_path):
        model.load_state_dict(torch.load(best_path, map_location=device))
    cross = evaluate_auc(model, test_loader, device)
    results = {"train_on": args.train_on, "test_on": args.test_on,
               "epochs": args.epochs, "minutes": (time.time() - t0) / 60.0,
               "in_domain_val_auc": best_auc, "cross_dataset": cross}
    with open(os.path.join(out_dir, "crossdataset.json"), "w") as f:
        json.dump(results, f, indent=2)

    print("\n" + "=" * 60)
    print(f"CROSS-DATASET: train={args.train_on} -> test={args.test_on}")
    print("=" * 60)
    print(f"  in-domain (dev) best AUC : {best_auc:.4f}")
    print(f"  CROSS test AUC           : {cross['auc']:.4f}")
    print(f"  CROSS test AP            : {cross['ap']:.4f}")
    print(f"  CROSS test EER           : {cross['eer']:.4f}")
    print(f"  CROSS test Acc@0.5       : {cross['acc@0.5']:.4f}  (n={cross['n']}, fake={cross['n_fake']})")
    print(f"  model + metrics saved to : {out_dir}")
    print("=" * 60)


if __name__ == "__main__":
    main()
