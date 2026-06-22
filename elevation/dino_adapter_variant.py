"""
dino_adapter_variant.py  --  OPTIONAL backbone ablation (Tier-2 upside).
=======================================================================
Swaps the ResNet-18 video backbone for a FROZEN DINOv2-S encoder + a small
trainable adapter, keeping everything else (MFCC-GRU audio, gated cross-attention
fusion, synchronization loss, classification/offset heads) identical. Only the
adapter + audio + fusion + heads are trained.

Purpose (from the elevation plan): show the synchronization mechanism is
BACKBONE-AGNOSTIC -- the sync-supervised, faithful-attention story rides on a
SOTA-class encoder, and AUC moves up -- so the contribution does not depend on
ResNet specifically.

DINOv2-S is pulled from torch.hub (facebookresearch/dinov2, dinov2_vits14, 384-d
CLS token). Frames are resized to a multiple of 14. Features are extracted under
no_grad, so this is cheap to train (only the adapter/fusion/heads see gradients).

It trains + tests on the MERGED set (source_filter=None) for a direct
ResNet-vs-DINOv2 comparison on the same in-domain benchmark.

USAGE (Kaggle, 1x T4; needs internet for the torch.hub download, or a cached
DINOv2 weights dataset):
    python dino_adapter_variant.py --epochs 8
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
from torch.utils.data import DataLoader
from torch.cuda.amp import GradScaler, autocast
from tqdm import tqdm

from pinpoint_core import (CoreConfig, AudioFeatureExtractor, GatedCrossAttentionBlock,
                           get_sinusoidal_embeddings, SynchronizationLoss,
                           load_lavdf_index, set_seed)
from crossdataset_train import CrossTrainDataset, train_collate, evaluate_auc, _mask_frames
import metrics_utils as M


class DinoVideoExtractor(nn.Module):
    def __init__(self, embed_dim, image_size=224, hub_name="dinov2_vits14"):
        super().__init__()
        self.image_size = image_size
        self.dino = torch.hub.load("facebookresearch/dinov2", hub_name)
        for p in self.dino.parameters():
            p.requires_grad = False
        self.dino.eval()
        feat_dim = getattr(self.dino, "embed_dim", 384)
        self.adapter = nn.Sequential(
            nn.Linear(feat_dim, embed_dim), nn.GELU(),
            nn.LayerNorm(embed_dim), nn.Linear(embed_dim, embed_dim),
        )

    def forward(self, x):
        b, t, c, h, w = x.shape
        x = x.view(b * t, c, h, w)
        if (h, w) != (self.image_size, self.image_size):
            x = F.interpolate(x, size=(self.image_size, self.image_size),
                              mode="bilinear", align_corners=False)
        with torch.no_grad():
            feats = self.dino(x)            # [b*t, feat_dim] CLS token
        feats = self.adapter(feats)
        return feats.view(b, t, -1)


class DinoPinpoint(nn.Module):
    """PinpointTransformer with the DINOv2 video backbone."""

    def __init__(self, config, image_size=224):
        super().__init__()
        self.config = config
        self.video_extractor = DinoVideoExtractor(config.EMBED_DIM, image_size=image_size)
        self.audio_extractor = AudioFeatureExtractor(config.NUM_MFCC, config.EMBED_DIM)
        self.video_pos_encoder = nn.Parameter(torch.randn(1, config.NUM_FRAMES, config.EMBED_DIM))
        self.gated_attention_layers = nn.ModuleList(
            [GatedCrossAttentionBlock(config.EMBED_DIM, config.NUM_HEADS, config.DROPOUT)
             for _ in range(config.NUM_LAYERS)])
        self.classification_head = nn.Linear(config.EMBED_DIM, 1)
        self.offset_head = nn.Linear(config.EMBED_DIM, 2 * config.MAX_OFFSET + 1)

    def forward(self, video, audio, video_mask=None):
        video_feat = self.video_extractor(video)
        audio_feat = self.audio_extractor(audio)
        video_feat = video_feat + self.video_pos_encoder[:, :video_feat.size(1), :]
        audio_pos = get_sinusoidal_embeddings(audio_feat.size(1), self.config.EMBED_DIM).to(audio_feat.device)
        audio_feat = audio_feat + audio_pos
        last_map = None
        for layer in self.gated_attention_layers:
            audio_feat, last_map = layer(audio_feat, video_feat, video_mask)
        pooled = audio_feat.mean(dim=1)
        return self.classification_head(pooled), self.offset_head(pooled), last_map


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=8)
    ap.add_argument("--batch-size", type=int, default=4)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--image-size", type=int, default=224)
    ap.add_argument("--phase1", type=int, default=1)
    ap.add_argument("--phase2", type=int, default=2)
    args = ap.parse_args()

    set_seed(42)
    cfg = CoreConfig()
    cfg.NUM_FRAMES = 30
    out_dir = os.path.join(cfg.OUTPUT_DIR, "dino_adapter")
    os.makedirs(out_dir, exist_ok=True)
    device = cfg.DEVICE
    lavdf_index = load_lavdf_index(cfg.LAVDF_METADATA_PATH)
    collate = lambda b: train_collate(b, cfg.NUM_FRAMES)

    train_ds = CrossTrainDataset(cfg, "train", None, augment=True, lavdf_index=lavdf_index)
    val_ds = CrossTrainDataset(cfg, "dev", None, augment=False, lavdf_index=lavdf_index)
    test_ds = CrossTrainDataset(cfg, "test", None, augment=False, lavdf_index=lavdf_index)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                              collate_fn=collate, num_workers=2, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=8, shuffle=False, collate_fn=collate, num_workers=2)
    test_loader = DataLoader(test_ds, batch_size=8, shuffle=False, collate_fn=collate, num_workers=2)

    model = DinoPinpoint(cfg, image_size=args.image_size).to(device)
    n_real = sum(1 for s in train_ds.samples if not s["is_fake"])
    n_fake = sum(1 for s in train_ds.samples if s["is_fake"])
    pos_weight = torch.tensor([n_real / max(n_fake, 1)], device=device) if n_fake and n_real else None
    loss_fns = {"cls": nn.BCEWithLogitsLoss(pos_weight=pos_weight),
                "off": nn.CrossEntropyLoss(), "attn": SynchronizationLoss(cfg)}
    trainable = [p for p in model.parameters() if p.requires_grad]
    print(f"Trainable params: {sum(p.numel() for p in trainable)/1e6:.2f}M "
          f"(DINOv2 backbone frozen)")
    optim = torch.optim.AdamW(trainable, lr=args.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(optim, T_max=args.epochs * max(len(train_loader), 1))
    scaler = GradScaler()

    best_auc, best_path = -1.0, os.path.join(out_dir, "model.pth")
    t0 = time.time()
    for epoch in range(args.epochs):
        phase = 1 if epoch < args.phase1 else (2 if epoch < args.phase1 + args.phase2 else 3)
        model.train()
        for batch in tqdm(train_loader, desc=f"E{epoch+1} P{phase}", leave=False):
            if batch is None:
                continue
            video = batch["video"].to(device); audio = batch["audio"].to(device)
            vmask = batch["video_mask"].to(device); yl = batch["label"].to(device)
            ol = batch["offset_label"].to(device); fk = batch["is_fake"].to(device)
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
            torch.nn.utils.clip_grad_norm_(trainable, 1.0)
            scaler.step(optim); scaler.update(); sched.step()

        val = evaluate_auc(model, val_loader, device) if len(val_ds) else {"auc": float("nan")}
        print(f"Epoch {epoch+1}: val AUC = {val['auc']:.4f}")
        if val["auc"] == val["auc"] and val["auc"] > best_auc:
            best_auc = val["auc"]
            torch.save(model.state_dict(), best_path)

    if os.path.exists(best_path):
        model.load_state_dict(torch.load(best_path, map_location=device))
    test = evaluate_auc(model, test_loader, device)
    results = {"backbone": "dinov2_vits14", "image_size": args.image_size,
               "epochs": args.epochs, "minutes": (time.time() - t0) / 60.0,
               "best_val_auc": best_auc, "test": test}
    with open(os.path.join(out_dir, "dino_adapter.json"), "w") as f:
        json.dump(results, f, indent=2)

    print("\n" + "=" * 60)
    print("DINOv2-S + ADAPTER VARIANT (backbone-agnostic check)")
    print("=" * 60)
    print(f"  test AUC : {test['auc']:.4f}")
    print(f"  test AP  : {test['ap']:.4f}")
    print(f"  test EER : {test['eer']:.4f}")
    print(f"  test Acc : {test['acc@0.5']:.4f}  (n={test['n']}, fake={test['n_fake']})")
    print(f"  saved to : {out_dir}")
    print("=" * 60)


if __name__ == "__main__":
    main()
