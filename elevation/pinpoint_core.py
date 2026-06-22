"""
pinpoint_core.py
================
Shared, dependency-light core for the PinPoint *elevation* evaluation suite.

This module factors the pieces that every elevation script needs out of the
monolithic ``PinPoint.py`` so the new evaluation scripts can ``import`` them
without pulling in the heavy XAI stack (lime / shap / seaborn / ...).

What lives here:
  * ``CoreConfig``          - paths + model hyper-parameters (edit the PATHS block).
  * The model               - VideoFeatureExtractor / AudioFeatureExtractor /
                              GatedCrossAttentionBlock / PinpointTransformer.
                              (identical architecture to PinPoint.py, with the
                              ResNet inplace-ops XAI patch retained so Grad-CAM /
                              IG work in faithfulness_eval.py.)
  * ``SynchronizationLoss`` - reused by crossdataset_train.py / dino variant.
  * ``EvalDataset`` / ``collate_fn`` - a metadata-carrying dataset that exposes
                              ``fake_periods`` / ``n_fakes`` / ``modify_video`` /
                              ``modify_audio`` / ``source`` / ``file`` so the
                              localization and per-category metrics can be
                              computed without a second data load.
  * ``load_model``          - loads a checkpoint and AUTO-DETECTS ``NUM_FRAMES``
                              and ``NUM_MFCC`` from the saved weights, so you do
                              not have to remember whether the model was trained
                              at 30 or 64 frames.
  * ``source_of`` / GT helpers.

Nothing in this file runs on import; it is pure library code.
"""

import os
import json
import math
import random
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset
import torchvision.models as models
import torchvision.transforms as transforms
from torchvision.models.resnet import BasicBlock
import types


# =================================================================================
# CONFIG  -- edit the PATHS block for your Kaggle run
# =================================================================================
class CoreConfig:
    # ---------------------------- PATHS (EDIT ME) --------------------------------
    # The merged (LAV-DF + FakeAVCeleb) preprocessed data + unified metadata,
    # exactly as used by PinPoint.py.
    DATA_DIRECTORY = "/kaggle/input/new-model-unified-pre-processing/preprocessed_data"
    METADATA_PATH = "/kaggle/input/new-model-unified-pre-processing/preprocessed_data/unified_metadata.json"

    # The trained checkpoint produced by PinPoint.py.
    MODEL_PATH = "/kaggle/input/pinpoint-checkpoint/best_pinpoint_model_antisocial.pth"

    # OPTIONAL: the ORIGINAL LAV-DF metadata.json. Used ONLY as a fallback to
    # recover per-segment manipulation timestamps (``fake_periods``) and to tag a
    # clip's source dataset when the merged metadata does not already carry them.
    # Leave as-is if available on Kaggle; set to None to disable the join.
    LAVDF_METADATA_PATH = "/kaggle/input/localized-audio-visual-deepfake-dataset-lav-df/LAV-DF/metadata.json"

    # Where elevation scripts write their outputs.
    OUTPUT_DIR = "/kaggle/working/elevation_outputs"
    # -----------------------------------------------------------------------------

    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

    # Data / preprocessing constants (must match the preprocessing that built the
    # tensors). NUM_FRAMES is auto-detected from the checkpoint in load_model();
    # the value here is only a default for dataset padding when no checkpoint is
    # involved (e.g. cross-dataset training from scratch).
    NUM_FRAMES = 30
    VIDEO_SIZE = (128, 128)
    NUM_MFCC = 13
    NORM_MEAN = [0.485, 0.456, 0.406]
    NORM_STD = [0.229, 0.224, 0.225]

    # Model architecture (must match the trained checkpoint).
    EMBED_DIM = 256
    NUM_HEADS = 8
    NUM_LAYERS = 3
    DROPOUT = 0.1
    MAX_OFFSET = 5
    MFCC_FRAMES_PER_VIDEO_FRAME = 2

    # Synchronization-loss parameters (only used when (re)training).
    SYNC_LOSS_BANDWIDTH = 2
    W_SYNC_DIRECT = 1.0
    W_SYNC_DOMINANCE = 0.5
    W_SYNC_SMOOTHNESS = 0.2
    W_CLASSIFICATION = 1.0
    W_OFFSET = 0.5
    W_ATTENTION = 3.0

    # LAV-DF is 25 fps; used to convert fake_periods (seconds) to frame indices.
    LAVDF_FPS = 25.0


# =================================================================================
# MODEL  (architecture identical to PinPoint.py; ResNet inplace ops patched)
# =================================================================================
def _new_basic_block_forward(self, x):
    """Non-inplace BasicBlock.forward so gradient-based XAI (IG/Grad-CAM) works."""
    identity = x
    out = self.conv1(x)
    out = self.bn1(out)
    out = self.relu(out)
    out = self.conv2(out)
    out = self.bn2(out)
    if self.downsample is not None:
        identity = self.downsample(x)
    out = out + identity          # non-inplace
    out = self.relu(out)
    return out


class VideoFeatureExtractor(nn.Module):
    def __init__(self, embed_dim, pretrained=True):
        super().__init__()
        weights = models.ResNet18_Weights.DEFAULT if pretrained else None
        resnet = models.resnet18(weights=weights)
        for module in resnet.modules():
            if isinstance(module, nn.ReLU):
                module.inplace = False
            elif isinstance(module, BasicBlock):
                module.forward = types.MethodType(_new_basic_block_forward, module)
        modules = list(resnet.children())[:-2]
        self.feature_extractor = nn.Sequential(*modules)
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.projection = nn.Linear(resnet.fc.in_features, embed_dim)
        for param in self.feature_extractor[:6].parameters():
            param.requires_grad = False

    def forward(self, x):
        b, t, c, h, w = x.shape
        x = x.view(b * t, c, h, w)
        features = self.feature_extractor(x)
        pooled = self.pool(features).view(b * t, -1)
        projected = self.projection(pooled)
        return projected.view(b, t, -1)


class AudioFeatureExtractor(nn.Module):
    def __init__(self, num_mfcc, embed_dim):
        super().__init__()
        self.conv1 = nn.Conv1d(num_mfcc, 64, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(64, 128, kernel_size=3, padding=1)
        self.ln = nn.LayerNorm(128)
        self.gru = nn.GRU(input_size=128, hidden_size=embed_dim, batch_first=True)

    def forward(self, x):
        x = x.transpose(1, 2)
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = x.transpose(1, 2)
        x = self.ln(x)
        output, _ = self.gru(x)
        return output


def get_sinusoidal_embeddings(n_position, d_hid):
    def angle_vec(position):
        return [position / np.power(10000, 2 * (j // 2) / d_hid) for j in range(d_hid)]
    table = np.array([angle_vec(p) for p in range(n_position)])
    table[:, 0::2] = np.sin(table[:, 0::2])
    table[:, 1::2] = np.cos(table[:, 1::2])
    return torch.FloatTensor(table).unsqueeze(0)


class GatedCrossAttentionBlock(nn.Module):
    def __init__(self, embed_dim, num_heads, dropout):
        super().__init__()
        self.ln1 = nn.LayerNorm(embed_dim)
        self.ln2 = nn.LayerNorm(embed_dim)
        self.audio_to_video_attn = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout, batch_first=True)
        self.gate = nn.Sequential(nn.Linear(embed_dim, embed_dim), nn.Sigmoid())
        self.self_attn = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout, batch_first=True)
        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 4), nn.GELU(),
            nn.Dropout(dropout), nn.Linear(embed_dim * 4, embed_dim),
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, audio_feat, video_feat, video_mask=None):
        audio_norm = self.ln1(audio_feat)
        video_norm = self.ln1(video_feat)
        cross_out, cross_map = self.audio_to_video_attn(
            query=audio_norm, key=video_norm, value=video_norm, key_padding_mask=video_mask)
        audio_feat = audio_feat + self.dropout(cross_out)
        gated = audio_feat * self.gate(audio_feat)
        gated_norm = self.ln2(gated)
        self_out, _ = self.self_attn(gated_norm, gated_norm, gated_norm)
        gated = gated + self.dropout(self_out)
        gated_norm2 = self.ln2(gated)
        ffn_out = self.ffn(gated_norm2)
        return gated + self.dropout(ffn_out), cross_map


class PinpointTransformer(nn.Module):
    def __init__(self, config, pretrained_video=True):
        super().__init__()
        self.config = config
        self.video_extractor = VideoFeatureExtractor(config.EMBED_DIM, pretrained=pretrained_video)
        self.audio_extractor = AudioFeatureExtractor(config.NUM_MFCC, config.EMBED_DIM)
        self.video_pos_encoder = nn.Parameter(torch.randn(1, config.NUM_FRAMES, config.EMBED_DIM))
        self.gated_attention_layers = nn.ModuleList(
            [GatedCrossAttentionBlock(config.EMBED_DIM, config.NUM_HEADS, config.DROPOUT)
             for _ in range(config.NUM_LAYERS)])
        self.classification_head = nn.Linear(config.EMBED_DIM, 1)
        self.offset_head = nn.Linear(config.EMBED_DIM, 2 * config.MAX_OFFSET + 1)
        # modality dropout is disabled at eval time (self.training == False)
        self.modality_dropout_prob = getattr(config, "MODALITY_DROPOUT_PROB", 0.0)

    def forward(self, video, audio, video_mask=None):
        video_feat = self.video_extractor(video)
        audio_feat = self.audio_extractor(audio)

        if self.training and self.modality_dropout_prob > 0:
            v_mask = torch.ones(video_feat.size(0), 1, 1, device=video_feat.device)
            a_mask = torch.ones(audio_feat.size(0), 1, 1, device=audio_feat.device)
            for i in range(video.size(0)):
                if random.random() < self.modality_dropout_prob:
                    if random.random() < 0.5:
                        v_mask[i] = 0
                    else:
                        a_mask[i] = 0
            video_feat = video_feat * v_mask
            audio_feat = audio_feat * a_mask

        video_feat = video_feat + self.video_pos_encoder[:, :video_feat.size(1), :]
        audio_len = audio_feat.size(1)
        audio_pos = get_sinusoidal_embeddings(audio_len, self.config.EMBED_DIM).to(audio_feat.device)
        audio_feat = audio_feat + audio_pos

        last_map = None
        for layer in self.gated_attention_layers:
            audio_feat, last_map = layer(audio_feat, video_feat, video_mask)
        pooled = audio_feat.mean(dim=1)
        return self.classification_head(pooled), self.offset_head(pooled), last_map


# =================================================================================
# SYNCHRONIZATION LOSS  (reused for (re)training)
# =================================================================================
class SynchronizationLoss(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config

    def _create_sync_target(self, b, audio_len, video_len, device):
        target = torch.zeros(audio_len, video_len, device=device)
        for i in range(audio_len):
            center = int(i / self.config.MFCC_FRAMES_PER_VIDEO_FRAME)
            for j in range(video_len):
                dist = abs(j - center)
                target[i, j] = -(dist ** 2) / (2 * (self.config.SYNC_LOSS_BANDWIDTH ** 2))
        target = torch.exp(target)
        target = target / (target.sum(dim=1, keepdim=True) + 1e-8)
        return target.unsqueeze(0).repeat(b, 1, 1)

    def _diagonal_dominance_loss(self, attn):
        b, audio_len, video_len = attn.shape
        mask = torch.zeros_like(attn)
        for i in range(audio_len):
            center = int(i / self.config.MFCC_FRAMES_PER_VIDEO_FRAME)
            start = max(0, center - self.config.SYNC_LOSS_BANDWIDTH)
            end = min(video_len, center + self.config.SYNC_LOSS_BANDWIDTH + 1)
            mask[:, i, start:end] = 1.0
        on = (attn * mask).sum(dim=(-1, -2))
        total = attn.sum(dim=(-1, -2))
        return (1.0 - (on / (total + 1e-8))).mean()

    def _temporal_smoothness_loss(self, attn):
        a = F.mse_loss(attn[:, 1:, :], attn[:, :-1, :])
        v = F.mse_loss(attn[:, :, 1:], attn[:, :, :-1])
        return a + v

    def forward(self, attention_map, is_synced_mask):
        if attention_map is None or not is_synced_mask.any():
            return torch.tensor(0.0, device=self.config.DEVICE)
        synced = attention_map[is_synced_mask]
        b_s, t_a, t_v = synced.shape
        target = self._create_sync_target(b_s, t_a, t_v, synced.device)
        direct = F.mse_loss(synced, target)
        dominance = self._diagonal_dominance_loss(synced)
        smooth = self._temporal_smoothness_loss(synced)
        return (self.config.W_SYNC_DIRECT * direct +
                self.config.W_SYNC_DOMINANCE * dominance +
                self.config.W_SYNC_SMOOTHNESS * smooth)


# =================================================================================
# SOURCE / GROUND-TRUTH HELPERS
# =================================================================================
def _basename(path_or_file):
    return os.path.splitext(os.path.basename(str(path_or_file)))[0]


def load_lavdf_index(lavdf_metadata_path):
    """Return {basename: original_lavdf_item} for fake_periods / source fallback."""
    if not lavdf_metadata_path or not os.path.exists(lavdf_metadata_path):
        return {}
    with open(lavdf_metadata_path, "r") as f:
        meta = json.load(f)
    return {_basename(item.get("file", "")): item for item in meta}


def source_of(item, lavdf_index=None):
    """Best-effort source-dataset tag: 'lavdf' or 'fakeavceleb'.

    Priority: explicit field in the unified metadata -> presence in the LAV-DF
    index -> filename heuristic (FakeAVCeleb ids look like 'id00076' / contain
    'FakeAVCeleb'; LAV-DF files are zero-padded numeric like '000123').
    """
    for key in ("source", "dataset", "origin_dataset"):
        v = item.get(key)
        if isinstance(v, str) and v:
            low = v.lower()
            if "lav" in low:
                return "lavdf"
            if "fake" in low or "avceleb" in low or "celeb" in low:
                return "fakeavceleb"
    base = _basename(item.get("file", item.get("preprocessed_video_path", "")))
    if lavdf_index and base in lavdf_index:
        return "lavdf"
    # LAV-DF basenames are purely numeric; FakeAVCeleb are not.
    if base.isdigit():
        return "lavdf"
    return "fakeavceleb"


def fake_periods_of(item, lavdf_index=None):
    """Return list of [start_sec, end_sec] manipulation segments, or [] if none."""
    fp = item.get("fake_periods")
    if fp is None and lavdf_index is not None:
        base = _basename(item.get("file", item.get("preprocessed_video_path", "")))
        ref = lavdf_index.get(base)
        if ref is not None:
            fp = ref.get("fake_periods")
    if not fp:
        return []
    out = []
    for seg in fp:
        if isinstance(seg, (list, tuple)) and len(seg) >= 2:
            out.append([float(seg[0]), float(seg[1])])
    return out


def manipulation_type(item, lavdf_index=None):
    """Coarse manipulation category for per-category AUC:
    'real', 'audio_only', 'video_only', 'both', or 'fake_unknown'."""
    label = item.get("label")
    n_fakes = item.get("n_fakes", None)
    is_fake = (label == "fake") if label is not None else (bool(n_fakes) if n_fakes is not None else None)
    ref = item
    if (item.get("modify_video") is None and item.get("modify_audio") is None
            and lavdf_index is not None):
        base = _basename(item.get("file", item.get("preprocessed_video_path", "")))
        ref = lavdf_index.get(base, item)
    mv = ref.get("modify_video")
    ma = ref.get("modify_audio")
    if is_fake is False:
        return "real"
    if mv is None and ma is None:
        return "fake_unknown" if is_fake else "real"
    if mv and ma:
        return "both"
    if ma and not mv:
        return "audio_only"
    if mv and not ma:
        return "video_only"
    return "fake_unknown"


# =================================================================================
# DATASET  (metadata-carrying; works on the merged unified_metadata.json)
# =================================================================================
class EvalDataset(Dataset):
    """Loads preprocessed (video, audio) tensors for a split and carries the
    metadata needed by the elevation metrics.

    ``source_filter`` (None | 'lavdf' | 'fakeavceleb') restricts to one dataset.
    ``fake_only`` keeps only manipulated clips (used by localization).
    """

    def __init__(self, config, split="test", source_filter=None, fake_only=False,
                 require_fake_periods=False, max_samples=None, lavdf_index=None):
        self.config = config
        self.split = split
        self.lavdf_index = lavdf_index if lavdf_index is not None else load_lavdf_index(
            getattr(config, "LAVDF_METADATA_PATH", None))

        if not os.path.exists(config.METADATA_PATH):
            raise FileNotFoundError(f"Unified metadata not found: {config.METADATA_PATH}")
        with open(config.METADATA_PATH, "r") as f:
            all_meta = json.load(f)

        self.samples = []
        for item in all_meta:
            if item.get("split") != split:
                continue
            src = source_of(item, self.lavdf_index)
            if source_filter is not None and src != source_filter:
                continue

            label = item.get("label")
            if label is None:
                label = "fake" if item.get("n_fakes", 0) else "real"
            is_fake = (label == "fake")
            if fake_only and not is_fake:
                continue

            fperiods = fake_periods_of(item, self.lavdf_index)
            if require_fake_periods and not fperiods:
                continue

            vpath = os.path.join(config.DATA_DIRECTORY, item["preprocessed_video_path"])
            apath = os.path.join(config.DATA_DIRECTORY, item["preprocessed_audio_path"])
            if not (os.path.exists(vpath) and os.path.exists(apath)):
                continue

            self.samples.append({
                "video_path": vpath,
                "audio_path": apath,
                "label": 1.0 if is_fake else 0.0,
                "is_fake": is_fake,
                "source": src,
                "file": item.get("file", os.path.basename(vpath)),
                "fake_periods": fperiods,
                "n_fakes": item.get("n_fakes", len(fperiods)),
                "manip_type": manipulation_type(item, self.lavdf_index),
                "duration": item.get("duration", None),
                "video_frames": item.get("video_frames", item.get("frames", None)),
            })

        if max_samples is not None and len(self.samples) > max_samples:
            random.shuffle(self.samples)
            self.samples = self.samples[:max_samples]

        self.normalize = transforms.Normalize(mean=config.NORM_MEAN, std=config.NORM_STD)
        print(f"[EvalDataset] split={split} source_filter={source_filter} "
              f"fake_only={fake_only} -> {len(self.samples)} samples")

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
        elif audio.ndim == 2:
            if audio.shape[0] == nm and audio.shape[1] != nm:
                audio = audio.transpose(0, 1)
            elif audio.shape[1] != nm:
                return None
        else:
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
            return {
                "video": video, "audio": audio,
                "label": info["label"], "is_fake": info["is_fake"],
                "index": idx,
            }
        except Exception as e:
            print(f"[EvalDataset] skip idx {idx}: {e}")
            return None


def collate_fn(batch, num_frames):
    batch = [b for b in batch if b is not None]
    if not batch:
        return None
    max_audio = max(b["audio"].shape[0] for b in batch)
    videos, audios, masks, labels, fakes, idxs = [], [], [], [], [], []
    for item in batch:
        v = item["video"]
        n = v.shape[0]
        if n > num_frames:
            sel = torch.linspace(0, n - 1, num_frames, dtype=torch.long)
            v = v[sel]
        elif n < num_frames:
            pad = torch.zeros(num_frames - n, v.shape[1], v.shape[2], v.shape[3], dtype=v.dtype)
            v = torch.cat([v, pad], dim=0)
        videos.append(v)
        masks.append(torch.zeros(num_frames, dtype=torch.bool))
        a = item["audio"]
        audios.append(F.pad(a, (0, 0, 0, max_audio - a.shape[0]), "constant", 0))
        labels.append(item["label"])
        fakes.append(item["is_fake"])
        idxs.append(item["index"])
    return {
        "video": torch.stack(videos), "audio": torch.stack(audios),
        "video_mask": torch.stack(masks),
        "label": torch.tensor(labels, dtype=torch.float32),
        "is_fake": torch.tensor(fakes, dtype=torch.bool),
        "index": torch.tensor(idxs, dtype=torch.long),
    }


def make_collate(num_frames):
    """collate_fn closes over num_frames so DataLoader can pickle it (num_workers)."""
    import functools
    return functools.partial(collate_fn, num_frames=num_frames)


# =================================================================================
# CHECKPOINT LOADING  (auto-detects NUM_FRAMES / NUM_MFCC from the weights)
# =================================================================================
def infer_dims_from_state_dict(sd):
    """Return (num_frames, num_mfcc, embed_dim) inferred from a saved state_dict."""
    num_frames = None
    embed_dim = None
    num_mfcc = None
    if "video_pos_encoder" in sd:
        _, num_frames, embed_dim = sd["video_pos_encoder"].shape
    # conv1 of the audio extractor: weight shape [out=64, in=num_mfcc, k=3]
    for k in sd:
        if k.endswith("audio_extractor.conv1.weight"):
            num_mfcc = sd[k].shape[1]
            break
    return num_frames, num_mfcc, embed_dim


def load_model(config, model_path=None, device=None, verbose=True):
    """Build a PinpointTransformer matching the checkpoint and load it.

    The returned config is a (possibly) adjusted copy where NUM_FRAMES / NUM_MFCC
    / EMBED_DIM agree with the saved weights, so collate padding is correct.
    """
    device = device or config.DEVICE
    model_path = model_path or config.MODEL_PATH
    sd = torch.load(model_path, map_location="cpu")
    if isinstance(sd, dict) and "state_dict" in sd and "video_pos_encoder" not in sd:
        sd = sd["state_dict"]

    nf, nm, ed = infer_dims_from_state_dict(sd)
    if nf is not None and nf != config.NUM_FRAMES:
        if verbose:
            print(f"[load_model] auto-detected NUM_FRAMES={nf} (config had {config.NUM_FRAMES})")
        config.NUM_FRAMES = nf
    if nm is not None and nm != config.NUM_MFCC:
        if verbose:
            print(f"[load_model] auto-detected NUM_MFCC={nm} (config had {config.NUM_MFCC})")
        config.NUM_MFCC = nm
    if ed is not None and ed != config.EMBED_DIM:
        if verbose:
            print(f"[load_model] auto-detected EMBED_DIM={ed} (config had {config.EMBED_DIM})")
        config.EMBED_DIM = ed

    model = PinpointTransformer(config).to(device)
    missing, unexpected = model.load_state_dict(sd, strict=False)
    if verbose and (missing or unexpected):
        print(f"[load_model] missing={list(missing)[:4]}... unexpected={list(unexpected)[:4]}...")
    model.eval()
    if verbose:
        print(f"[load_model] loaded {model_path} on {device} "
              f"(NUM_FRAMES={config.NUM_FRAMES}, NUM_MFCC={config.NUM_MFCC})")
    return model, config


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
