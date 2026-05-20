# evaluate_and_visualize.py
#
# This script loads a pre-trained Pinpoint-Transformer model and evaluates it
# on a complete test dataset that may be scattered across multiple directories.
#
# TASKS:
# 1. Loads the specified model checkpoint.
# 2. Loads the test dataset from multiple source directories by dynamically finding all files.
# 3. Runs a full evaluation and prints a classification report and confusion matrix.
# 4. Iterates through every sample in the test set to generate and save:
#    - Cross-attention map visualizations.
#    - Grad-CAM comparison visualizations (Original vs. Standard vs. Multi-Layer).
#
# Author: AI Assistant
# Date: October 27, 2023
# ---------------------------------------------------------------------------------

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
from torchvision.transforms import v2 as T
import torchvision.models as models
import cv2
from sklearn.metrics import classification_report, confusion_matrix
import numpy as np
import json
import os
import random
import matplotlib.pyplot as plt
from tqdm import tqdm
import glob
import traceback
from collections import Counter

# =================================================================================
# 1. CONFIGURATION
# =================================================================================
class Config:
    # --- Paths for this Evaluation Script ---
    # !!! IMPORTANT: UPDATE THESE PATHS !!!
    MODEL_PATH = "./best_pinpoint_model_antisocial.pth"

    # The canonical metadata file from the original LAV-DF dataset.
    # This is now the "source of truth" for which files are in the test set.
    ORIGINAL_METADATA_PATH = "/kaggle/input/localized-audio-visual-deepfake-dataset-lav-df/LAV-DF/metadata.json"
    
    # List all directories containing your pre-processed 'test' data splits
    TEST_DATA_DIRECTORIES = [
        "/kaggle/input/la-df-testrin-1",
        "/kaggle/input/lav-df-testing-part-2",
        "/kaggle/input/lav-df-testing-part-3",
        "/kaggle/input/lavdf-testing-part-4"
    ]
    
    # Directory to save all generated images
    OUTPUT_VISUALS_DIR = "/kaggle/working/evaluation_outputs"

    # --- Evaluation Settings ---
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    METRICS_BATCH_SIZE = 8
    VISUALIZATION_BATCH_SIZE = 1
    
    # --- Sample Count Controller ---
    # Set to None to use all samples, or specify a number to limit samples for testing
    # Useful for quick testing, debugging, or when you have limited time/resources
    # Example: MAX_SAMPLES = 100 (will use first 100 samples for quick testing)
    #          MAX_SAMPLES = 1000 (will use first 1000 samples for medium testing)
    #          MAX_SAMPLES = None (will use all ~26,000 samples for full evaluation)
    MAX_SAMPLES = None  # Change this to limit samples for testing
    
    # --- Debugging Settings ---
    # Enable additional debugging to investigate suspiciously high performance
    DEBUG_MODE = False
    RANDOMIZE_SAMPLES = True  # Randomize sample order instead of taking first N samples
    SAVE_MISCLASSIFIED_SAMPLES = True  # Save examples of misclassified samples for inspection
    
    # For debugging, you can set a small sample size like:
    MAX_SAMPLES = 50000  # Test with 500 samples to see debug output 
    
    # --- Parameters from Original Training (MUST MATCH THE SAVED MODEL) ---
    NUM_FRAMES = 64
    VIDEO_SIZE = (128, 128)
    NUM_MFCC = 13
    NORM_MEAN = [0.485, 0.456, 0.406]
    NORM_STD = [0.229, 0.224, 0.225]
    EMBED_DIM = 256
    NUM_HEADS = 8
    NUM_LAYERS = 3
    DROPOUT = 0.1
    MAX_OFFSET = 5
    OFFSET_PROB = 0.0
    MFCC_FRAMES_PER_VIDEO_FRAME = 2
    TESTING = False 
    MODALITY_DROPOUT_PROB = 0.0

config = Config()


# =================================================================================
# 2. HELPER/MODEL CLASSES (Unchanged)
# =================================================================================

class AddRandomNoise:
    def __init__(self, min_noise=0.01, max_noise=0.1):
        self.min_noise = min_noise
        self.max_noise = max_noise
    def __call__(self, tensor):
        noise_level = random.uniform(self.min_noise, self.max_noise)
        return tensor + torch.randn_like(tensor) * noise_level

class VideoFeatureExtractor(nn.Module):
    def __init__(self, embed_dim):
        super().__init__()
        resnet = models.resnet18(weights=None)
        modules = list(resnet.children())[:-2]
        self.feature_extractor = nn.Sequential(*modules)
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.projection = nn.Linear(resnet.fc.in_features, embed_dim)
    def forward(self, x):
        b, t, c, h, w = x.shape
        x = x.view(b * t, c, h, w)
        features = self.feature_extractor(x)
        pooled_features = self.pool(features).view(b * t, -1)
        projected_features = self.projection(pooled_features)
        output = projected_features.view(b, t, -1)
        return output

class AudioFeatureExtractor(nn.Module):
    def __init__(self, num_mfcc, embed_dim):
        super().__init__()
        self.conv1 = nn.Conv1d(in_channels=num_mfcc, out_channels=64, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(in_channels=64, out_channels=128, kernel_size=3, padding=1)
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
    def get_position_angle_vec(position):
        return [position / np.power(10000, 2 * (hid_j // 2) / d_hid) for hid_j in range(d_hid)]
    sinusoid_table = np.array([get_position_angle_vec(pos_i) for pos_i in range(n_position)])
    sinusoid_table[:, 0::2] = np.sin(sinusoid_table[:, 0::2])
    sinusoid_table[:, 1::2] = np.cos(sinusoid_table[:, 1::2])
    return torch.FloatTensor(sinusoid_table).unsqueeze(0)

class GatedCrossAttentionBlock(nn.Module):
    def __init__(self, embed_dim, num_heads, dropout):
        super().__init__()
        self.ln1 = nn.LayerNorm(embed_dim)
        self.ln2 = nn.LayerNorm(embed_dim)
        self.audio_to_video_attn = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout, batch_first=True)
        self.gate = nn.Sequential(nn.Linear(embed_dim, embed_dim), nn.Sigmoid())
        self.self_attn = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout, batch_first=True)
        self.ffn = nn.Sequential(nn.Linear(embed_dim, embed_dim * 4), nn.GELU(), nn.Dropout(dropout), nn.Linear(embed_dim * 4, embed_dim))
        self.dropout = nn.Dropout(dropout)
    def forward(self, audio_feat, video_feat, video_mask=None):
        audio_norm = self.ln1(audio_feat)
        video_norm = self.ln1(video_feat)
        cross_attn_output, cross_attn_map = self.audio_to_video_attn(query=audio_norm, key=video_norm, value=video_norm, key_padding_mask=video_mask)
        audio_feat = audio_feat + self.dropout(cross_attn_output)
        gated_audio_feat = audio_feat * self.gate(audio_feat)
        gated_audio_norm = self.ln2(gated_audio_feat)
        self_attn_output, _ = self.self_attn(gated_audio_norm, gated_audio_norm, gated_audio_norm)
        gated_audio_feat = gated_audio_feat + self.dropout(self_attn_output)
        gated_audio_norm2 = self.ln2(gated_audio_feat)
        ffn_output = self.ffn(gated_audio_norm2)
        final_output = gated_audio_feat + self.dropout(ffn_output)
        return final_output, cross_attn_map

class PinpointTransformer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.video_extractor = VideoFeatureExtractor(config.EMBED_DIM)
        self.audio_extractor = AudioFeatureExtractor(config.NUM_MFCC, config.EMBED_DIM)
        self.video_pos_encoder = nn.Parameter(torch.randn(1, config.NUM_FRAMES, config.EMBED_DIM))
        self.gated_attention_layers = nn.ModuleList([GatedCrossAttentionBlock(config.EMBED_DIM, config.NUM_HEADS, config.DROPOUT) for _ in range(config.NUM_LAYERS)])
        self.classification_head = nn.Linear(config.EMBED_DIM, 1)
        num_offset_classes = 2 * config.MAX_OFFSET + 1
        self.offset_head = nn.Linear(config.EMBED_DIM, num_offset_classes)
    def forward(self, video, audio, video_mask=None):
        video_feat = self.video_extractor(video)
        audio_feat = self.audio_extractor(audio)
        video_feat = video_feat + self.video_pos_encoder[:, :video_feat.size(1), :]
        audio_len = audio_feat.size(1)
        audio_pos_encoding = get_sinusoidal_embeddings(audio_len, self.config.EMBED_DIM).to(audio_feat.device)
        audio_feat = audio_feat + audio_pos_encoding
        last_attention_map = None
        for layer in self.gated_attention_layers:
            audio_feat, attention_map = layer(audio_feat, video_feat, video_mask)
            last_attention_map = attention_map
        pooled_output = audio_feat.mean(dim=1)
        classification_logits = self.classification_head(pooled_output)
        offset_logits = self.offset_head(pooled_output)
        return classification_logits, offset_logits, last_attention_map

# =================================================================================
# =================================================================================
# FINAL CORRECTED DATASET CLASS
# This version correctly infers labels from the 'n_fakes' key in the original metadata,
# matching the true structure of the LAV-DF dataset.
# =================================================================================
class LAVDFDataset(Dataset):
    def __init__(self, config, split='test'):
        self.config = config
        self.split = split
        self.samples = []
        
        print(f"--- Dynamically building '{split}' dataset ---")

        # Step 1: Load original metadata to get a complete list of test files and their labels
        if not os.path.exists(config.ORIGINAL_METADATA_PATH):
            raise FileNotFoundError(f"Canonical metadata not found: {config.ORIGINAL_METADATA_PATH}")

        print(f"Loading original metadata from: {config.ORIGINAL_METADATA_PATH}")
        with open(config.ORIGINAL_METADATA_PATH, 'r') as f:
            original_metadata = json.load(f)
        
        expected_test_files = {} # {basename: label_str}
        for item in original_metadata:
            # FIXED: Use same logic as preprocessing script - determine label from n_fakes field
            # The original LAV-DF metadata doesn't have a 'label' field, only 'n_fakes'
            if item.get('split') == self.split:
                base_name = os.path.splitext(os.path.basename(item['file']))[0]
                # n_fakes == 0 means real video, n_fakes > 0 means fake video
                n_fakes = item.get('n_fakes', 0)
                label_str = 'fake' if n_fakes > 0 else 'real'
                expected_test_files[base_name] = label_str
        
        print(f"Found {len(expected_test_files)} expected files in '{self.split}' split from original metadata.")
        if not expected_test_files:
             raise ValueError(f"Could not find any files for the '{self.split}' split in the metadata. Check paths and file content.")


        # Step 2: Scan all preprocessed directories to find matching files
        print(f"Scanning {len(config.TEST_DATA_DIRECTORIES)} directories for preprocessed files...")
        found_files = 0
        for data_dir in config.TEST_DATA_DIRECTORIES:
            if not os.path.isdir(data_dir):
                print(f"  - Warning: Directory not found, skipping: {data_dir}")
                continue
            
            # Use os.walk to search recursively
            for root, _, files in tqdm(os.walk(data_dir), desc=f"Scanning {os.path.basename(data_dir)}"):
                for file in files:
                    if file.endswith("_video.pt"):
                        base_name = file.replace("_video.pt", "")
                        
                        # Check if this file is part of the target split
                        if base_name in expected_test_files:
                            video_path = os.path.join(root, file)
                            audio_path = os.path.join(root, f"{base_name}_audio.pt")
                            
                            if os.path.exists(audio_path):
                                label_str = expected_test_files[base_name]
                                is_fake = (label_str == 'fake')
                                self.samples.append({
                                    "video_path": video_path,
                                    "audio_path": audio_path,
                                    "label": 1.0 if is_fake else 0.0,
                                    "is_fake": is_fake,
                                    "original_label": label_str # Keep original label for debugging
                                })
                                found_files += 1

        print(f"Successfully located and loaded {len(self.samples)} test samples.")
        if not self.samples:
            raise FileNotFoundError("No preprocessed test data was found after scanning all directories. Check TEST_DATA_DIRECTORIES and ORIGINAL_METADATA_PATH.")
        
        # Apply sample count limit if specified
        if self.config.MAX_SAMPLES is not None and len(self.samples) > self.config.MAX_SAMPLES:
            print(f"Limiting dataset to {self.config.MAX_SAMPLES} samples (from {len(self.samples)} total)")
            
            # Randomize samples if enabled (to avoid bias from taking first N samples)
            if self.config.RANDOMIZE_SAMPLES:
                print("Randomizing sample order for more representative subset")
                random.shuffle(self.samples)
            
            self.samples = self.samples[:self.config.MAX_SAMPLES]
            print(f"Using {len(self.samples)} samples for evaluation")
            
            # Show distribution of real vs fake in the limited sample
            fake_count = sum(1 for s in self.samples if s['is_fake'])
            real_count = len(self.samples) - fake_count
            print(f"Sample distribution: {real_count} real, {fake_count} fake")
            
            # Debug: Show file naming patterns to check for spurious correlations
            if self.config.DEBUG_MODE:
                print("\n--- DEBUG: Sample Analysis ---")
                print("First 10 sample filenames:")
                for i, sample in enumerate(self.samples[:10]):
                    filename = os.path.basename(sample['video_path'])
                    label = "FAKE" if sample['is_fake'] else "REAL"
                    print(f"  {i+1:2d}. {filename:<25} -> {label}")
                
                # Check for filename patterns that might leak labels
                fake_filenames = [os.path.basename(s['video_path']) for s in self.samples if s['is_fake']]
                real_filenames = [os.path.basename(s['video_path']) for s in self.samples if not s['is_fake']]
                
                print(f"\nFilename pattern analysis:")
                print(f"  Real files: {len(real_filenames)} samples")
                print(f"  Fake files: {len(fake_filenames)} samples")
                
                # Check for common prefixes/patterns
                if fake_filenames:
                    fake_starts = [f[:6] for f in fake_filenames]
                    print(f"  Fake file prefixes (first 6 chars): {set(fake_starts)}")
                if real_filenames:
                    real_starts = [f[:6] for f in real_filenames]
                    print(f"  Real file prefixes (first 6 chars): {set(real_starts)}")
                
                # Check source directory distribution
                print(f"\nSource directory analysis:")
                
                # --- NEW: More robust method to find the source dataset directory ---
                def get_source_dataset_name(path, source_dirs):
                    for source_dir in source_dirs:
                        # Check if the file path is inside one of the configured test directories
                        if os.path.abspath(path).startswith(os.path.abspath(source_dir)):
                            return os.path.basename(source_dir)
                    return "Unknown" # Fallback
                
                fake_dirs = [get_source_dataset_name(s['video_path'], config.TEST_DATA_DIRECTORIES) for s in self.samples if s['is_fake']]
                real_dirs = [get_source_dataset_name(s['video_path'], config.TEST_DATA_DIRECTORIES) for s in self.samples if not s['is_fake']]
                
                fake_dir_counts = Counter(fake_dirs)
                real_dir_counts = Counter(real_dirs)
                
                print(f"  Real samples by directory:")
                for dir_name, count in real_dir_counts.most_common():
                    print(f"    {dir_name}: {count} samples")
                
                print(f"  Fake samples by directory:")
                for dir_name, count in fake_dir_counts.most_common():
                    print(f"    {dir_name}: {count} samples")
                
                # Check if there's a pattern in the filename ranges
                fake_nums = [int(f.split('_')[0]) for f in fake_filenames if f.split('_')[0].isdigit()]
                real_nums = [int(f.split('_')[0]) for f in real_filenames if f.split('_')[0].isdigit()]
                
                if fake_nums and real_nums:
                    print(f"\nFilename number analysis:")
                    print(f"  Real file numbers: min={min(real_nums)}, max={max(real_nums)}, mean={sum(real_nums)/len(real_nums):.0f}")
                    print(f"  Fake file numbers: min={min(fake_nums)}, max={max(fake_nums)}, mean={sum(fake_nums)/len(fake_nums):.0f}")
                
                print("--- END DEBUG ---\n")
        
        self.normalize_transform = T.Normalize(mean=self.config.NORM_MEAN, std=self.config.NORM_STD)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        try:
            sample_info = self.samples[idx]
            video_tensor_uint8 = torch.load(sample_info["video_path"])
            video_tensor = video_tensor_uint8.to(torch.float32) / 255.0
            audio_tensor = torch.load(sample_info["audio_path"])
            video_tensor = self.normalize_transform(video_tensor)
            return {
                "video": video_tensor, "audio": audio_tensor, "label": sample_info["label"],
                "is_fake": sample_info["is_fake"], "video_path": sample_info["video_path"],
                "original_label": sample_info["original_label"]
            }
        except Exception as e:
            print(f"Warning: Error loading sample at index {idx} ({self.samples[idx]['video_path']}): {e}")
            return None

def collate_fn(batch):
    batch = [b for b in batch if b is not None]
    if not batch: return None
    max_audio_len = max([b['audio'].shape[0] for b in batch])
    padded_videos, padded_audios, video_masks, video_paths = [], [], [], []
    labels, is_fakes, original_labels = [], [], []
    for item in batch:
        padded_videos.append(item['video'])
        video_masks.append(torch.zeros(config.NUM_FRAMES, dtype=torch.bool))
        a = item['audio']
        a_pad_len = max_audio_len - a.shape[0]
        padded_a = F.pad(a, (0, 0, 0, a_pad_len), "constant", 0)
        padded_audios.append(padded_a)
        labels.append(item['label'])
        is_fakes.append(item['is_fake'])
        video_paths.append(item['video_path'])
        original_labels.append(item['original_label'])
    return {
        "video": torch.stack(padded_videos), "audio": torch.stack(padded_audios),
        "video_mask": torch.stack(video_masks), "label": torch.tensor(labels, dtype=torch.float32),
        "is_fake": torch.tensor(is_fakes, dtype=torch.bool), "video_path": video_paths,
        "original_label": original_labels
    }


# =================================================================================
# 3. CORE EVALUATION & VISUALIZATION FUNCTIONS (Unchanged)
# =================================================================================

def run_full_evaluation(model, dataloader, device, config=None):
    """Runs model on the entire test set and prints metrics."""
    print("\n" + "="*50)
    print("--- Starting Full Evaluation for Metrics ---")
    model.eval()
    all_labels, all_preds, all_probs = [], [], []
    all_paths = []
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Calculating Metrics"):
            if batch is None: continue
            video, audio, video_mask = batch['video'].to(device), batch['audio'].to(device), batch['video_mask'].to(device)
            labels = batch['label']
            paths = batch['video_path']
            
            with torch.amp.autocast('cuda'):
                cls_logits, _, _ = model(video, audio, video_mask)
            
            probs = torch.sigmoid(cls_logits).squeeze(1).cpu().numpy()
            preds = (probs > 0.5).astype(int)
            
            all_labels.extend(labels.numpy().astype(int))
            all_preds.extend(preds)
            all_probs.extend(probs)
            all_paths.extend(paths)
    
    print("\n--- Final Test Results ---")
    if not all_labels:
        print("Warning: No samples were evaluated. Cannot calculate metrics.")
        return
    
    report = classification_report(all_labels, all_preds, target_names=['Real (0)', 'Fake (1)'], zero_division=0)
    cm = confusion_matrix(all_labels, all_preds)
    print(f"Total Samples Tested: {len(all_labels)}")
    print("\n--- Classification Report ---")
    print(report)
    print("\n--- Confusion Matrix ---")
    print("        Predicted")
    print("       Real  Fake")
    print(f"Real   {cm[0,0]:<5} {cm[0,1]:<5}")
    print(f"Fake   {cm[1,0]:<5} {cm[1,1]:<5}")
    
    # DEBUG: Analysis of misclassified samples
    if config and config.DEBUG_MODE:
        print("\n--- DEBUG: Misclassification Analysis ---")
        
        # Find misclassified samples
        misclassified = []
        for i, (true_label, pred_label, prob, path) in enumerate(zip(all_labels, all_preds, all_probs, all_paths)):
            if true_label != pred_label:
                misclassified.append({
                    'idx': i,
                    'true_label': 'Real' if true_label == 0 else 'Fake',
                    'pred_label': 'Real' if pred_label == 0 else 'Fake',
                    'confidence': prob if pred_label == 1 else (1 - prob),
                    'filename': os.path.basename(path)
                })
        
        print(f"Total misclassified samples: {len(misclassified)}")
        
        if len(misclassified) > 0:
            print("\nFirst 10 misclassified samples:")
            for i, sample in enumerate(misclassified[:10]):
                print(f"  {i+1:2d}. {sample['filename']:<25} True: {sample['true_label']:<4} Pred: {sample['pred_label']:<4} Conf: {sample['confidence']:.3f}")
            
            # Analyze confidence distribution
            high_conf_errors = [s for s in misclassified if s['confidence'] > 0.9]
            print(f"\nHigh-confidence errors (>90%): {len(high_conf_errors)}")
            if high_conf_errors:
                print("High-confidence misclassifications:")
                for sample in high_conf_errors[:5]:
                    print(f"  {sample['filename']:<25} True: {sample['true_label']:<4} Pred: {sample['pred_label']:<4} Conf: {sample['confidence']:.3f}")
        
        # Check for patterns in correct predictions
        print(f"\nConfidence distribution for correct predictions:")
        correct_probs = [probs for true_label, pred_label, probs in zip(all_labels, all_preds, all_probs) if true_label == pred_label]
        if correct_probs:
            import numpy as np
            print(f"  Mean confidence: {np.mean(correct_probs):.3f}")
            print(f"  Min confidence: {np.min(correct_probs):.3f}")
            print(f"  Max confidence: {np.max(correct_probs):.3f}")
            print(f"  Std deviation: {np.std(correct_probs):.3f}")
        
        print("--- END DEBUG ---\n")
    
    print("="*50 + "\n")

def save_attention_map_for_sample(model, batch, device, output_path):
    """Generates and saves a single attention map visualization."""
    model.eval()
    video, audio, video_mask = batch['video'].to(device), batch['audio'].to(device), batch['video_mask'].to(device)
    label = "Fake" if batch['is_fake'][0] else "Real"
    with torch.no_grad():
        with torch.amp.autocast('cuda'):
            cls_logits, _, attention_map = model(video, audio, video_mask)
    pred_prob = torch.sigmoid(cls_logits).item()
    pred_label = "Fake" if pred_prob > 0.5 else "Real"
    avg_attention_map = attention_map.squeeze(0).cpu().numpy()
    fig, ax = plt.subplots(figsize=(8, 8))
    im = ax.imshow(avg_attention_map, cmap='viridis', aspect='auto')
    ax.set_xlabel("Video Frames")
    ax.set_ylabel("Audio Time Steps (MFCCs)")
    ax.set_title(f"Cross-Attention Map\nGT: {label} | Pred: {pred_label} ({pred_prob:.2f})")
    fig.colorbar(im, ax=ax)
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close(fig)

def save_grad_cam_for_sample(model, dataset, sample_idx, device, output_path):
    """Generates and saves a single 3-panel Grad-CAM visualization."""
    original_sample_info = dataset.samples[sample_idx]
    original_video_uint8 = torch.load(original_sample_info['video_path'])
    processed_sample = dataset[sample_idx]
    if processed_sample is None: 
        print(f"Skipping Grad-CAM for sample {sample_idx} due to loading error.")
        return
    video_tensor_model = processed_sample['video']
    audio_tensor_model = processed_sample['audio']
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    frame_idx = config.NUM_FRAMES // 2
    original_frame_np = original_video_uint8[frame_idx].permute(1, 2, 0).numpy()
    axes[0].imshow(original_frame_np)
    axes[0].set_title(f"Original Frame {frame_idx}")
    axes[0].axis('off')
    heatmap_standard = generate_grad_cam(model, video_tensor_model, audio_tensor_model, frame_idx, device)
    heatmap_standard_resized = cv2.resize(heatmap_standard, (original_frame_np.shape[1], original_frame_np.shape[0]))
    heatmap_color = cv2.applyColorMap((255 * heatmap_standard_resized).astype(np.uint8), cv2.COLORMAP_JET)
    superimposed_standard = cv2.addWeighted(original_frame_np, 0.6, heatmap_color, 0.4, 0)
    axes[1].imshow(cv2.cvtColor(superimposed_standard, cv2.COLOR_BGR2RGB))
    axes[1].set_title("Standard Grad-CAM")
    axes[1].axis('off')
    heatmap_multi = generate_multi_layer_grad_cam(model, video_tensor_model, audio_tensor_model, frame_idx, device)
    heatmap_multi_resized = cv2.resize(heatmap_multi, (original_frame_np.shape[1], original_frame_np.shape[0]))
    heatmap_color_multi = cv2.applyColorMap((255 * heatmap_multi_resized).astype(np.uint8), cv2.COLORMAP_JET)
    superimposed_multi = cv2.addWeighted(original_frame_np, 0.6, heatmap_color_multi, 0.4, 0)
    axes[2].imshow(cv2.cvtColor(superimposed_multi, cv2.COLOR_BGR2RGB))
    axes[2].set_title("Multi-Layer Grad-CAM (Enhanced)")
    axes[2].axis('off')
    base_filename = os.path.basename(original_sample_info['video_path']).replace('_video.pt', '')
    gt_label = "Fake" if processed_sample['is_fake'] else "Real"
    fig.suptitle(f"Grad-CAM: {base_filename} (Ground Truth: {gt_label})", fontsize=16)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig(output_path, dpi=100)
    plt.close(fig)


# =================================================================================
# 4. GRAD-CAM HELPERS (Unchanged)
# =================================================================================

def generate_grad_cam(model, video_tensor, actual_audio, target_frame_idx, device):
    model.eval()
    original_audio_training_state = model.audio_extractor.training
    model.audio_extractor.train()
    video_tensor = video_tensor.clone().detach().to(device).requires_grad_(True)
    audio_tensor = actual_audio.unsqueeze(0).to(device)
    feature_maps, gradients = [], []
    def save_feature_map(m, i, o): feature_maps.append(o)
    def save_gradient(m, g_in, g_out): 
        if g_out[0] is not None: gradients.append(g_out[0])
    target_layer = model.video_extractor.feature_extractor[7]
    handle_fw = target_layer.register_forward_hook(save_feature_map)
    handle_bw = target_layer.register_backward_hook(save_gradient)
    default_heatmap = np.zeros((config.VIDEO_SIZE[0], config.VIDEO_SIZE[1]), dtype=np.float32)
    try:
        with torch.enable_grad():
            cls_logits, _, _ = model(video_tensor.unsqueeze(0), audio_tensor)
        if not feature_maps:
            handle_fw.remove(); handle_bw.remove()
            model.audio_extractor.train(original_audio_training_state)
            return default_heatmap
        pred_class_score = torch.sigmoid(cls_logits).squeeze()
        model.zero_grad()
        pred_class_score.backward(retain_graph=False)
        handle_fw.remove(); handle_bw.remove()
        if not gradients:
            model.audio_extractor.train(original_audio_training_state)
            return default_heatmap
        B_f, C, H_f, W_f = feature_maps[0].shape
        num_frames = video_tensor.shape[0]
        frame_feat = feature_maps[0].view(num_frames, C, H_f, W_f)[target_frame_idx].detach()
        frame_grad = gradients[0].view(num_frames, C, H_f, W_f)[target_frame_idx].detach()
        pooled_gradients = torch.mean(frame_grad, dim=(1, 2))
        for i in range(C):
            frame_feat[i] *= pooled_gradients[i]
        heatmap = torch.mean(frame_feat, dim=0).cpu().numpy()
        heatmap = np.maximum(heatmap, 0)
        if np.max(heatmap) > 0: heatmap /= np.max(heatmap)
    except Exception:
        handle_fw.remove(); handle_bw.remove()
        model.audio_extractor.train(original_audio_training_state)
        return default_heatmap
    model.audio_extractor.train(original_audio_training_state)
    return heatmap

def generate_multi_layer_grad_cam(model, video_tensor, actual_audio, target_frame_idx, device):
    model.eval()
    original_audio_training_state = model.audio_extractor.training
    model.audio_extractor.train()
    video_tensor = video_tensor.clone().detach().to(device).requires_grad_(True)
    audio_tensor = actual_audio.unsqueeze(0).to(device)
    target_layers = [model.video_extractor.feature_extractor[6], model.video_extractor.feature_extractor[7]]
    all_feature_maps, all_gradients, handles = [], [], []
    def make_hook(layer_idx):
        def save_feature_map(m, i, o): all_feature_maps.append((layer_idx, o))
        def save_gradient(m, g_in, g_out): 
            if g_out[0] is not None: all_gradients.append((layer_idx, g_out[0]))
        return save_feature_map, save_gradient
    for i, layer in enumerate(target_layers):
        fw_hook, bw_hook = make_hook(i)
        handles.append(layer.register_forward_hook(fw_hook))
        handles.append(layer.register_backward_hook(bw_hook))
    default_heatmap = np.zeros((config.VIDEO_SIZE[0], config.VIDEO_SIZE[1]), dtype=np.float32)
    try:
        with torch.enable_grad():
            cls_logits, _, _ = model(video_tensor.unsqueeze(0), audio_tensor)
        pred_confidence = torch.sigmoid(cls_logits).squeeze()
        model.zero_grad()
        pred_confidence.backward()
        for handle in handles: handle.remove()
        if not all_feature_maps or not all_gradients:
            model.audio_extractor.train(original_audio_training_state)
            return default_heatmap
        combined_heatmap = None
        layer_weights = [0.3, 0.7]
        for layer_idx in range(len(target_layers)):
            layer_features = next((feat for l_idx, feat in all_feature_maps if l_idx == layer_idx), None)
            layer_grads = next((grad for l_idx, grad in all_gradients if l_idx == layer_idx), None)
            if layer_features is None or layer_grads is None: continue
            num_frames = video_tensor.shape[0]
            B_f, C, H_f, W_f = layer_features.shape
            frame_feat = layer_features.view(num_frames, C, H_f, W_f)[target_frame_idx].detach()
            frame_grad = layer_grads.view(num_frames, C, H_f, W_f)[target_frame_idx].detach()
            pooled_gradients = torch.mean(frame_grad, dim=(1, 2))
            for c in range(C):
                frame_feat[c] *= pooled_gradients[c]
            layer_heatmap = torch.mean(frame_feat, dim=0).cpu().numpy()
            layer_heatmap = np.maximum(layer_heatmap, 0)
            if layer_heatmap.shape != (config.VIDEO_SIZE[0], config.VIDEO_SIZE[1]):
                layer_heatmap = cv2.resize(layer_heatmap, (config.VIDEO_SIZE[1], config.VIDEO_SIZE[0]))
            if np.max(layer_heatmap) > 0: layer_heatmap /= np.max(layer_heatmap)
            if combined_heatmap is None:
                combined_heatmap = layer_weights[layer_idx] * layer_heatmap
            else:
                combined_heatmap += layer_weights[layer_idx] * layer_heatmap
        if combined_heatmap is None:
            model.audio_extractor.train(original_audio_training_state)
            return default_heatmap
        combined_heatmap = np.maximum(combined_heatmap, 0)
        if np.max(combined_heatmap) > 0: combined_heatmap /= np.max(combined_heatmap)
    except Exception:
        for handle in handles: handle.remove()
        model.audio_extractor.train(original_audio_training_state)
        return default_heatmap
    model.audio_extractor.train(original_audio_training_state)
    return combined_heatmap


# =================================================================================
# 5. MAIN EXECUTION BLOCK
# =================================================================================
# if __name__ == '__main__':
    print("--- Starting Evaluation & Visualization Script ---")
    print(f"Using device: {config.DEVICE}")
    print(f"Loading model from: {config.MODEL_PATH}")
    
    # Show sample limit configuration
    if config.MAX_SAMPLES is not None:
        print(f"Sample limit: {config.MAX_SAMPLES} samples (for testing)")
    else:
        print("Sample limit: No limit (using all available samples)")
    
    for d in config.TEST_DATA_DIRECTORIES:
        print(f" -> Will search for test data in: {d}")

    # --- Setup Output Directories ---
    attention_dir = os.path.join(config.OUTPUT_VISUALS_DIR, "attention_maps")
    gradcam_dir = os.path.join(config.OUTPUT_VISUALS_DIR, "grad_cam_visuals")
    os.makedirs(attention_dir, exist_ok=True)
    os.makedirs(gradcam_dir, exist_ok=True)
    print(f"Visualizations will be saved to: {config.OUTPUT_VISUALS_DIR}")

    # --- Load Model ---
    model = PinpointTransformer(config).to(config.DEVICE)
    try:
        model.load_state_dict(torch.load(config.MODEL_PATH, map_location=config.DEVICE))
        print("Model loaded successfully.")
    except Exception as e:
        print(f"FATAL: Could not load model. Error: {e}")
        exit()

    # --- Load Dataset ---
    try:
        test_dataset = LAVDFDataset(config, split='test')
        test_loader_metrics = DataLoader(test_dataset, batch_size=config.METRICS_BATCH_SIZE, shuffle=False, collate_fn=collate_fn, num_workers=2)
        test_loader_viz = DataLoader(test_dataset, batch_size=config.VISUALIZATION_BATCH_SIZE, shuffle=False, collate_fn=collate_fn, num_workers=2)
    except Exception as e:
        print(f"FATAL: Could not load dataset. Error: {e}")
        traceback.print_exc()
        exit()
        
    # --- STEP 1: Run Full Evaluation and Print Report ---
    run_full_evaluation(model, test_loader_metrics, config.DEVICE, config)
    
    # --- STEP 2: Generate and Save Visualizations for Every Sample ---
    print("\n" + "="*50)
    print("--- Starting Visualization Generation for Entire Test Set ---")
    print(f"Processing {len(test_dataset)} samples one by one...")

    model.eval()
    # We need the index to retrieve original file paths and for unique naming
    for i, batch in enumerate(tqdm(test_loader_viz, desc="Generating Visuals")):
        if batch is None:
            print(f"Warning: Skipping sample at index {i} due to a batching error.")
            continue
        
        try:
            # Get a unique filename for saving
            base_filename = os.path.basename(batch['video_path'][0]).replace('_video.pt', '')
            gt_label = "Fake" if batch['is_fake'][0] else "Real"
            
            # --- Generate and Save Attention Map ---
            attention_output_path = os.path.join(attention_dir, f"{i:04d}_{base_filename}_{gt_label}_attn.png")
            # This function runs with no_grad internally
            save_attention_map_for_sample(model, batch, config.DEVICE, attention_output_path)
        except Exception as e:
            print(f"\nError processing attention map for sample {i} ({base_filename}): {e}")
        
        try:
            base_filename = os.path.basename(batch['video_path'][0]).replace('_video.pt', '')
            gt_label = "Fake" if batch['is_fake'][0] else "Real"
            gradcam_output_path = os.path.join(gradcam_dir, f"{i:04d}_{base_filename}_{gt_label}_gradcam.png")
            # The Grad-CAM functions handle their own model.eval() and gradient contexts
            save_grad_cam_for_sample(model, test_dataset, i, config.DEVICE, gradcam_output_path)
        except Exception as e:
            print(f"\nError processing Grad-CAM for sample {i} ({base_filename}): {e}")
            traceback.print_exc()
    
    print("\n--- All tasks complete. ---")
    print(f"Evaluation metrics have been printed.")
    print(f"Visualizations saved in '{config.OUTPUT_VISUALS_DIR}'")