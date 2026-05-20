# %%writefile PinPoint.py

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
from torchvision.transforms import v2 as T # Use v2 for modern tensor transforms
import torchvision.models as models
import cv2 # for Grad-CAM
# Add this with your other imports at the top
from sklearn.metrics import precision_score, recall_score, f1_score, classification_report, confusion_matrix
# --- MODIFIED: Import for Mixed-Precision Training ---
from torch.cuda.amp import GradScaler, autocast

import numpy as np
import json
import os
import random
import matplotlib.pyplot as plt
import time
from tqdm import tqdm
import glob ### MODIFIED: Added glob to find metadata files automatically ###

# --- NEW: Explainable AI Imports ---
import shap
from typing import List, Dict, Tuple, Optional, Union, Callable
from copy import deepcopy
import seaborn as sns
from scipy.ndimage import gaussian_filter
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import warnings
warnings.filterwarnings("ignore", category=UserWarning)



# =================================================================================
# HELPER CLASS FOR AUGMENTATION (Unchanged)
# =================================================================================
class AddRandomNoise:
    """A callable class to add random noise to a tensor. Replaces the non-picklable lambda."""
    def __init__(self, min_noise=0.01, max_noise=0.1):
        self.min_noise = min_noise
        self.max_noise = max_noise

    def __call__(self, tensor):
        noise_level = random.uniform(self.min_noise, self.max_noise)
        return tensor + torch.randn_like(tensor) * noise_level


# =================================================================================
# 1. CONFIGURATION (### MODIFIED for Directory Structure ###)
# =================================================================================
class Config:
    # --- Testing Flag ---
    TESTING = False # If True, only use a subset of samples for each split.

    ### MODIFIED: Paths now point to the single, unified preprocessed data directory ###
    DATA_DIRECTORY = "/kaggle/input/new-model-unified-pre-processing/preprocessed_data"
    METADATA_PATH = "/kaggle/input/new-model-unified-pre-processing/preprocessed_data/unified_metadata.json"

    # --- Training ---
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    EPOCHS = 15 # Increased epochs for harder curriculum learning
    BATCH_SIZE = 4
    LEARNING_RATE = 1e-4
    WEIGHT_DECAY = 1e-4

    # --- NEW: Curriculum Training Phases (in epochs) ---
    CURRICULUM_PHASE1_EPOCHS = 2  # Video masking phase
    CURRICULUM_PHASE2_EPOCHS = 3  # Sync-focus phase
    # Phase 3 is the remaining epochs

    # --- NEW: Modality Dropout ---
    MODALITY_DROPOUT_PROB = 0.15 # Probability of dropping a modality per sample

    # --- Loss Weights ---
    W_CLASSIFICATION = 1.0
    W_OFFSET = 0.5
    W_ATTENTION = 3.0 # Increased weight for the more powerful sync loss

    # --- NEW: Synchronization Loss Parameters ---
    SYNC_LOSS_BANDWIDTH = 2 # How wide the 'correct' diagonal band is
    W_SYNC_DIRECT = 1.0     # Weight for direct MSE against target
    W_SYNC_DOMINANCE = 0.5  # Weight for penalizing off-diagonal attention
    W_SYNC_SMOOTHNESS = 0.2 # Weight for encouraging smooth attention shifts

    # --- Data Preprocessing (Values from the preprocessing script) ---
    NUM_FRAMES = 30
    VIDEO_SIZE = (128, 128)
    NUM_MFCC = 13
    NORM_MEAN = [0.485, 0.456, 0.406]
    NORM_STD = [0.229, 0.224, 0.225]

    # --- Model Architecture ---
    EMBED_DIM = 256
    NUM_HEADS = 8
    NUM_LAYERS = 3
    DROPOUT = 0.1

    # --- Auxiliary Task: Offset Prediction ---
    MAX_OFFSET = 5
    OFFSET_PROB = 0.5
    MFCC_FRAMES_PER_VIDEO_FRAME = 2


# =================================================================================
# 2. ENHANCED LOSS FUNCTION (Unchanged)
# =================================================================================
class SynchronizationLoss(nn.Module):
    """
    A multi-component loss to force the model to learn temporal synchronization.
    It guides the attention map of real, synchronized samples to be:
    1. Diagonal: Audio features should attend to corresponding video frames.
    2. Dominant: Most attention energy should be ON the diagonal.
    3. Smooth: Attention should not jump erratically between frames.
    """
    def __init__(self, config):
        super().__init__()
        self.config = config

    def _create_sync_target(self, b, audio_len, video_len, device):
        target = torch.zeros(audio_len, video_len, device=device)
        for i in range(audio_len):
            center = int(i / self.config.MFCC_FRAMES_PER_VIDEO_FRAME)
            for j in range(video_len):
                dist = abs(j - center)
                target[i, j] = -(dist**2) / (2 * (self.config.SYNC_LOSS_BANDWIDTH**2))
        target = torch.exp(target)
        target = target / (target.sum(dim=1, keepdim=True) + 1e-8)
        return target.unsqueeze(0).repeat(b, 1, 1)

    def _diagonal_dominance_loss(self, attention_map):
        b, audio_len, video_len = attention_map.shape
        diagonal_mask = torch.zeros_like(attention_map)
        for i in range(audio_len):
            center = int(i / self.config.MFCC_FRAMES_PER_VIDEO_FRAME)
            start = max(0, center - self.config.SYNC_LOSS_BANDWIDTH)
            end = min(video_len, center + self.config.SYNC_LOSS_BANDWIDTH + 1)
            diagonal_mask[:, i, start:end] = 1.0

        on_diagonal_energy = (attention_map * diagonal_mask).sum(dim=(-1, -2))
        total_energy = attention_map.sum(dim=(-1, -2))
        loss = 1.0 - (on_diagonal_energy / (total_energy + 1e-8))
        return loss.mean()

    def _temporal_smoothness_loss(self, attention_map):
        audio_diff = F.mse_loss(attention_map[:, 1:, :], attention_map[:, :-1, :])
        video_diff = F.mse_loss(attention_map[:, :, 1:], attention_map[:, :, :-1])
        return audio_diff + video_diff

    def forward(self, attention_map, is_synced_mask):
        if not is_synced_mask.any() or attention_map is None:
            return torch.tensor(0.0, device=self.config.DEVICE)

        synced_attention = attention_map[is_synced_mask]
        b_s, t_a, t_v = synced_attention.shape

        sync_target = self._create_sync_target(b_s, t_a, t_v, synced_attention.device)
        direct_loss = F.mse_loss(synced_attention, sync_target)
        dominance_loss = self._diagonal_dominance_loss(synced_attention)
        smoothness_loss = self._temporal_smoothness_loss(synced_attention)

        total_loss = (self.config.W_SYNC_DIRECT * direct_loss +
                      self.config.W_SYNC_DOMINANCE * dominance_loss +
                      self.config.W_SYNC_SMOOTHNESS * smoothness_loss)
        return total_loss


# =================================================================================
# 3. MODEL ARCHITECTURE (Unchanged)
# =================================================================================
class VideoFeatureExtractor(nn.Module):
    def __init__(self, embed_dim):
        super().__init__()
        resnet = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        modules = list(resnet.children())[:-2]
        self.feature_extractor = nn.Sequential(*modules)
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.projection = nn.Linear(resnet.fc.in_features, embed_dim)
        for param in self.feature_extractor[:6].parameters():
            param.requires_grad = False
        print("Initialized VideoFeatureExtractor with a pretrained ResNet-18 backbone.")

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
        print("Initialized AudioFeatureExtractor with a CNN-GRU backbone. (Using LayerNorm instead of BatchNorm)")

    def forward(self, x):
        x = x.transpose(1, 2)
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = x.transpose(1, 2)
        x = self.ln(x)
        output, _ = self.gru(x)
        return output

def get_sinusoidal_embeddings(n_position, d_hid):
    """ Sinusoidal position encoding table """
    def get_position_angle_vec(position):
        return [position / np.power(10000, 2 * (hid_j // 2) / d_hid) for hid_j in range(d_hid)]

    sinusoid_table = np.array([get_position_angle_vec(pos_i) for pos_i in range(n_position)])
    sinusoid_table[:, 0::2] = np.sin(sinusoid_table[:, 0::2])  # dim 2i
    sinusoid_table[:, 1::2] = np.cos(sinusoid_table[:, 1::2])  # dim 2i+1

    return torch.FloatTensor(sinusoid_table).unsqueeze(0)

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
            nn.Dropout(dropout), nn.Linear(embed_dim * 4, embed_dim)
        )
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
        print("PinpointTransformer model initialized.")

    def forward(self, video, audio, video_mask=None):
        video_feat = self.video_extractor(video)
        audio_feat = self.audio_extractor(audio)

        if self.training and self.config.MODALITY_DROPOUT_PROB > 0:
            video_drop_mask = torch.ones(video_feat.size(0), 1, 1, device=video_feat.device)
            audio_drop_mask = torch.ones(audio_feat.size(0), 1, 1, device=audio_feat.device)
            for i in range(video.size(0)):
                if random.random() < self.config.MODALITY_DROPOUT_PROB:
                    if random.random() < 0.5: video_drop_mask[i] = 0
                    else: audio_drop_mask[i] = 0
            video_feat = video_feat * video_drop_mask
            audio_feat = audio_feat * audio_drop_mask

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
# 3.5. COMPREHENSIVE EXPLAINABLE AI TECHNIQUES
# =================================================================================

class SHAPExplainer:
    """SHAP (SHapley Additive exPlanations) integration for multimodal deepfake detection."""
    
    def __init__(self, model, config, background_samples=50):
        self.model = model
        self.config = config
        self.device = config.DEVICE
        self.background_samples = background_samples
        self.video_explainer = None
        self.audio_explainer = None
        
    def setup_explainers(self, background_data):
        """Setup SHAP explainers with background data."""
        print("Setting up SHAP explainers...")
        
        # Extract background samples
        video_bg = []
        audio_bg = []
        
        for i, batch in enumerate(background_data):
            if i >= self.background_samples // len(background_data) + 1:
                break
            if batch is not None:
                video_bg.append(batch['video'])
                audio_bg.append(batch['audio'])
        
        if video_bg and audio_bg:
            video_background = torch.cat(video_bg)[:self.background_samples]
            audio_background = torch.cat(audio_bg)[:self.background_samples]
            
            # Create wrapper for video modality
            def video_model_wrapper(video_input):
                with torch.no_grad():
                    # Use mean audio from background for video explanations
                    mean_audio = audio_background.mean(dim=0, keepdim=True).repeat(video_input.shape[0], 1, 1)
                    logits, _, _ = self.model(video_input, mean_audio)
                    return torch.sigmoid(logits).cpu().numpy()
            
            # Create wrapper for audio modality  
            def audio_model_wrapper(audio_input):
                with torch.no_grad():
                    # Use mean video from background for audio explanations
                    mean_video = video_background.mean(dim=0, keepdim=True).repeat(audio_input.shape[0], 1, 1, 1, 1)
                    logits, _, _ = self.model(mean_video, audio_input)
                    return torch.sigmoid(logits).cpu().numpy()
            
            self.video_explainer = shap.DeepExplainer(video_model_wrapper, video_background)
            self.audio_explainer = shap.DeepExplainer(audio_model_wrapper, audio_background)
            print("SHAP explainers setup complete.")
        
    def explain_sample(self, video, audio, save_path=None):
        """Generate SHAP explanations for a sample."""
        if self.video_explainer is None or self.audio_explainer is None:
            raise ValueError("SHAP explainers not initialized. Call setup_explainers first.")
        
        video_shap = self.video_explainer.shap_values(video)
        audio_shap = self.audio_explainer.shap_values(audio)
        
        if save_path:
            self._visualize_shap(video, audio, video_shap, audio_shap, save_path)
        
        return video_shap, audio_shap
    
    def _visualize_shap(self, video, audio, video_shap, audio_shap, save_path):
        """Visualize SHAP explanations."""
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        
        # Video SHAP visualization
        if len(video_shap) > 0:
            video_importance = np.mean(np.abs(video_shap[0]), axis=0)
            
            # Show original frame, SHAP heatmap, and overlay
            frame_idx = video_importance.sum(axis=(1, 2)).argmax()
            orig_frame = video[0, frame_idx].cpu().permute(1, 2, 0).numpy()
            shap_frame = video_importance[frame_idx]
            
            axes[0, 0].imshow(orig_frame)
            axes[0, 0].set_title('Original Video Frame')
            axes[0, 0].axis('off')
            
            im1 = axes[0, 1].imshow(shap_frame, cmap='RdBu', alpha=0.8)
            axes[0, 1].set_title('Video SHAP Heatmap')
            axes[0, 1].axis('off')
            plt.colorbar(im1, ax=axes[0, 1])
            
            # Overlay
            axes[0, 2].imshow(orig_frame)
            axes[0, 2].imshow(shap_frame, cmap='RdBu', alpha=0.5)
            axes[0, 2].set_title('SHAP Overlay')
            axes[0, 2].axis('off')
        
        # Audio SHAP visualization
        if len(audio_shap) > 0:
            audio_importance = np.mean(np.abs(audio_shap[0]), axis=0)
            
            axes[1, 0].plot(audio[0].cpu().numpy().mean(axis=1))
            axes[1, 0].set_title('Original Audio Signal')
            
            im2 = axes[1, 1].imshow(audio_importance.T, cmap='RdBu', aspect='auto')
            axes[1, 1].set_title('Audio SHAP Heatmap')
            axes[1, 1].set_xlabel('Time Steps')
            axes[1, 1].set_ylabel('MFCC Features')
            plt.colorbar(im2, ax=axes[1, 1])
            
            # Feature importance over time
            feature_importance = audio_importance.sum(axis=1)
            axes[1, 2].plot(feature_importance)
            axes[1, 2].set_title('Audio Feature Importance')
            axes[1, 2].set_xlabel('Time Steps')
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"SHAP visualization saved to: {save_path}")


class LayerRelevancePropagation:
    """Layer-wise Relevance Propagation (LRP) implementation."""
    
    def __init__(self, model, config):
        self.model = model
        self.config = config
        self.device = config.DEVICE
        self.relevance_scores = {}
        
    def register_hooks(self):
        """Register backward hooks for LRP."""
        self.hooks = []
        
        def lrp_relu_hook(module, grad_in, grad_out):
            # LRP rule for ReLU layers
            return (grad_out[0].clamp(min=0),)
        
        def lrp_linear_hook(module, grad_in, grad_out):
            # LRP rule for linear layers with epsilon stabilization
            epsilon = 1e-7
            weight = module.weight.data
            bias = module.bias.data if module.bias is not None else 0
            
            # Forward pass values
            activations = grad_in[0]
            relevance = grad_out[0]
            
            # LRP-epsilon rule
            z = F.linear(activations, weight, bias) + epsilon
            s = relevance / z
            c = F.linear(s, weight.t())
            
            return (activations * c,)
        
        # Register hooks for different layer types
        for name, module in self.model.named_modules():
            if isinstance(module, nn.ReLU):
                hook = module.register_backward_hook(lrp_relu_hook)
                self.hooks.append(hook)
            elif isinstance(module, nn.Linear):
                hook = module.register_backward_hook(lrp_linear_hook)
                self.hooks.append(hook)
    
    def remove_hooks(self):
        """Remove all registered hooks."""
        for hook in self.hooks:
            hook.remove()
        self.hooks = []
    
    def compute_lrp(self, video, audio, target_class=None):
        """Compute LRP relevance scores."""
        video.requires_grad_(True)
        audio.requires_grad_(True)
        
        self.register_hooks()
        
        try:
            # Forward pass
            logits, _, _ = self.model(video, audio)
            
            if target_class is None:
                target_class = (torch.sigmoid(logits) > 0.5).long()
            
            # Backward pass for LRP
            logits.backward(torch.ones_like(logits))
            
            video_relevance = video.grad.clone()
            audio_relevance = audio.grad.clone()
            
        finally:
            self.remove_hooks()
            video.grad = None
            audio.grad = None
        
        return video_relevance, audio_relevance
    
    def visualize_lrp(self, video, audio, video_relevance, audio_relevance, save_path):
        """Visualize LRP results."""
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        
        # Video LRP visualization
        video_rel_sum = video_relevance[0].abs().sum(dim=0).cpu().numpy()
        frame_importance = video_rel_sum.sum(axis=(1, 2))
        most_relevant_frame = frame_importance.argmax()
        
        orig_frame = video[0, most_relevant_frame].cpu().permute(1, 2, 0).numpy()
        rel_frame = video_rel_sum[most_relevant_frame]
        
        axes[0, 0].imshow(orig_frame)
        axes[0, 0].set_title('Original Frame')
        axes[0, 0].axis('off')
        
        im1 = axes[0, 1].imshow(rel_frame, cmap='hot')
        axes[0, 1].set_title('LRP Relevance')
        axes[0, 1].axis('off')
        plt.colorbar(im1, ax=axes[0, 1])
        
        axes[0, 2].plot(frame_importance)
        axes[0, 2].set_title('Frame Importance')
        axes[0, 2].set_xlabel('Frame Index')
        
        # Audio LRP visualization
        audio_rel_sum = audio_relevance[0].abs().cpu().numpy()
        
        axes[1, 0].plot(audio[0].cpu().numpy().mean(axis=1))
        axes[1, 0].set_title('Original Audio')
        
        im2 = axes[1, 1].imshow(audio_rel_sum.T, cmap='hot', aspect='auto')
        axes[1, 1].set_title('Audio LRP Relevance')
        plt.colorbar(im2, ax=axes[1, 1])
        
        feature_importance = audio_rel_sum.sum(axis=0)
        axes[1, 2].plot(feature_importance)
        axes[1, 2].set_title('MFCC Feature Importance')
        axes[1, 2].set_xlabel('Feature Index')
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"LRP visualization saved to: {save_path}")


class IntegratedGradients:
    """Integrated Gradients implementation for attribution analysis."""
    
    def __init__(self, model, config, steps=50):
        self.model = model
        self.config = config
        self.device = config.DEVICE
        self.steps = steps
    
    def compute_integrated_gradients(self, video, audio, baseline_video=None, baseline_audio=None):
        """Compute integrated gradients."""
        if baseline_video is None:
            baseline_video = torch.zeros_like(video)
        if baseline_audio is None:
            baseline_audio = torch.zeros_like(audio)
        
        # Generate interpolated inputs
        alphas = torch.linspace(0, 1, self.steps, device=self.device)
        
        video_grads = []
        audio_grads = []
        
        for alpha in alphas:
            # Linear interpolation between baseline and input
            interpolated_video = baseline_video + alpha * (video - baseline_video)
            interpolated_audio = baseline_audio + alpha * (audio - baseline_audio)
            
            interpolated_video.requires_grad_(True)
            interpolated_audio.requires_grad_(True)
            
            # Forward pass
            logits, _, _ = self.model(interpolated_video, interpolated_audio)
            pred_prob = torch.sigmoid(logits)
            
            # Backward pass
            pred_prob.backward(torch.ones_like(pred_prob))
            
            video_grads.append(interpolated_video.grad.clone())
            audio_grads.append(interpolated_audio.grad.clone())
            
            interpolated_video.grad = None
            interpolated_audio.grad = None
        
        # Average gradients and multiply by input difference
        avg_video_grads = torch.stack(video_grads).mean(dim=0)
        avg_audio_grads = torch.stack(audio_grads).mean(dim=0)
        
        video_ig = (video - baseline_video) * avg_video_grads
        audio_ig = (audio - baseline_audio) * avg_audio_grads
        
        return video_ig, audio_ig
    
    def visualize_integrated_gradients(self, video, audio, video_ig, audio_ig, save_path):
        """Visualize integrated gradients."""
        fig, axes = plt.subplots(2, 4, figsize=(20, 10))
        
        # Video IG visualization
        video_ig_abs = video_ig[0].abs().sum(dim=0).cpu().numpy()
        frame_importance = video_ig_abs.sum(axis=(1, 2))
        most_important_frame = frame_importance.argmax()
        
        orig_frame = video[0, most_important_frame].cpu().permute(1, 2, 0).numpy()
        ig_frame = video_ig_abs[most_important_frame]
        
        axes[0, 0].imshow(orig_frame)
        axes[0, 0].set_title('Original Frame')
        axes[0, 0].axis('off')
        
        im1 = axes[0, 1].imshow(ig_frame, cmap='viridis')
        axes[0, 1].set_title('IG Attribution')
        axes[0, 1].axis('off')
        plt.colorbar(im1, ax=axes[0, 1])
        
        # Overlay
        axes[0, 2].imshow(orig_frame, alpha=0.7)
        axes[0, 2].imshow(ig_frame, cmap='viridis', alpha=0.3)
        axes[0, 2].set_title('IG Overlay')
        axes[0, 2].axis('off')
        
        axes[0, 3].plot(frame_importance)
        axes[0, 3].set_title('Temporal Importance')
        axes[0, 3].set_xlabel('Frame Index')
        
        # Audio IG visualization
        audio_ig_abs = audio_ig[0].abs().cpu().numpy()
        
        axes[1, 0].plot(audio[0].cpu().numpy().mean(axis=1))
        axes[1, 0].set_title('Original Audio Signal')
        
        im2 = axes[1, 1].imshow(audio_ig_abs.T, cmap='viridis', aspect='auto')
        axes[1, 1].set_title('Audio IG Attribution')
        axes[1, 1].set_xlabel('Time Steps')
        axes[1, 1].set_ylabel('MFCC Features')
        plt.colorbar(im2, ax=axes[1, 1])
        
        # Feature importance over time
        temporal_importance = audio_ig_abs.sum(axis=1)
        feature_importance = audio_ig_abs.sum(axis=0)
        
        axes[1, 2].plot(temporal_importance)
        axes[1, 2].set_title('Temporal Audio Importance')
        axes[1, 2].set_xlabel('Time Steps')
        
        axes[1, 3].plot(feature_importance)
        axes[1, 3].set_title('MFCC Feature Importance')
        axes[1, 3].set_xlabel('Feature Index')
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Integrated Gradients visualization saved to: {save_path}")


class AttentionRollout:
    """Attention Rollout and Flow Analysis for transformer layers."""
    
    def __init__(self, model, config):
        self.model = model
        self.config = config
        self.device = config.DEVICE
        self.attention_maps = []
        
    def register_attention_hooks(self):
        """Register hooks to capture attention maps."""
        self.attention_maps = []
        self.hooks = []
        
        def attention_hook(module, input, output):
            if hasattr(module, 'audio_to_video_attn'):
                # This is a GatedCrossAttentionBlock
                audio_feat, video_feat = input[0], input[1]
                with torch.no_grad():
                    _, attn_weights = module.audio_to_video_attn(
                        query=module.ln1(audio_feat),
                        key=module.ln1(video_feat),
                        value=module.ln1(video_feat),
                        average_attn_weights=True
                    )
                    self.attention_maps.append(attn_weights.cpu())
        
        # Register hooks for attention layers
        for name, module in self.model.named_modules():
            if isinstance(module, GatedCrossAttentionBlock):
                hook = module.register_forward_hook(attention_hook)
                self.hooks.append(hook)
    
    def remove_hooks(self):
        """Remove attention hooks."""
        for hook in self.hooks:
            hook.remove()
        self.hooks = []
    
    def compute_attention_rollout(self, video, audio):
        """Compute attention rollout across layers."""
        self.register_attention_hooks()
        
        try:
            with torch.no_grad():
                _ = self.model(video, audio)
            
            if not self.attention_maps:
                return None
            
            # Initialize with first layer attention
            rollout = self.attention_maps[0][0]  # Remove batch dimension
            
            # Roll out attention through layers
            for i in range(1, len(self.attention_maps)):
                rollout = torch.matmul(rollout, self.attention_maps[i][0])
            
            return rollout
            
        finally:
            self.remove_hooks()
    
    def compute_attention_flow(self, video, audio):
        """Compute attention flow analysis."""
        self.register_attention_hooks()
        
        try:
            with torch.no_grad():
                _ = self.model(video, audio)
            
            if not self.attention_maps:
                return None, None
            
            # Analyze attention flow
            layer_entropies = []
            layer_maxes = []
            
            for attn_map in self.attention_maps:
                attn = attn_map[0]  # Remove batch dimension
                
                # Compute entropy (attention dispersion)
                attn_norm = F.softmax(attn, dim=-1)
                entropy = -torch.sum(attn_norm * torch.log(attn_norm + 1e-8), dim=-1)
                layer_entropies.append(entropy.mean().item())
                
                # Compute max attention (attention concentration)
                layer_maxes.append(attn.max().item())
            
            return layer_entropies, layer_maxes
            
        finally:
            self.remove_hooks()
    
    def visualize_attention_analysis(self, video, audio, save_path):
        """Visualize complete attention analysis."""
        rollout = self.compute_attention_rollout(video, audio)
        entropies, maxes = self.compute_attention_flow(video, audio)
        
        if rollout is None:
            print("No attention maps captured for visualization")
            return
        
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        
        # Attention Rollout visualization
        rollout_np = rollout.numpy()
        im1 = axes[0, 0].imshow(rollout_np, cmap='viridis', aspect='auto')
        axes[0, 0].set_title('Attention Rollout')
        axes[0, 0].set_xlabel('Video Frames')
        axes[0, 0].set_ylabel('Audio Time Steps')
        plt.colorbar(im1, ax=axes[0, 0])
        
        # Temporal attention summary
        temporal_attention = rollout_np.sum(axis=0)
        axes[0, 1].plot(temporal_attention)
        axes[0, 1].set_title('Temporal Attention Distribution')
        axes[0, 1].set_xlabel('Video Frame')
        axes[0, 1].set_ylabel('Cumulative Attention')
        
        # Audio attention summary
        audio_attention = rollout_np.sum(axis=1)
        axes[0, 2].plot(audio_attention)
        axes[0, 2].set_title('Audio Attention Distribution')
        axes[0, 2].set_xlabel('Audio Time Step')
        axes[0, 2].set_ylabel('Cumulative Attention')
        
        # Attention flow analysis
        if entropies and maxes:
            layer_nums = list(range(len(entropies)))
            
            axes[1, 0].plot(layer_nums, entropies, 'o-', label='Entropy')
            axes[1, 0].set_title('Attention Entropy by Layer')
            axes[1, 0].set_xlabel('Layer')
            axes[1, 0].set_ylabel('Entropy')
            axes[1, 0].legend()
            
            axes[1, 1].plot(layer_nums, maxes, 'o-', label='Max Attention', color='red')
            axes[1, 1].set_title('Max Attention by Layer')
            axes[1, 1].set_xlabel('Layer')
            axes[1, 1].set_ylabel('Max Attention Value')
            axes[1, 1].legend()
            
            # Combined flow analysis
            axes[1, 2].plot(layer_nums, entropies, 'o-', label='Entropy', alpha=0.7)
            ax2 = axes[1, 2].twinx()
            ax2.plot(layer_nums, maxes, 'o-', label='Max Attention', color='red', alpha=0.7)
            axes[1, 2].set_xlabel('Layer')
            axes[1, 2].set_ylabel('Entropy', color='blue')
            ax2.set_ylabel('Max Attention', color='red')
            axes[1, 2].set_title('Attention Flow Analysis')
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Attention analysis visualization saved to: {save_path}")


class CounterfactualExplainer:
    """Counterfactual Explanations for what-if scenario analysis."""
    
    def __init__(self, model, config, num_steps=100, lr=0.01):
        self.model = model
        self.config = config
        self.device = config.DEVICE
        self.num_steps = num_steps
        self.lr = lr
    
    def generate_counterfactual(self, video, audio, target_label=None, lambda_sparse=0.1, lambda_smooth=0.05):
        """Generate counterfactual explanation."""
        if target_label is None:
            # Get current prediction
            with torch.no_grad():
                logits, _, _ = self.model(video, audio)
                current_pred = (torch.sigmoid(logits) > 0.5).int().item()
                target_label = 1 - current_pred  # Flip the prediction
        
        # Create perturbed versions
        video_pert = video.clone().requires_grad_(True)
        audio_pert = audio.clone().requires_grad_(True)
        
        optimizer = torch.optim.Adam([video_pert, audio_pert], lr=self.lr)
        
        losses = []
        
        for step in range(self.num_steps):
            optimizer.zero_grad()
            
            logits, _, _ = self.model(video_pert, audio_pert)
            pred_prob = torch.sigmoid(logits)
            
            # Classification loss (move towards target)
            if target_label == 1:
                class_loss = F.binary_cross_entropy(pred_prob, torch.ones_like(pred_prob))
            else:
                class_loss = F.binary_cross_entropy(pred_prob, torch.zeros_like(pred_prob))
            
            # Sparsity loss (minimize changes)
            video_diff = video_pert - video
            audio_diff = audio_pert - audio
            sparse_loss = torch.mean(video_diff ** 2) + torch.mean(audio_diff ** 2)
            
            # Smoothness loss (encourage smooth changes)
            video_smooth_loss = torch.mean((video_diff[:, 1:] - video_diff[:, :-1]) ** 2)
            audio_smooth_loss = torch.mean((audio_diff[:, 1:] - audio_diff[:, :-1]) ** 2)
            smooth_loss = video_smooth_loss + audio_smooth_loss
            
            # Total loss
            total_loss = class_loss + lambda_sparse * sparse_loss + lambda_smooth * smooth_loss
            
            total_loss.backward()
            optimizer.step()
            
            losses.append(total_loss.item())
            
            # Check if target reached
            with torch.no_grad():
                current_pred = (torch.sigmoid(self.model(video_pert, audio_pert)[0]) > 0.5).int().item()
                if current_pred == target_label:
                    print(f"Counterfactual found at step {step}")
                    break
        
        return video_pert.detach(), audio_pert.detach(), losses
    
    def analyze_counterfactual(self, original_video, original_audio, cf_video, cf_audio):
        """Analyze the differences in counterfactual."""
        video_diff = (cf_video - original_video).abs()
        audio_diff = (cf_audio - original_audio).abs()
        
        # Video analysis
        video_importance = video_diff.sum(dim=1)  # Sum over channels
        frame_importance = video_importance.sum(dim=(1, 2))  # Sum over spatial dims
        
        # Audio analysis  
        audio_importance = audio_diff.sum(dim=2)  # Sum over features
        temporal_importance = audio_importance.sum(dim=1)  # Sum over time
        
        return {
            'video_diff': video_diff,
            'audio_diff': audio_diff, 
            'frame_importance': frame_importance,
            'temporal_importance': temporal_importance,
            'total_video_change': video_diff.sum().item(),
            'total_audio_change': audio_diff.sum().item()
        }
    
    def visualize_counterfactual(self, original_video, original_audio, cf_video, cf_audio, analysis, save_path):
        """Visualize counterfactual explanation."""
        fig, axes = plt.subplots(3, 4, figsize=(20, 15))
        
        # Get most changed frame
        most_changed_frame = analysis['frame_importance'].argmax().item()
        
        # Original vs Counterfactual video
        orig_frame = original_video[0, most_changed_frame].cpu().permute(1, 2, 0).numpy()
        cf_frame = cf_video[0, most_changed_frame].cpu().permute(1, 2, 0).numpy()
        diff_frame = analysis['video_diff'][0, most_changed_frame].sum(dim=0).cpu().numpy()
        
        axes[0, 0].imshow(orig_frame)
        axes[0, 0].set_title('Original Frame')
        axes[0, 0].axis('off')
        
        axes[0, 1].imshow(cf_frame)
        axes[0, 1].set_title('Counterfactual Frame')
        axes[0, 1].axis('off')
        
        im1 = axes[0, 2].imshow(diff_frame, cmap='hot')
        axes[0, 2].set_title('Video Difference')
        axes[0, 2].axis('off')
        plt.colorbar(im1, ax=axes[0, 2])
        
        axes[0, 3].plot(analysis['frame_importance'].cpu().numpy())
        axes[0, 3].set_title('Frame Change Importance')
        axes[0, 3].set_xlabel('Frame Index')
        
        # Original vs Counterfactual audio
        orig_audio = original_audio[0].cpu().numpy()
        cf_audio_np = cf_audio[0].cpu().numpy()
        audio_diff = analysis['audio_diff'][0].cpu().numpy()
        
        axes[1, 0].plot(orig_audio.mean(axis=1))
        axes[1, 0].set_title('Original Audio')
        axes[1, 0].set_xlabel('Time Steps')
        
        axes[1, 1].plot(cf_audio_np.mean(axis=1))
        axes[1, 1].set_title('Counterfactual Audio')
        axes[1, 1].set_xlabel('Time Steps')
        
        im2 = axes[1, 2].imshow(audio_diff.T, cmap='hot', aspect='auto')
        axes[1, 2].set_title('Audio Difference')
        axes[1, 2].set_xlabel('Time Steps')
        axes[1, 2].set_ylabel('MFCC Features')
        plt.colorbar(im2, ax=axes[1, 2])
        
        axes[1, 3].plot(analysis['temporal_importance'].cpu().numpy())
        axes[1, 3].set_title('Temporal Change Importance')
        axes[1, 3].set_xlabel('Time Steps')
        
        # Summary statistics
        axes[2, 0].bar(['Video', 'Audio'], [analysis['total_video_change'], analysis['total_audio_change']])
        axes[2, 0].set_title('Total Change by Modality')
        axes[2, 0].set_ylabel('Total Change')
        
        # Feature-wise audio changes
        feature_changes = audio_diff.sum(axis=0)
        axes[2, 1].plot(feature_changes)
        axes[2, 1].set_title('MFCC Feature Changes')
        axes[2, 1].set_xlabel('Feature Index')
        
        # Temporal video changes
        temporal_video_changes = analysis['video_diff'][0].sum(dim=(1, 2, 3)).cpu().numpy()
        axes[2, 2].plot(temporal_video_changes)
        axes[2, 2].set_title('Temporal Video Changes')
        axes[2, 2].set_xlabel('Frame Index')
        
        # Clear unused subplot
        axes[2, 3].axis('off')
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Counterfactual visualization saved to: {save_path}")


class ConceptActivationVectors:
    """Concept Activation Vectors (CAV) and Testing with CAV (TCAV) implementation."""
    
    def __init__(self, model, config):
        self.model = model
        self.config = config
        self.device = config.DEVICE
        self.cavs = {}
        self.activations = {}
        
    def register_activation_hooks(self, layer_names):
        """Register hooks to capture activations from specified layers."""
        self.activations = {name: [] for name in layer_names}
        self.hooks = []
        
        def get_activation(name):
            def hook(module, input, output):
                self.activations[name].append(output.detach().cpu())
            return hook
        
        for name, module in self.model.named_modules():
            if name in layer_names:
                hook = module.register_forward_hook(get_activation(name))
                self.hooks.append(hook)
    
    def remove_hooks(self):
        """Remove activation hooks."""
        for hook in self.hooks:
            hook.remove()
        self.hooks = []
    
    def extract_concept_activations(self, concept_data, random_data, layer_names):
        """Extract activations for concept and random examples."""
        self.register_activation_hooks(layer_names)
        
        concept_activations = {name: [] for name in layer_names}
        random_activations = {name: [] for name in layer_names}
        
        # Extract concept activations
        with torch.no_grad():
            for batch in concept_data:
                if batch is None:
                    continue
                self.activations = {name: [] for name in layer_names}
                _ = self.model(batch['video'].to(self.device), batch['audio'].to(self.device))
                
                for name in layer_names:
                    if self.activations[name]:
                        concept_activations[name].extend(self.activations[name])
        
        # Extract random activations
        with torch.no_grad():
            for batch in random_data:
                if batch is None:
                    continue
                self.activations = {name: [] for name in layer_names}
                _ = self.model(batch['video'].to(self.device), batch['audio'].to(self.device))
                
                for name in layer_names:
                    if self.activations[name]:
                        random_activations[name].extend(self.activations[name])
        
        self.remove_hooks()
        return concept_activations, random_activations
    
    def train_cav(self, concept_activations, random_activations, layer_name):
        """Train a CAV for a specific concept and layer."""
        if layer_name not in concept_activations or layer_name not in random_activations:
            raise ValueError(f"No activations found for layer {layer_name}")
        
        # Flatten activations
        concept_acts = torch.cat(concept_activations[layer_name]).flatten(1).numpy()
        random_acts = torch.cat(random_activations[layer_name]).flatten(1).numpy()
        
        # Create labels (1 for concept, 0 for random)
        concept_labels = np.ones(len(concept_acts))
        random_labels = np.zeros(len(random_acts))
        
        # Combine data
        X = np.vstack([concept_acts, random_acts])
        y = np.concatenate([concept_labels, random_labels])
        
        # Train linear classifier
        clf = LogisticRegression(max_iter=1000)
        clf.fit(X, y)
        
        # The CAV is the normal vector to the decision boundary
        cav = clf.coef_[0]
        cav = cav / np.linalg.norm(cav)  # Normalize
        
        self.cavs[layer_name] = {
            'vector': cav,
            'classifier': clf,
            'accuracy': accuracy_score(y, clf.predict(X))
        }
        
        return cav
    
    def compute_tcav_score(self, test_data, layer_name, target_class=1):
        """Compute TCAV score for a concept."""
        if layer_name not in self.cavs:
            raise ValueError(f"CAV not trained for layer {layer_name}")
        
        cav = self.cavs[layer_name]['vector']
        directional_derivatives = []
        
        self.register_activation_hooks([layer_name])
        
        for batch in test_data:
            if batch is None:
                continue
                
            video = batch['video'].to(self.device).requires_grad_(True)
            audio = batch['audio'].to(self.device).requires_grad_(True)
            
            self.activations = {layer_name: []}
            
            # Forward pass
            logits, _, _ = self.model(video, audio)
            
            if self.activations[layer_name]:
                layer_output = self.activations[layer_name][0]
                
                # Compute gradient of prediction w.r.t. layer activations
                if target_class == 1:
                    pred_score = torch.sigmoid(logits)
                else:
                    pred_score = 1 - torch.sigmoid(logits)
                
                grad_outputs = torch.ones_like(pred_score)
                gradients = torch.autograd.grad(
                    outputs=pred_score,
                    inputs=layer_output,
                    grad_outputs=grad_outputs,
                    create_graph=True,
                    retain_graph=True
                )[0]
                
                # Compute directional derivative along CAV
                grad_flat = gradients.flatten(1).cpu().numpy()
                for i in range(grad_flat.shape[0]):
                    dd = np.dot(grad_flat[i], cav)
                    directional_derivatives.append(dd > 0)
        
        self.remove_hooks()
        
        # TCAV score is the fraction of positive directional derivatives
        tcav_score = np.mean(directional_derivatives) if directional_derivatives else 0
        return tcav_score
    
    def visualize_cav_analysis(self, layer_names, concept_name, save_path):
        """Visualize CAV analysis results."""
        if not self.cavs:
            print("No CAVs trained. Run train_cav first.")
            return
        
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        
        # CAV accuracies
        layers = list(self.cavs.keys())
        accuracies = [self.cavs[layer]['accuracy'] for layer in layers]
        
        axes[0, 0].bar(range(len(layers)), accuracies)
        axes[0, 0].set_xticks(range(len(layers)))
        axes[0, 0].set_xticklabels(layers, rotation=45)
        axes[0, 0].set_title(f'CAV Classification Accuracy\nConcept: {concept_name}')
        axes[0, 0].set_ylabel('Accuracy')
        axes[0, 0].axhline(y=0.5, color='r', linestyle='--', alpha=0.7)
        
        # CAV vector magnitudes
        vector_norms = [np.linalg.norm(self.cavs[layer]['vector']) for layer in layers]
        axes[0, 1].bar(range(len(layers)), vector_norms)
        axes[0, 1].set_xticks(range(len(layers)))
        axes[0, 1].set_xticklabels(layers, rotation=45)
        axes[0, 1].set_title('CAV Vector Magnitudes')
        axes[0, 1].set_ylabel('L2 Norm')
        
        # Placeholder for TCAV scores (would need to be computed separately)
        axes[1, 0].text(0.5, 0.5, 'TCAV Scores\n(Compute using compute_tcav_score)', 
                       transform=axes[1, 0].transAxes, ha='center', va='center')
        axes[1, 0].set_title('TCAV Scores by Layer')
        
        # CAV vector visualization (first few components)
        if layers:
            first_layer = layers[0]
            cav_vector = self.cavs[first_layer]['vector']
            
            # Show first 50 components or all if fewer
            n_components = min(50, len(cav_vector))
            axes[1, 1].plot(cav_vector[:n_components])
            axes[1, 1].set_title(f'CAV Vector Components\nLayer: {first_layer}')
            axes[1, 1].set_xlabel('Component Index')
            axes[1, 1].set_ylabel('Weight')
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"CAV analysis visualization saved to: {save_path}")


# =================================================================================
# 3.6. UNIFIED XAI VISUALIZATION FRAMEWORK
# =================================================================================

class UnifiedXAIVisualizer:
    """Unified framework for visualizing all explainability techniques."""
    
    def __init__(self, model, config):
        self.model = model
        self.config = config
        self.device = config.DEVICE
        
        # Initialize all XAI techniques
        self.shap_explainer = SHAPExplainer(model, config)
        self.lrp = LayerRelevancePropagation(model, config)
        self.integrated_gradients = IntegratedGradients(model, config)
        self.attention_rollout = AttentionRollout(model, config)
        self.counterfactual = CounterfactualExplainer(model, config)
        self.cav = ConceptActivationVectors(model, config)
    
    def comprehensive_explanation(self, video, audio, sample_idx, output_dir="xai_outputs"):
        """Generate comprehensive explanations using all techniques."""
        os.makedirs(output_dir, exist_ok=True)
        
        print(f"Generating comprehensive XAI analysis for sample {sample_idx}...")
        
        # Get model prediction
        with torch.no_grad():
            logits, _, attention_map = self.model(video, audio)
            pred_prob = torch.sigmoid(logits).item()
            pred_label = "Fake" if pred_prob > 0.5 else "Real"
        
        print(f"Model prediction: {pred_label} (confidence: {pred_prob:.3f})")
        
        results = {
            'sample_idx': sample_idx,
            'prediction': pred_label,
            'confidence': pred_prob,
            'techniques': {}
        }
        
        # 1. Integrated Gradients
        try:
            print("Computing Integrated Gradients...")
            video_ig, audio_ig = self.integrated_gradients.compute_integrated_gradients(video, audio)
            ig_path = os.path.join(output_dir, f"integrated_gradients_sample_{sample_idx}.png")
            self.integrated_gradients.visualize_integrated_gradients(video, audio, video_ig, audio_ig, ig_path)
            results['techniques']['integrated_gradients'] = ig_path
        except Exception as e:
            print(f"Integrated Gradients failed: {e}")
        
        # 2. Layer-wise Relevance Propagation
        try:
            print("Computing LRP...")
            video_rel, audio_rel = self.lrp.compute_lrp(video, audio)
            lrp_path = os.path.join(output_dir, f"lrp_sample_{sample_idx}.png")
            self.lrp.visualize_lrp(video, audio, video_rel, audio_rel, lrp_path)
            results['techniques']['lrp'] = lrp_path
        except Exception as e:
            print(f"LRP failed: {e}")
        
        # 3. Attention Analysis
        try:
            print("Computing Attention Analysis...")
            attention_path = os.path.join(output_dir, f"attention_analysis_sample_{sample_idx}.png")
            self.attention_rollout.visualize_attention_analysis(video, audio, attention_path)
            results['techniques']['attention'] = attention_path
        except Exception as e:
            print(f"Attention Analysis failed: {e}")
        
        # 4. Counterfactual Explanations
        try:
            print("Generating Counterfactual Explanations...")
            cf_video, cf_audio, losses = self.counterfactual.generate_counterfactual(video, audio)
            cf_analysis = self.counterfactual.analyze_counterfactual(video, audio, cf_video, cf_audio)
            cf_path = os.path.join(output_dir, f"counterfactual_sample_{sample_idx}.png")
            self.counterfactual.visualize_counterfactual(video, audio, cf_video, cf_audio, cf_analysis, cf_path)
            results['techniques']['counterfactual'] = cf_path
        except Exception as e:
            print(f"Counterfactual Explanations failed: {e}")
        
        # 5. Summary visualization
        self._create_summary_visualization(video, audio, sample_idx, results, output_dir)
        
        return results
    
    def _create_summary_visualization(self, video, audio, sample_idx, results, output_dir):
        """Create a summary visualization combining key insights."""
        fig = plt.figure(figsize=(20, 12))
        gs = fig.add_gridspec(3, 4, hspace=0.3, wspace=0.3)
        
        # Title
        fig.suptitle(f'Comprehensive XAI Analysis - Sample {sample_idx}\n'
                    f'Prediction: {results["prediction"]} (Confidence: {results["confidence"]:.3f})', 
                    fontsize=16, fontweight='bold')
        
        # Original data
        ax1 = fig.add_subplot(gs[0, 0])
        if video.shape[1] > 0:
            frame = video[0, video.shape[1]//2].cpu().permute(1, 2, 0).numpy()
            ax1.imshow(frame)
        ax1.set_title('Original Video Frame')
        ax1.axis('off')
        
        ax2 = fig.add_subplot(gs[0, 1])
        if audio.shape[1] > 0:
            ax2.plot(audio[0].cpu().numpy().mean(axis=1))
        ax2.set_title('Original Audio Signal')
        ax2.set_xlabel('Time Steps')
        
        # Model architecture info
        ax3 = fig.add_subplot(gs[0, 2:])
        info_text = f"""
Model: PinPoint Transformer
Video Frames: {video.shape[1]}
Audio Length: {audio.shape[1]}
Features: {audio.shape[2]}

XAI Techniques Applied:
• Integrated Gradients: {'✓' if 'integrated_gradients' in results['techniques'] else '✗'}
• Layer-wise Relevance Propagation: {'✓' if 'lrp' in results['techniques'] else '✗'}
• Attention Analysis: {'✓' if 'attention' in results['techniques'] else '✗'}
• Counterfactual Explanations: {'✓' if 'counterfactual' in results['techniques'] else '✗'}
        """
        ax3.text(0.1, 0.5, info_text, transform=ax3.transAxes, fontsize=12, 
                verticalalignment='center', fontfamily='monospace')
        ax3.axis('off')
        
        # Placeholder for technique summaries
        technique_axes = [
            fig.add_subplot(gs[1, 0]),  # IG summary
            fig.add_subplot(gs[1, 1]),  # LRP summary  
            fig.add_subplot(gs[1, 2]),  # Attention summary
            fig.add_subplot(gs[1, 3]),  # Counterfactual summary
        ]
        
        technique_names = ['Integrated Gradients', 'Layer-wise RP', 'Attention Analysis', 'Counterfactual']
        
        for ax, name in zip(technique_axes, technique_names):
            ax.text(0.5, 0.5, f'{name}\nSee detailed\nvisualization', 
                   transform=ax.transAxes, ha='center', va='center')
            ax.set_title(name)
            ax.axis('off')
        
        # Bottom row for summary insights
        ax_insights = fig.add_subplot(gs[2, :])
        insights_text = f"""
Key Insights from XAI Analysis:
• The model's prediction confidence is {results['confidence']:.1%}
• Multiple explanation techniques provide complementary views of the decision process
• Detailed visualizations saved to: {output_dir}/
• Each technique highlights different aspects: attribution (IG), layer relevance (LRP), attention flow, and counterfactual scenarios
        """
        ax_insights.text(0.05, 0.5, insights_text, transform=ax_insights.transAxes, 
                        fontsize=11, verticalalignment='center')
        ax_insights.axis('off')
        
        summary_path = os.path.join(output_dir, f"xai_summary_sample_{sample_idx}.png")
        plt.savefig(summary_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        results['techniques']['summary'] = summary_path
        print(f"XAI summary visualization saved to: {summary_path}")


# =================================================================================
# 4. DATASET & COLLATOR (### FIXED and ENHANCED ###)
# =================================================================================
class LAVDFDataset(Dataset):
    def __init__(self, config, split='train'):
        self.config = config
        self.split = split
        self.current_epoch = 0 # This is used for curriculum learning logic
        self.samples = []

        print(f"--- Loading '{split}' data from unified metadata ---")
        
        # --- MODIFIED: Load the single metadata file ---
        if not os.path.exists(config.METADATA_PATH):
            raise FileNotFoundError(
                f"Unified metadata file not found at: {config.METADATA_PATH}\n"
                "Please run the `unified_preprocessing.py` script first."
            )
            
        with open(config.METADATA_PATH, 'r') as f:
            all_metadata = json.load(f)
            
        # Filter items for the target split
        split_metadata = [item for item in all_metadata if item.get('split') == self.split]
        print(f"Found {len(split_metadata)} entries for split '{self.split}' in the unified metadata.")
        
        # --- MODIFIED: Simplified path construction ---
        for item in split_metadata:
            # Paths in the unified metadata are relative to the data directory
            video_path = os.path.join(config.DATA_DIRECTORY, item['preprocessed_video_path'])
            audio_path = os.path.join(config.DATA_DIRECTORY, item['preprocessed_audio_path'])

            if os.path.exists(video_path) and os.path.exists(audio_path):
                is_fake = (item['label'] == 'fake')
                label_value = 1.0 if is_fake else 0.0
                self.samples.append({
                    "video_path": video_path, "audio_path": audio_path,
                    "label": label_value, "is_fake": is_fake
                })
            else:
                # This warning should ideally not appear if the unified preprocessor ran correctly
                print(f"  [PATH NOT FOUND] Video: {video_path} (Exists: {os.path.exists(video_path)})")


        print(f"Successfully located and loaded {len(self.samples)} samples for the '{self.split}' split.")
        
        # --- Add testing flag logic ---
        if self.config.TESTING:
            print(f"!!! TESTING FLAG IS TRUE: Truncating '{self.split}' split to a maximum of 100 samples. !!!")
            random.shuffle(self.samples)
            self.samples = self.samples[:100]

        # This error will trigger if the JSON files are found but contain no entries for the requested split OR if all paths failed.
        # This error will trigger if the JSON files are found but contain no entries for the requested split OR if all paths failed.
        if not self.samples:
            # For train and dev splits, an empty dataset is a critical error.
            if self.split in ['train', 'dev']:
                raise FileNotFoundError(
                    f"CRITICAL: No preprocessed data found for the '{self.split}' split.\n"
                    f"Please check that your metadata file contains valid entries for this split."
                )
            # For the 'test' split, we can allow it to be empty and just show a warning.
            else:
                print(f"WARNING: No samples were found for the '{self.split}' split. This part of the evaluation will be skipped.")

        self.normalize_transform = T.Normalize(mean=self.config.NORM_MEAN, std=self.config.NORM_STD)
        if self.split == 'train':
            print("### Training mode: Activating AGGRESSIVE anti-shortcut augmentations. ###")
            self.visual_augmentations = T.Compose([
                T.RandomApply([T.Compose([T.ToDtype(torch.uint8, scale=True), T.JPEG(quality=(20, 95)), T.ToDtype(torch.float32, scale=True)])], p=0.8),
                T.RandomApply([AddRandomNoise(min_noise=0.01, max_noise=0.1)], p=0.6),
                T.RandomApply([T.GaussianBlur(kernel_size=5, sigma=(0.5, 3.0))], p=0.7),
                T.RandomApply([T.Compose([T.Resize((config.VIDEO_SIZE[0]//2, config.VIDEO_SIZE[1]//2)), T.Resize(config.VIDEO_SIZE)])], p=0.5),
                T.RandomApply([T.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.1)], p=0.7),
                T.RandomErasing(p=0.5, scale=(0.05, 0.25), ratio=(0.5, 2.0), value=0),
            ])
        else:
            self.visual_augmentations = None

    def __len__(self):
        return len(self.samples)

    def set_epoch(self, epoch):
        self.current_epoch = epoch

    def __getitem__(self, idx):
        try:
            sample_info = self.samples[idx]
            video_path = sample_info["video_path"]
            audio_path = sample_info["audio_path"]
    
            # Load video tensor
            video_tensor_uint8 = torch.load(video_path)
            video_tensor = video_tensor_uint8.to(torch.float32) / 255.0
    
            # Load and fix audio tensor
            audio_tensor = torch.load(audio_path)
            original_shape = audio_tensor.shape
    
            if audio_tensor.ndim == 3:
                if audio_tensor.shape[0] == self.config.NUM_MFCC:
                    # Shape [13, extra, time], e.g., [13, 2, 922]
                    audio_tensor = audio_tensor.mean(dim=1)  # Average over extra dimension -> [13, 922]
                    audio_tensor = audio_tensor.transpose(0, 1)  # [13, 922] -> [922, 13]
                elif audio_tensor.shape[1] == self.config.NUM_MFCC:
                    # Shape [channels, 13, time], e.g., [2, 13, 584]
                    audio_tensor = audio_tensor.mean(dim=0)  # Average over channels -> [13, 584]
                    audio_tensor = audio_tensor.transpose(0, 1)  # [13, 584] -> [584, 13]
                elif audio_tensor.shape[2] == self.config.NUM_MFCC:
                    # Shape [time, channels, 13], less likely
                    audio_tensor = audio_tensor.mean(dim=1)  # Average over channels -> [time, 13]
                else:
                    print(f"Warning: Unfixable audio shape {original_shape} at {audio_path}. Skipping.")
                    return None
                print(f"Reshaped audio from {original_shape} to {audio_tensor.shape}")
            elif audio_tensor.ndim == 2:
                if audio_tensor.shape[0] == self.config.NUM_MFCC:
                    audio_tensor = audio_tensor.transpose(0, 1)  # [13, time] -> [time, 13]
                elif audio_tensor.shape[1] != self.config.NUM_MFCC:
                    print(f"Warning: Audio shape {original_shape} has incorrect MFCC count. Skipping.")
                    return None
            else:
                print(f"Warning: Audio tensor has invalid dimensions {audio_tensor.ndim}. Skipping.")
                return None
    
            if self.split == 'train' and self.visual_augmentations:
                video_tensor = self.visual_augmentations(video_tensor)
    
            video_tensor = self.normalize_transform(video_tensor)
    
            label = sample_info["label"]
            is_fake = sample_info["is_fake"]
            offset_label = self.config.MAX_OFFSET
    
            if self.split == 'train' and not is_fake and random.random() < self.config.OFFSET_PROB:
                offset_frames = random.randint(-self.config.MAX_OFFSET, self.config.MAX_OFFSET)
                if offset_frames != 0:
                    offset_label += offset_frames
                    audio_offset = offset_frames * self.config.MFCC_FRAMES_PER_VIDEO_FRAME
                    if audio_offset > 0:
                        audio_tensor = audio_tensor[audio_offset:]
                    elif audio_offset < 0:
                        audio_tensor = audio_tensor[:audio_offset]
    
            return {"video": video_tensor, "audio": audio_tensor, "label": label, "is_fake": is_fake, "offset_label": offset_label}
    
        except Exception as e:
            print(f"Warning: Skipping sample {idx} due to error: {e}")
            return None


def collate_fn(batch):
    batch = [b for b in batch if b is not None]
    if not batch: return None
    config = Config() # Need a local config for collate_fn to access constants
    
    # --- START: MODIFIED BLOCK FOR ROBUSTNESS ---
    processed_batch = []
    expected_features = config.NUM_MFCC

    for item in batch:
        audio = item['audio']

        # 1. Validate the audio tensor's shape
        if audio.ndim == 2 and audio.shape[1] == expected_features:
            # Correct shape: (time, features). No action needed.
            pass
        elif audio.ndim == 2 and audio.shape[0] == expected_features:
            # Swapped shape: (features, time). Transpose it.
            item['audio'] = audio.transpose(0, 1)
        else:
            # Incorrect shape (e.g., 3D tensor or wrong 2D dimensions).
            # This sample is corrupt and will be skipped.
            print(f"Warning: Skipping sample due to unexpected audio shape: {audio.shape}. Expected a 2D tensor with one dimension of size {expected_features}.")
            continue # Skip this malformed item

        processed_batch.append(item)

    if not processed_batch: return None
    batch = processed_batch # Continue with the clean batch
    # --- END: MODIFIED BLOCK ---

    max_audio_len = max([b['audio'].shape[0] for b in batch])
    padded_videos, padded_audios, video_masks = [], [], []
    labels, offset_labels, is_fakes = [], [], []

    for item in batch:
        video = item['video']
        # Handle video frame count mismatch
        current_frames = video.shape[0]  # (frames, channels, height, width)

        if current_frames > config.NUM_FRAMES:
            # Sample evenly spaced frames if we have more frames than expected
            indices = torch.linspace(0, current_frames - 1, config.NUM_FRAMES, dtype=torch.long)
            video = video[indices]
        elif current_frames < config.NUM_FRAMES:
            # Pad with zeros if we have fewer frames than expected
            pad_frames = config.NUM_FRAMES - current_frames
            padding = torch.zeros(pad_frames, video.shape[1], video.shape[2], video.shape[3],
                                dtype=video.dtype, device=video.device)
            video = torch.cat([video, padding], dim=0)

        padded_videos.append(video)

        # Create appropriate video mask (all False since we're not masking any frames)
        video_masks.append(torch.zeros(config.NUM_FRAMES, dtype=torch.bool))

        a = item['audio']
        a_pad_len = max_audio_len - a.shape[0]
        padded_a = F.pad(a, (0, 0, 0, a_pad_len), "constant", 0)
        padded_audios.append(padded_a)
        labels.append(item['label'])
        offset_labels.append(item['offset_label'])
        is_fakes.append(item['is_fake'])
        
    return {
        "video": torch.stack(padded_videos), "audio": torch.stack(padded_audios),
        "video_mask": torch.stack(video_masks), "label": torch.tensor(labels, dtype=torch.float32),
        "offset_label": torch.tensor(offset_labels, dtype=torch.long), "is_fake": torch.tensor(is_fakes, dtype=torch.bool)
    }

# =================================================================================
# 5. TRAINING & EVALUATION (Unchanged)
# =================================================================================
def _mask_video_frames(video_tensor, mask_ratio_range=(0.5, 0.8)):
    masked_video = video_tensor.clone()
    b, t, c, h, w = masked_video.shape
    for i in range(b):
        mask_ratio = random.uniform(*mask_ratio_range)
        num_masked = int(t * mask_ratio)
        mask_indices = torch.randperm(t)[:num_masked]
        masked_video[i, mask_indices] = 0
    return masked_video

def train_one_epoch(model, dataloader, optimizer, scheduler, loss_fns, device, config, epoch, scaler):
    model.train()
    total_loss, correct_preds, total_samples = 0, 0, 0
    progress_bar = tqdm(dataloader, desc=f"Training E{epoch+1}", leave=False)

    phase = 3
    phase1_end = config.CURRICULUM_PHASE1_EPOCHS
    phase2_end = phase1_end + config.CURRICULUM_PHASE2_EPOCHS
    if epoch < phase1_end: phase = 1
    elif epoch < phase2_end: phase = 2
    if epoch == 0: print("--- Starting Curriculum Phase 1: Video Masking ---")
    if epoch == phase1_end: print("--- Starting Curriculum Phase 2: Sync-Focus ---")
    if epoch == phase2_end: print("--- Starting Curriculum Phase 3: Full Training ---")

    for batch in progress_bar:
        if batch is None: continue
        video, audio, video_mask = batch['video'].to(device), batch['audio'].to(device), batch['video_mask'].to(device)
        cls_labels, offset_labels, is_fake_mask = batch['label'].to(device), batch['offset_label'].to(device), batch['is_fake'].to(device)

        if phase == 1:
            video = _mask_video_frames(video)

        optimizer.zero_grad(set_to_none=True)

        with autocast():
            cls_logits, offset_logits, attention_map = model(video, audio, video_mask)
            loss_cls = loss_fns['classification'](cls_logits.squeeze(1), cls_labels)

            real_samples_mask = ~is_fake_mask
            loss_offset = loss_fns['offset'](offset_logits[real_samples_mask], offset_labels[real_samples_mask]) if real_samples_mask.sum() > 0 else torch.tensor(0.0, device=device)
            if phase == 2:
                loss_offset = loss_offset * 0.1

            synced_mask = (offset_labels == config.MAX_OFFSET) & real_samples_mask
            loss_attn = loss_fns['attention'](attention_map, synced_mask)

            combined_loss = (config.W_CLASSIFICATION * loss_cls +
                             config.W_OFFSET * loss_offset +
                             config.W_ATTENTION * loss_attn)

        if not torch.isfinite(combined_loss):
            print("WARNING: Encountered non-finite loss. Skipping batch.")
            continue

        scaler.scale(combined_loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        scaler.step(optimizer)
        scaler.update()

        scheduler.step()

        total_loss += combined_loss.item()
        preds = (torch.sigmoid(cls_logits) > 0.5).squeeze(1)
        correct_preds += (preds == cls_labels.bool()).sum().item()
        total_samples += cls_labels.size(0)
        progress_bar.set_postfix({"Phase": phase, "Loss": f"{combined_loss.item():.4f}", "AttnL": f"{loss_attn.item():.3f}", "Acc": f"{correct_preds/total_samples:.2f}"})

    return total_loss / len(dataloader), correct_preds / total_samples

def evaluate_model(model, dataloader, loss_fns, device, config):
    model.eval()
    total_loss, correct_preds, total_samples = 0, 0, 0
    with torch.no_grad():
        progress_bar = tqdm(dataloader, desc="Evaluating", leave=False)
        for batch in progress_bar:
            if batch is None: continue
            video, audio, video_mask = batch['video'].to(device), batch['audio'].to(device), batch['video_mask'].to(device)
            cls_labels = batch['label'].to(device)

            with autocast():
                cls_logits, _, _ = model(video, audio, video_mask)
                loss_cls = loss_fns['classification'](cls_logits.squeeze(1), cls_labels)

            total_loss += loss_cls.item()
            preds = (torch.sigmoid(cls_logits) > 0.5).squeeze(1)
            correct_preds += (preds == cls_labels.bool()).sum().item()
            total_samples += cls_labels.size(0)
    return total_loss / len(dataloader), correct_preds / total_samples

# =================================================================================
# 6. VISUALIZATION & MAIN EXECUTION (Unchanged)
# =================================================================================
# NOTE: The rest of the functions below this point did not need modification as
# they operate on data loaded by the Dataset, not on file paths directly.
# Only the main execution block at the very end was slightly adjusted.
# =================================================================================

def visualize_attention(model, dataset, device, config, sample_idx=None):
    model.eval()
    if sample_idx is None:
        sample_idx = random.randint(0, len(dataset) - 1)
    print(f"\n--- Visualizing Attention for Sample {sample_idx} ---")
    sample = dataset[sample_idx]
    if sample is None:
        print("Could not load sample for visualization.")
        return
    batch = collate_fn([sample])
    video, audio, video_mask = batch['video'].to(device), batch['audio'].to(device), batch['video_mask'].to(device)
    label = "Fake" if batch['is_fake'][0] else "Real"
    with torch.no_grad():
        with autocast(): # Use autocast for consistency
            cls_logits, offset_logits, attention_map = model(video, audio, video_mask)
    pred_prob = torch.sigmoid(cls_logits).item()
    pred_label = "Fake" if pred_prob > 0.5 else "Real"
    offset_pred_class = torch.argmax(offset_logits, dim=1).item()
    predicted_offset = offset_pred_class - config.MAX_OFFSET
    avg_attention_map = attention_map.squeeze(0).cpu().numpy()
    fig, ax = plt.subplots(figsize=(8, 8))
    im = ax.imshow(avg_attention_map, cmap='viridis', aspect='auto')
    ax.set_xlabel("Video Frames"), ax.set_ylabel("Audio Time Steps (MFCCs)")
    ax.set_title(f"Cross-Attention Map\nGT: {label} | Pred: {pred_label} ({pred_prob:.2f}) | Pred Offset: {predicted_offset} frames")
    fig.colorbar(im, ax=ax)
    plt.tight_layout()
    save_path = f"attention_map_sample_{sample_idx}_{label}.png"
    plt.savefig(save_path)
    plt.close(fig)
    print(f"Attention map visualization saved to: {save_path}")
    print("--- Visualization Complete ---")

# FILE: PinPoint.py

# ... (all the code before this function remains the same) ...

def visualize_grad_cam_grid(model, dataset, device, config, num_samples=5):
    print(f"\n--- [6/6] Generating Grad-CAM Grid for {num_samples} Fake Samples ---")
    fake_indices = [i for i, s in enumerate(dataset.samples) if s['is_fake']]
    if len(fake_indices) < num_samples: num_samples = len(fake_indices)
    if num_samples == 0: print("No fake samples found to visualize."); return
    selected_indices = random.sample(fake_indices, num_samples)
    fig, axes = plt.subplots(num_samples, 2, figsize=(10, 4 * num_samples))
    if num_samples == 1: axes = np.expand_dims(axes, axis=0)
    fig.suptitle("Grad-CAM: Visual Regions Most Inconsistent with Audio", fontsize=16)
    for i, sample_idx in enumerate(selected_indices):
        try:
            frame_idx = random.randint(config.NUM_FRAMES // 4, 3 * config.NUM_FRAMES // 4)
            original_sample_info = dataset.samples[sample_idx]
            original_video_uint8 = torch.load(original_sample_info['video_path'])
            original_frame_np = original_video_uint8[frame_idx].permute(1, 2, 0).numpy() / 255.0
            processed_sample = dataset[sample_idx]
            if processed_sample is None: continue
            video_tensor_model = processed_sample['video']; audio_tensor_model = processed_sample['audio']
            
            # --- MODIFIED LINE ---
            # We now pass the 'config' object to the function
            heatmap = generate_grad_cam(model, video_tensor_model, audio_tensor_model, frame_idx, device, config)
            # --- END MODIFIED LINE ---

            heatmap_resized = cv2.resize(heatmap, (original_frame_np.shape[1], original_frame_np.shape[0]))
            heatmap_color = cv2.applyColorMap(np.uint8(255 * heatmap_resized), cv2.COLORMAP_JET)
            superimposed_img = cv2.addWeighted(np.uint8(255 * original_frame_np), 0.6, heatmap_color, 0.4, 0)
            axes[i, 0].imshow(original_frame_np); axes[i, 0].set_title(f"Sample {sample_idx} | Frame {frame_idx} (Original)"); axes[i, 0].axis('off')
            axes[i, 1].imshow(cv2.cvtColor(superimposed_img, cv2.COLOR_BGR2RGB)); axes[i, 1].set_title("Grad-CAM Overlay"); axes[i, 1].axis('off')
        except Exception as e:
            print(f"Error during Grad-CAM for sample {sample_idx}: {e}")
            axes[i, 0].set_title(f"Sample {sample_idx}\nError"); axes[i, 0].axis('off'); axes[i, 1].axis('off')
    plt.tight_layout(rect=[0, 0.03, 1, 0.96])
    save_path = "grad_cam_visualization_grid.png"
    plt.savefig(save_path)
    plt.close(fig)
    print(f"Grad-CAM grid visualization saved to: {save_path}")
    print("--- Grad-CAM Visualization Complete ---")


# --- MODIFIED FUNCTION SIGNATURE ---
# The 'config' object is now a required argument
def generate_grad_cam(model, video_tensor, actual_audio, target_frame_idx, device, config):
# --- END MODIFIED FUNCTION SIGNATURE ---
    model.eval()
    original_audio_train_state = model.audio_extractor.training
    model.audio_extractor.train()

    video_tensor = video_tensor.clone().detach().to(device).requires_grad_(True)
    audio_tensor = actual_audio.unsqueeze(0).to(device)

    feature_maps, gradients = [], []
    def save_feature_map(module, input, output): feature_maps.append(output)
    def save_gradient(module, grad_in, grad_out): gradients.append(grad_out[0])

    target_layer = model.video_extractor.feature_extractor[6]
    handle_fw = target_layer.register_forward_hook(save_feature_map)
    handle_bw = target_layer.register_backward_hook(save_gradient)

    # Default to a black heatmap in case of any failure
    # THIS LINE IS NOW FIXED BECAUSE 'config' is passed in
    H, W = config.VIDEO_SIZE
    default_heatmap = np.zeros((H, W), dtype=np.float32)

    try:
        with torch.enable_grad(), autocast():
            cls_logits, offset_logits, _ = model(video_tensor.unsqueeze(0), audio_tensor)

        # --- FINAL ROBUSTNESS FIX STARTS HERE ---

        # 1. Check if we have the necessary data from the hooks and the model output
        if not feature_maps or not offset_logits.nelement() or not gradients or gradients[0] is None:
            handle_fw.remove(), handle_bw.remove()
            if not original_audio_train_state: model.audio_extractor.eval()
            return default_heatmap

        predicted_offset_class = torch.argmax(offset_logits, dim=1).item()
        model.zero_grad()
        offset_logits[0, predicted_offset_class].backward()
        handle_fw.remove(), handle_bw.remove()

        # 2. Re-check gradients after backward pass, as the hook might have failed
        if not gradients or gradients[0] is None:
             if not original_audio_train_state: model.audio_extractor.eval()
             return default_heatmap

        B_f, C, H_f, W_f = feature_maps[0].shape
        num_frames = video_tensor.shape[0]
        frame_feat = feature_maps[0].view(num_frames, C, H_f, W_f)[target_frame_idx].detach()
        frame_grad = gradients[0].view(num_frames, C, H_f, W_f)[target_frame_idx].detach()

        pooled_gradients = torch.mean(frame_grad, dim=(1, 2))
        for i in range(C): frame_feat[i] *= pooled_gradients[i]

        heatmap = torch.mean(frame_feat, dim=0).cpu().numpy()

        # 3. The most critical check: ensure heatmap is a valid, finite 2D array
        if not isinstance(heatmap, np.ndarray) or heatmap.ndim != 2 or not np.isfinite(heatmap).all() or np.max(heatmap) <= 1e-8:
            if not original_audio_train_state: model.audio_extractor.eval()
            return default_heatmap

        # --- FINAL ROBUSTNESS FIX ENDS HERE ---

        # If we passed all checks, proceed with normalization
        heatmap = np.maximum(heatmap, 0)
        heatmap /= np.max(heatmap)
        heatmap = np.power(heatmap, 1.5)
        heatmap[heatmap < 0.5] = 0

    except Exception:
        # Broad except to catch any other unforeseen error during Grad-CAM and prevent a crash
        handle_fw.remove()
        handle_bw.remove()
        if not original_audio_train_state: model.audio_extractor.eval()
        return default_heatmap # Return the safe, black heatmap

    if not original_audio_train_state: model.audio_extractor.eval()
    return heatmap


# =================================================================================
# NEW: COMPREHENSIVE XAI EVALUATION FUNCTIONS
# =================================================================================

def comprehensive_xai_evaluation(model, test_dataset, config, num_samples=10, output_dir="comprehensive_xai_outputs"):
    """
    Perform comprehensive XAI evaluation using all implemented techniques.
    
    Args:
        model: Trained PinPoint model
        test_dataset: Test dataset
        config: Configuration object  
        num_samples: Number of samples to analyze
        output_dir: Output directory for visualizations
    """
    print("\n" + "="*60)
    print("--- COMPREHENSIVE EXPLAINABLE AI EVALUATION ---")
    print("="*60)
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Initialize unified XAI visualizer
    xai_visualizer = UnifiedXAIVisualizer(model, config)
    
    # Create output directories for different techniques
    technique_dirs = {
        'integrated_gradients': os.path.join(output_dir, 'integrated_gradients'),
        'lrp': os.path.join(output_dir, 'lrp'), 
        'attention_analysis': os.path.join(output_dir, 'attention_analysis'),
        'counterfactual': os.path.join(output_dir, 'counterfactual'),
        'shap': os.path.join(output_dir, 'shap'),
        'cav_tcav': os.path.join(output_dir, 'cav_tcav'),
        'summaries': os.path.join(output_dir, 'summaries')
    }
    
    for dir_path in technique_dirs.values():
        os.makedirs(dir_path, exist_ok=True)
    
    # Select diverse samples for analysis
    fake_indices = [i for i, s in enumerate(test_dataset.samples) if s['is_fake']]
    real_indices = [i for i, s in enumerate(test_dataset.samples) if not s['is_fake']]
    
    # Select balanced samples
    num_fake = min(num_samples // 2, len(fake_indices))
    num_real = min(num_samples // 2, len(real_indices))
    
    selected_fake = random.sample(fake_indices, num_fake) if fake_indices else []
    selected_real = random.sample(real_indices, num_real) if real_indices else []
    selected_indices = selected_fake + selected_real
    
    print(f"Selected {len(selected_indices)} samples for XAI analysis:")
    print(f"  - {len(selected_fake)} fake samples")  
    print(f"  - {len(selected_real)} real samples")
    
    # Setup SHAP explainer with background data
    print("\nSetting up SHAP explainer...")
    try:
        # Create a small background dataset
        background_loader = DataLoader(test_dataset, batch_size=4, shuffle=True, collate_fn=collate_fn)
        background_batches = []
        for i, batch in enumerate(background_loader):
            if i >= 5:  # Use first 5 batches as background
                break
            if batch is not None:
                background_batches.append(batch)
        
        if background_batches:
            xai_visualizer.shap_explainer.setup_explainers(background_batches)
            print("SHAP explainer setup complete.")
        else:
            print("Warning: Could not setup SHAP explainer - no valid background batches.")
    except Exception as e:
        print(f"SHAP setup failed: {e}")
    
    # Process each selected sample
    all_results = []
    
    for i, sample_idx in enumerate(selected_indices):
        print(f"\n{'='*40}")
        print(f"Processing Sample {i+1}/{len(selected_indices)} (Index: {sample_idx})")
        print(f"{'='*40}")
        
        try:
            # Load sample
            sample = test_dataset[sample_idx]
            if sample is None:
                print(f"Skipping sample {sample_idx} - failed to load")
                continue
                
            # Prepare batch
            batch = collate_fn([sample])
            if batch is None:
                print(f"Skipping sample {sample_idx} - failed to collate")
                continue
                
            video = batch['video'].to(config.DEVICE)
            audio = batch['audio'].to(config.DEVICE)
            ground_truth = "Fake" if batch['is_fake'][0] else "Real"
            
            print(f"Ground truth: {ground_truth}")
            
            # Generate comprehensive explanations
            sample_output_dir = os.path.join(technique_dirs['summaries'], f"sample_{sample_idx}")
            results = xai_visualizer.comprehensive_explanation(video, audio, sample_idx, sample_output_dir)
            results['ground_truth'] = ground_truth
            
            all_results.append(results)
            
        except Exception as e:
            print(f"Error processing sample {sample_idx}: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    # Generate overall summary report
    _generate_xai_summary_report(all_results, output_dir)
    
    print(f"\n{'='*60}")
    print("--- XAI EVALUATION COMPLETE ---")
    print(f"Results saved to: {output_dir}")
    print(f"Processed {len(all_results)} samples successfully")
    print("="*60)
    
    return all_results


def demonstrate_cav_tcav(model, test_dataset, config, output_dir="cav_outputs"):
    """
    Demonstrate CAV and TCAV functionality with concept analysis.
    
    Args:
        model: Trained model
        test_dataset: Test dataset  
        config: Configuration object
        output_dir: Output directory
    """
    print("\n--- CAV/TCAV Concept Analysis Demonstration ---")
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Initialize CAV analyzer
    cav_analyzer = ConceptActivationVectors(model, config)
    
    # Define layer names to analyze
    layer_names = [
        'video_extractor.projection',
        'audio_extractor.gru', 
        'gated_attention_layers.0.audio_to_video_attn',
        'gated_attention_layers.1.audio_to_video_attn', 
        'classification_head'
    ]
    
    # Create concept datasets (fake vs real as concept)
    fake_samples = [s for s in test_dataset.samples if s['is_fake']][:20]
    real_samples = [s for s in test_dataset.samples if not s['is_fake']][:20]
    
    if not fake_samples or not real_samples:
        print("Insufficient samples for CAV analysis")
        return
    
    # Create data loaders for concept analysis
    fake_indices = [test_dataset.samples.index(s) for s in fake_samples]
    real_indices = [test_dataset.samples.index(s) for s in real_samples]
    
    fake_subset = torch.utils.data.Subset(test_dataset, fake_indices)
    real_subset = torch.utils.data.Subset(test_dataset, real_indices)
    
    fake_loader = DataLoader(fake_subset, batch_size=4, collate_fn=collate_fn)
    real_loader = DataLoader(real_subset, batch_size=4, collate_fn=collate_fn)
    
    try:
        # Extract concept activations
        print("Extracting concept activations...")
        concept_activations, random_activations = cav_analyzer.extract_concept_activations(
            fake_loader, real_loader, layer_names
        )
        
        # Train CAVs for each layer
        print("Training CAVs...")
        tcav_scores = {}
        
        for layer_name in layer_names:
            try:
                cav = cav_analyzer.train_cav(concept_activations, random_activations, layer_name)
                print(f"  CAV trained for {layer_name} - Accuracy: {cav_analyzer.cavs[layer_name]['accuracy']:.3f}")
                
                # Compute TCAV score
                test_loader = DataLoader(test_dataset, batch_size=4, shuffle=False, collate_fn=collate_fn)
                tcav_score = cav_analyzer.compute_tcav_score(
                    [next(iter(test_loader))], layer_name, target_class=1
                )
                tcav_scores[layer_name] = tcav_score
                print(f"  TCAV Score: {tcav_score:.3f}")
                
            except Exception as e:
                print(f"  Failed to train CAV for {layer_name}: {e}")
        
        # Visualize CAV analysis
        cav_viz_path = os.path.join(output_dir, "cav_analysis_fake_vs_real.png")
        cav_analyzer.visualize_cav_analysis(layer_names, "Fake vs Real", cav_viz_path)
        
        # Create TCAV summary
        _create_tcav_summary(tcav_scores, output_dir)
        
        print(f"CAV/TCAV analysis saved to: {output_dir}")
        
    except Exception as e:
        print(f"CAV/TCAV analysis failed: {e}")
        import traceback
        traceback.print_exc()


def _generate_xai_summary_report(results, output_dir):
    """Generate a comprehensive summary report of XAI analysis."""
    if not results:
        print("No results to summarize")
        return
    
    # Create summary statistics
    correct_predictions = sum(1 for r in results if r['prediction'] == r['ground_truth'])
    total_samples = len(results)
    accuracy = correct_predictions / total_samples
    
    # Analyze confidence by correctness
    correct_confidences = [r['confidence'] for r in results if r['prediction'] == r['ground_truth']]
    incorrect_confidences = [r['confidence'] for r in results if r['prediction'] != r['ground_truth']]
    
    # Count successful technique applications
    technique_success = {}
    for technique in ['integrated_gradients', 'lrp', 'attention', 'counterfactual']:
        success_count = sum(1 for r in results if technique in r['techniques'])
        technique_success[technique] = success_count
    
    # Create summary visualization
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # Accuracy and sample distribution
    prediction_counts = {'Correct': correct_predictions, 'Incorrect': total_samples - correct_predictions}
    axes[0, 0].pie(prediction_counts.values(), labels=prediction_counts.keys(), autopct='%1.1f%%')
    axes[0, 0].set_title(f'Prediction Accuracy\n({accuracy:.1%} correct)')
    
    # Confidence distributions
    if correct_confidences and incorrect_confidences:
        axes[0, 1].hist([correct_confidences, incorrect_confidences], 
                       label=['Correct', 'Incorrect'], alpha=0.7, bins=10)
        axes[0, 1].set_xlabel('Confidence')
        axes[0, 1].set_ylabel('Count')
        axes[0, 1].set_title('Confidence Distributions')
        axes[0, 1].legend()
    
    # Technique success rates
    techniques = list(technique_success.keys())
    success_rates = [technique_success[t] / total_samples for t in techniques]
    
    bars = axes[1, 0].bar(techniques, success_rates)
    axes[1, 0].set_ylabel('Success Rate')
    axes[1, 0].set_title('XAI Technique Success Rates')
    axes[1, 0].set_ylim(0, 1)
    
    # Add value labels on bars
    for bar, rate in zip(bars, success_rates):
        height = bar.get_height()
        axes[1, 0].text(bar.get_x() + bar.get_width()/2., height + 0.01,
                       f'{rate:.1%}', ha='center', va='bottom')
    
    # Summary text
    summary_text = f"""
XAI Analysis Summary Report

Total Samples Analyzed: {total_samples}
Prediction Accuracy: {accuracy:.1%} ({correct_predictions}/{total_samples})

Average Confidence:
• Correct Predictions: {np.mean(correct_confidences):.3f} ± {np.std(correct_confidences):.3f}
• Incorrect Predictions: {np.mean(incorrect_confidences):.3f} ± {np.std(incorrect_confidences):.3f}

Technique Application Success:
• Integrated Gradients: {technique_success.get('integrated_gradients', 0)}/{total_samples}
• Layer-wise Relevance Propagation: {technique_success.get('lrp', 0)}/{total_samples}  
• Attention Analysis: {technique_success.get('attention', 0)}/{total_samples}
• Counterfactual Explanations: {technique_success.get('counterfactual', 0)}/{total_samples}

Key Insights:
• XAI techniques provide complementary explanations
• Multiple attribution methods help validate findings
• Attention analysis reveals model focus patterns
• Counterfactual analysis shows decision boundaries
    """
    
    axes[1, 1].text(0.05, 0.95, summary_text, transform=axes[1, 1].transAxes, 
                   fontsize=10, verticalalignment='top', fontfamily='monospace')
    axes[1, 1].axis('off')
    
    plt.tight_layout()
    summary_path = os.path.join(output_dir, "xai_summary_report.png")
    plt.savefig(summary_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    # Save detailed results as JSON
    json_path = os.path.join(output_dir, "xai_results.json")
    with open(json_path, 'w') as f:
        # Convert results to JSON-serializable format
        json_results = []
        for r in results:
            json_result = {
                'sample_idx': r['sample_idx'],
                'prediction': r['prediction'],
                'ground_truth': r['ground_truth'],
                'confidence': float(r['confidence']),
                'techniques_applied': list(r['techniques'].keys())
            }
            json_results.append(json_result)
        
        json.dump({
            'summary': {
                'total_samples': total_samples,
                'accuracy': float(accuracy),
                'avg_correct_confidence': float(np.mean(correct_confidences)) if correct_confidences else 0,
                'avg_incorrect_confidence': float(np.mean(incorrect_confidences)) if incorrect_confidences else 0,
                'technique_success_rates': {k: float(v/total_samples) for k, v in technique_success.items()}
            },
            'detailed_results': json_results
        }, f, indent=2)
    
    print(f"Summary report saved to: {summary_path}")
    print(f"Detailed results saved to: {json_path}")


def _create_tcav_summary(tcav_scores, output_dir):
    """Create a summary visualization for TCAV scores."""
    if not tcav_scores:
        return
    
    fig, ax = plt.subplots(figsize=(12, 8))
    
    layers = list(tcav_scores.keys())
    scores = list(tcav_scores.values())
    
    bars = ax.bar(range(len(layers)), scores, color='skyblue', alpha=0.7)
    ax.set_xticks(range(len(layers)))
    ax.set_xticklabels([layer.split('.')[-1] for layer in layers], rotation=45, ha='right')
    ax.set_ylabel('TCAV Score')
    ax.set_title('Testing with Concept Activation Vectors (TCAV)\nConcept: Fake vs Real')
    ax.set_ylim(0, 1)
    ax.axhline(y=0.5, color='r', linestyle='--', alpha=0.5, label='Random Baseline')
    
    # Add value labels on bars
    for bar, score in zip(bars, scores):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
               f'{score:.3f}', ha='center', va='bottom')
    
    ax.legend()
    plt.tight_layout()
    
    tcav_path = os.path.join(output_dir, "tcav_scores_summary.png")
    plt.savefig(tcav_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"TCAV summary saved to: {tcav_path}")


# ================================================================================
# NEW: COMPREHENSIVE TESTING AND EVALUATION FUNCTION
# =================================================================================
def test_and_evaluate(model_path, test_loader, config):
    """
    Loads a saved model, performs inference on the test set, and calculates
    detailed evaluation metrics.

    Args:
        model_path (str): Path to the saved .pth model state_dict.
        test_loader (DataLoader): The DataLoader for the test set.
        config (Config): The configuration object.
    """
    print("\n" + "="*50)
    print("--- Starting Final Model Evaluation on Test Set ---")
    print(f"Loading model from: {model_path}")

    # 1. Initialize and load the model
    # We first create an instance of the model, then load the saved weights into it.
    device = config.DEVICE
    model = PinpointTransformer(config).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval() # IMPORTANT: Set model to evaluation mode

    all_labels = []
    all_preds = []

    # 2. Run inference on the test set
    with torch.no_grad(): # No need to calculate gradients
        progress_bar = tqdm(test_loader, desc="Testing", leave=False)
        for batch in progress_bar:
            if batch is None: continue

            # Move data to the configured device
            video = batch['video'].to(device)
            audio = batch['audio'].to(device)
            video_mask = batch['video_mask'].to(device)
            labels = batch['label'] # Keep labels on CPU for easier collection

            # Get model predictions (logits)
            cls_logits, _, _ = model(video, audio, video_mask)

            # Convert logits to binary predictions (0 or 1)
            # Sigmoid -> probability, > 0.5 -> class 1 (Fake)
            preds = (torch.sigmoid(cls_logits) > 0.5).squeeze(1).cpu().numpy().astype(int)

            # Collect the ground truth labels and predictions
            all_labels.extend(labels.numpy().astype(int))
            all_preds.extend(preds)

    print("Testing complete. Calculating metrics...")

    # 3. Calculate and Print Metrics
    if not all_labels:
        print("Warning: No samples were evaluated. Cannot calculate metrics.")
        return

    # Basic metrics
    precision = precision_score(all_labels, all_preds, zero_division=0)
    recall = recall_score(all_labels, all_preds, zero_division=0)
    f1 = f1_score(all_labels, all_preds, zero_division=0)

    # Classification Report (provides a detailed breakdown)
    # target_names makes the report more readable
    report = classification_report(all_labels, all_preds, target_names=['Real (0)', 'Fake (1)'], zero_division=0)

    # Confusion Matrix (provides insight into error types)
    cm = confusion_matrix(all_labels, all_preds)

    # 4. Display Results
    print("\n--- Final Test Results ---")
    print(f"Total Samples Tested: {len(all_labels)}")
    print("\n--- Key Metrics ---")
    print(f"  - Precision: {precision:.4f}")
    print(f"  - Recall:    {recall:.4f}")
    print(f"  - F1-Score:  {f1:.4f}")

    print("\n--- Classification Report ---")
    print(report)

    print("\n--- Confusion Matrix ---")
    print("        Predicted")
    print("       Real  Fake")
    print(f"Real   {cm[0,0]:<5} {cm[0,1]:<5}")
    print(f"Fake   {cm[1,0]:<5} {cm[1,1]:<5}")
    tn, fp, fn, tp = cm.ravel() if cm.size == 4 else (cm[0,0], 0, 0, 0) # Handle edge cases
    print(f"(TN: {tn}, FP: {fp}, FN: {fn}, TP: {tp})")

    print("="*50 + "\n")

if __name__ == '__main__':
    config = Config()
    if config.TESTING:
        config.EPOCHS = 2 # Also reduce epochs for a quick test
        print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
        print("!!!         RUNNING IN TEST MODE         !!!")
        print("!!! Datasets will be limited to 100 samples !!!")
        print("!!! Epochs reduced to 2                  !!!")
        print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
    else:
        print("****************************************")
        print("***      STARTING FULL TRAINING RUN      ***")
        print(f"***      Epochs: {config.EPOCHS}                   ***")
        print(f"***      Using complete dataset        ***")
        print("****************************************")


    print("\nAnti-Shortcut Configuration loaded. Using device:", config.DEVICE)
    ### MODIFIED: Updated print statement for new directory config ###
    print(f"Data will be loaded from: {config.DATA_DIRECTORY}")
    print(f"Metadata will be loaded from: {config.METADATA_PATH}")
    print(f"Curriculum: {config.CURRICULUM_PHASE1_EPOCHS} (Mask) + {config.CURRICULUM_PHASE2_EPOCHS} (Sync) + remaining (Full) Epochs")
    print("-" * 50)

    # Can be set to False for a minor performance boost in full training.
    # Keep True if you suspect gradient issues.
    torch.autograd.set_detect_anomaly(False)

    ### MODIFIED: Simplified directory check ###
    if not os.path.exists(config.DATA_DIRECTORY):
        print(f"!!! ERROR: Data directory NOT FOUND: {config.DATA_DIRECTORY} !!!")
        print("!!! Please run `unified_preprocessing.py` first.          !!!")
    ### END MODIFIED BLOCK ###
    else:
        try:
            print("\n--- [1/6] Setting up model and data ---")
            model = PinpointTransformer(config).to(config.DEVICE)

            print("Creating DataLoaders...")
            # Datasets are created first to allow for class weight calculation
            train_dataset = LAVDFDataset(config, split='train')
            dev_dataset = LAVDFDataset(config, split='dev')
            test_dataset = LAVDFDataset(config, split='test')

            # --- NEW: Calculate pos_weight to handle data imbalance ---
            print("Calculating class weights for the loss function...")
            num_real = sum(1 for s in train_dataset.samples if not s['is_fake'])
            num_fake = sum(1 for s in train_dataset.samples if s['is_fake'])

            pos_weight_tensor = None # Default for balanced datasets
            if num_real > 0 and num_fake > 0:
                # This is the crucial step to combat the model guessing the majority class.
                # The weight is applied to the positive class (Fake, label=1).
                # Formula is num_negatives / num_positives.
                pos_weight_value = num_real / num_fake
                print(f"Full training set composition: {num_real} Real, {num_fake} Fake.")
                print(f"Applying pos_weight to BCEWithLogitsLoss: {pos_weight_value:.4f}")
                pos_weight_tensor = torch.tensor([pos_weight_value], device=config.DEVICE)
            else:
                print("Warning: Training data contains only one class. Loss weighting is disabled.")

            # Define loss functions with the calculated class weight
            loss_fns = {
                'classification': nn.BCEWithLogitsLoss(pos_weight=pos_weight_tensor),
                'offset': nn.CrossEntropyLoss(),
                'attention': SynchronizationLoss(config)
            }

            train_loader = DataLoader(train_dataset, batch_size=config.BATCH_SIZE, shuffle=True, collate_fn=collate_fn, num_workers=2, pin_memory=True)
            dev_loader = DataLoader(dev_dataset, batch_size=config.BATCH_SIZE, shuffle=False, collate_fn=collate_fn, num_workers=2)
            test_loader = DataLoader(test_dataset, batch_size=config.BATCH_SIZE, shuffle=False, collate_fn=collate_fn, num_workers=2)

            optimizer = torch.optim.AdamW(model.parameters(), lr=config.LEARNING_RATE, weight_decay=config.WEIGHT_DECAY)
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config.EPOCHS * len(train_loader))
            scaler = GradScaler()

            print("\n--- [2/6] Starting Curriculum Training & Validation ---")
            start_time = time.time()
            best_val_loss = float('inf')
            best_model_path = "best_pinpoint_model_antisocial.pth"

            for epoch in range(config.EPOCHS):
                print(f"\n===== Epoch {epoch + 1}/{config.EPOCHS} =====")
                train_dataset.set_epoch(epoch)
                train_loss, train_acc = train_one_epoch(model, train_loader, optimizer, scheduler, loss_fns, config.DEVICE, config, epoch, scaler)
                print(f"Epoch {epoch + 1} Training   -> Avg Loss: {train_loss:.4f}, Accuracy: {train_acc:.4f}")

                val_loss, val_acc = evaluate_model(model, dev_loader, loss_fns, config.DEVICE, config)
                print(f"Epoch {epoch + 1} Validation -> Avg Loss: {val_loss:.4f}, Accuracy: {val_acc:.4f}")

                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    torch.save(model.state_dict(), best_model_path)
                    print(f"  -> New best model saved to {best_model_path} (Val Loss: {val_loss:.4f})")

            print(f"\n--- [3/6] Training Finished in {(time.time() - start_time)/60:.2f} minutes ---")
            print(f"\n--- [4/6] Loading best model and running on Test Set ---")
            if os.path.exists(best_model_path):
                # Use the dedicated test function for a full report
                test_and_evaluate(best_model_path, test_loader, config)
            else:
                print("No best model was saved. Skipping final test evaluation.")

            print("\n--- [5/8] Generating Example Attention Visualization ---")
            real_indices = [i for i, s in enumerate(test_dataset.samples) if not s['is_fake']]
            fake_indices = [i for i, s in enumerate(test_dataset.samples) if s['is_fake']]
            if real_indices: visualize_attention(model, test_dataset, config.DEVICE, config, sample_idx=random.choice(real_indices))
            if fake_indices: visualize_attention(model, test_dataset, config.DEVICE, config, sample_idx=random.choice(fake_indices))

            visualize_grad_cam_grid(model, test_dataset, config.DEVICE, config, num_samples=5)

            print("\n--- [6/8] Comprehensive Explainable AI Analysis ---")
            # Determine number of samples for XAI analysis based on testing mode
            xai_num_samples = 5 if config.TESTING else 10
            xai_results = comprehensive_xai_evaluation(
                model, test_dataset, config, 
                num_samples=xai_num_samples, 
                output_dir="comprehensive_xai_outputs"
            )
            
            print("\n--- [7/8] Concept Activation Vectors (CAV) and TCAV Analysis ---")
            demonstrate_cav_tcav(model, test_dataset, config, output_dir="cav_tcav_outputs")
            
            print("\n--- [8/8] All Analysis Complete ---")
            print("Generated comprehensive explainability analysis including:")
            print("  • SHAP (Shapley Additive Explanations)")
            print("  • Layer-wise Relevance Propagation (LRP)")  
            print("  • Integrated Gradients")
            print("  • Attention Rollout and Flow Analysis")
            print("  • Counterfactual Explanations")
            print("  • Concept Activation Vectors (CAV) and Testing with CAV (TCAV)")
            print("  • Unified XAI Visualization Framework")
            
            if xai_results:
                print(f"\nXAI Summary:")
                successful_analyses = len(xai_results)
                print(f"  - Successfully analyzed {successful_analyses} samples")
                print(f"  - Results saved to: comprehensive_xai_outputs/")
                print(f"  - CAV/TCAV analysis saved to: cav_tcav_outputs/")
                print("  - Check the summary reports for detailed insights")

        except (FileNotFoundError, RuntimeError, Exception) as e:
            import traceback
            print(f"\nAn error occurred during execution: {e}")
            traceback.print_exc()