# =================================================================================
# xai_insights.py
#
# A Comprehensive Explainable AI (XAI) Toolkit for the Pinpoint Model.
#
# Description:
# This script loads a pre-trained Pinpoint model and applies various XAI
# techniques to understand its decision-making process for deepfake detection.
# The generated visualizations and analyses are designed to be publication-ready
# for top-tier academic venues.
#
# Author: Gemini
# Date: August 27, 2025
# =================================================================================

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
import numpy as np
import os
import random
import matplotlib.pyplot as plt
import seaborn as sns
import cv2
from tqdm import tqdm
import shap

# --- Crucial: Import the necessary components from your PinPoint script ---
# This requires `xai_insights.py` to be in the same directory as `PinPoint.py`
try:
    from PinPoint import PinpointTransformer, LAVDFDataset, collate_fn, Config as PinpointConfig, generate_grad_cam
except ImportError:
    print("FATAL ERROR: Could not import from PinPoint.py.")
    print("Please ensure 'xai_insights.py' is in the same directory as 'PinPoint.py'.")
    exit()


# =================================================================================
# 1. XAI CONFIGURATION
# =================================================================================
class XAIConfig:
    """Configuration class specifically for the XAI analysis."""
    # --- PATHS (NEEDS TO BE MODIFIED BY THE USER) ---
    # Path to the trained model checkpoint
    CHECKPOINT_PATH = "/kaggle/input/pinpoint-xai/pytorch/default/1/best_pinpoint_model_antisocial.pth"
    # Path to the preprocessed data directory
    DATA_DIRECTORY = "/kaggle/input/new-model-unified-pre-processing/preprocessed_data"
    # Path to the unified metadata file
    METADATA_PATH = "/kaggle/input/new-model-unified-pre-processing/preprocessed_data/unified_metadata.json"
    # Directory to save all XAI outputs
    OUTPUT_DIR = "./xai_outputs"

    # --- DEVICE ---
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

    # --- Analysis Parameters ---
    NUM_SAMPLES_TO_ANALYZE = 5 # How many samples to run through the XAI pipeline
    # Select a specific frame for Grad-CAM to focus on (e.g., middle frame)
    GRAD_CAM_TARGET_FRAME = PinpointConfig.NUM_FRAMES // 2


# =================================================================================
# 2. HELPER FUNCTIONS
# =================================================================================

def setup_environment(config):
    """Prepares the environment, loads model and data."""
    print("--- [1/5] Setting up XAI Environment ---")
    
    # Create output directory
    os.makedirs(config.OUTPUT_DIR, exist_ok=True)
    print(f"XAI outputs will be saved to: {config.OUTPUT_DIR}")

    # --- Load Model ---
    print(f"Loading model from checkpoint: {config.CHECKPOINT_PATH}")
    if not os.path.exists(config.CHECKPOINT_PATH):
        raise FileNotFoundError(f"Checkpoint not found at {config.CHECKPOINT_PATH}")
    
    # --- MODIFIED BLOCK ---
    # We now create and return the original PinpointConfig
    model_config = PinpointConfig()
    model = PinpointTransformer(model_config).to(config.DEVICE)
    model.load_state_dict(torch.load(config.CHECKPOINT_PATH, map_location=config.DEVICE))
    model.eval()
    print("Model loaded successfully and set to evaluation mode.")

    print("Loading test dataset for analysis...")
    # The dataset needs its own copy of the config with the correct paths
    dataset_config = PinpointConfig()
    dataset_config.DATA_DIRECTORY = config.DATA_DIRECTORY
    dataset_config.METADATA_PATH = config.METADATA_PATH
    
    test_dataset = LAVDFDataset(dataset_config, split='test')
    if not test_dataset:
        raise ValueError("Failed to load the test dataset. Please check paths and data integrity.")
    
    print(f"Loaded {len(test_dataset)} samples from the test set.")
    
    # Return the model_config as well
    return model, test_dataset, model_config
    # --- END MODIFIED BLOCK ---


def get_prediction_details(model, batch, device):
    """Runs a single batch through the model and returns detailed predictions."""
    video = batch['video'].to(device)
    audio = batch['audio'].to(device)
    video_mask = batch['video_mask'].to(device)
    
    with torch.no_grad():
        cls_logits, _, attention_map = model(video, audio, video_mask)
        
    prob = torch.sigmoid(cls_logits).squeeze().item()
    prediction = "Fake" if prob > 0.5 else "Real"
    ground_truth = "Fake" if batch['is_fake'].item() else "Real"
    is_correct = (prediction == ground_truth)
    
    return {
        "prob": prob,
        "prediction": prediction,
        "ground_truth": ground_truth,
        "is_correct": is_correct,
        "attention_map": attention_map.squeeze(0).cpu().numpy()
    }


# =================================================================================
# 3. XAI TECHNIQUE IMPLEMENTATIONS
# =================================================================================

def analyze_attention_maps(details, sample_idx, config):
    """Visualizes and saves the cross-modal attention map."""
    attention_map = details['attention_map']
    
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(attention_map, cmap='viridis', ax=ax)
    
    title = (f"Cross-Attention Map for Sample {sample_idx}\n"
             f"GT: {details['ground_truth']} | Pred: {details['prediction']} ({details['prob']:.2f}) "
             f"| Result: {'Correct' if details['is_correct'] else 'Incorrect'}")
    ax.set_title(title, fontsize=14)
    ax.set_xlabel("Video Frames", fontsize=12)
    ax.set_ylabel("Audio Time Steps (MFCCs)", fontsize=12)
    
    save_path = os.path.join(config.OUTPUT_DIR, f"sample_{sample_idx}_attention_map.png")
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close(fig)
    print(f"  -> Saved attention map to {save_path}")

def analyze_grad_cam(model, dataset, sample_idx, config, model_config):
    """Generates and saves a Grad-CAM visualization."""
    sample = dataset[sample_idx]
    if sample is None: return

    # We need the original, unprocessed video frame for visualization
    original_video_uint8 = torch.load(dataset.samples[sample_idx]['video_path'])
    target_frame_np = original_video_uint8[config.GRAD_CAM_TARGET_FRAME].permute(1, 2, 0).numpy()

    # Get the processed tensors for the model
    video_tensor = sample['video']
    audio_tensor = sample['audio']
    # --- MODIFIED LINE ---
    # We now pass the required 'model_config' object
    heatmap = generate_grad_cam(model, video_tensor, audio_tensor, config.GRAD_CAM_TARGET_FRAME, config.DEVICE, model_config)
    # --- END MODIFIED LINE ---
    # Generate the heatmap
    if heatmap is None:
        print("  -> Grad-CAM generation failed for this sample.")
        return

    # Overlay the heatmap on the original frame
    heatmap_resized = cv2.resize(heatmap, (target_frame_np.shape[1], target_frame_np.shape[0]))
    heatmap_color = cv2.applyColorMap(np.uint8(255 * heatmap_resized), cv2.COLORMAP_JET)
    superimposed_img = cv2.addWeighted(target_frame_np, 0.6, heatmap_color, 0.4, 0)
    superimposed_img = cv2.cvtColor(superimposed_img, cv2.COLOR_BGR2RGB)

    # Plotting
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))
    axes[0].imshow(target_frame_np / 255.0) # Normalize for display
    axes[0].set_title(f"Original Frame {config.GRAD_CAM_TARGET_FRAME}")
    axes[0].axis('off')

    axes[1].imshow(superimposed_img)
    axes[1].set_title("Grad-CAM Overlay")
    axes[1].axis('off')

    details = get_prediction_details(model, collate_fn([sample]), config.DEVICE)
    fig.suptitle(f"Grad-CAM Saliency for Sample {sample_idx} (Pred: {details['prediction']} @ {details['prob']:.2f})", fontsize=16)

    save_path = os.path.join(config.OUTPUT_DIR, f"sample_{sample_idx}_grad_cam.png")
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close(fig)
    print(f"  -> Saved Grad-CAM to {save_path}")

def analyze_occlusion_sensitivity(model, sample, sample_idx, config):
    """Performs occlusion sensitivity analysis on video frames."""
    print(f"  -> Running Occlusion Sensitivity for sample {sample_idx}...")
    
    # Get baseline prediction
    batch = collate_fn([sample])
    baseline_details = get_prediction_details(model, batch, config.DEVICE)
    baseline_prob = baseline_details['prob']
    
    video_tensor = batch['video'].to(config.DEVICE)
    audio_tensor = batch['audio'].to(config.DEVICE)
    
    H, W = PinpointConfig.VIDEO_SIZE
    occlusion_size = (H // 4, W // 4)
    stride = occlusion_size[0] // 2

    heatmap = torch.zeros((H, W), device='cpu')
    
    # Iterate over the image with a sliding window
    for top in tqdm(range(0, H - occlusion_size[0], stride), leave=False, desc="Occluding"):
        for left in range(0, W - occlusion_size[1], stride):
            video_occluded = video_tensor.clone()
            # Occlude a patch in all frames
            video_occluded[:, :, :, top:top+occlusion_size[0], left:left+occlusion_size[1]] = 0

            with torch.no_grad():
                logits, _, _ = model(video_occluded, audio_tensor)
                prob_occluded = torch.sigmoid(logits).squeeze().item()
            
            # The change in probability is our sensitivity score
            score = baseline_prob - prob_occluded
            heatmap[top:top+occlusion_size[0], left:left+occlusion_size[1]] += score

    # Plotting
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))
    original_frame = sample['video'][config.GRAD_CAM_TARGET_FRAME].permute(1, 2, 0).numpy()
    # Un-normalize for visualization
    mean = np.array(PinpointConfig.NORM_MEAN)
    std = np.array(PinpointConfig.NORM_STD)
    original_frame = std * original_frame + mean
    original_frame = np.clip(original_frame, 0, 1)

    axes[0].imshow(original_frame)
    axes[0].set_title(f"Original Frame {config.GRAD_CAM_TARGET_FRAME}")
    axes[0].axis('off')
    
    sns.heatmap(heatmap.numpy(), cmap='Reds', ax=axes[1], cbar=True)
    axes[1].set_title("Occlusion Sensitivity Heatmap")
    axes[1].axis('off')

    fig.suptitle(f"Occlusion Sensitivity for Sample {sample_idx}\nBaseline Fake Prob: {baseline_prob:.3f}", fontsize=16)
    
    save_path = os.path.join(config.OUTPUT_DIR, f"sample_{sample_idx}_occlusion_map.png")
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close(fig)
    print(f"  -> Saved Occlusion Sensitivity map to {save_path}")

def analyze_with_shap(model, dataset, sample_idx, config):
    """
    Explains a prediction using SHAP. This is computationally expensive.
    We'll create a simplified explanation focusing on temporal segments.
    """
    print(f"  -> Running SHAP analysis for sample {sample_idx} (This may take a while)...")
    
    sample = dataset[sample_idx]
    batch = collate_fn([sample])
    video_tensor = batch['video'].to(config.DEVICE)
    audio_tensor = batch['audio'].to(config.DEVICE)
    
    # 1. Define the prediction function for SHAP
    # SHAP explainer expects a function that takes a numpy array and returns a numpy array.
    # We will mask temporal segments (chunks of frames/audio) and see the impact.
    # Let's divide video into 6 segments and audio into 6 segments.
    num_segments = 6
    video_seg_len = PinpointConfig.NUM_FRAMES // num_segments
    audio_seg_len = audio_tensor.shape[1] // num_segments

    def predict_for_shap(x_mask):
        # x_mask is a (num_samples, num_features) numpy array where features are our segments
        num_samples = x_mask.shape[0]
        preds = np.zeros(num_samples)
        
        for i in tqdm(range(num_samples), leave=False, desc="SHAP Samples"):
            video_masked = video_tensor.clone()
            audio_masked = audio_tensor.clone()
            
            # Apply masks based on the input vector x_mask[i]
            for seg_idx in range(num_segments): # Video segments
                if x_mask[i, seg_idx] == 0:
                    start = seg_idx * video_seg_len
                    end = start + video_seg_len
                    video_masked[:, start:end] = 0
            
            for seg_idx in range(num_segments): # Audio segments
                if x_mask[i, num_segments + seg_idx] == 0:
                    start = seg_idx * audio_seg_len
                    end = start + audio_seg_len
                    audio_masked[:, start:end] = 0

            with torch.no_grad():
                logits, _, _ = model(video_masked, audio_masked)
                preds[i] = torch.sigmoid(logits).squeeze().item()
        return preds

    # 2. Create the SHAP explainer
    # A background distribution of "off" segments (all masked)
    background = np.zeros((1, num_segments * 2))
    # The instance to explain (all segments "on")
    instance_to_explain = np.ones((1, num_segments * 2))
    
    explainer = shap.KernelExplainer(predict_for_shap, background)
    
    # 3. Get SHAP values
    # nsamples is the number of perturbations to run. Higher is better but slower.
    shap_values = explainer.shap_values(instance_to_explain, nsamples=100)

    # 4. Visualize the results
    feature_names = [f'VidSeg{i+1}' for i in range(num_segments)] + \
                    [f'AudSeg{i+1}' for i in range(num_segments)]

    shap.summary_plot(shap_values, features=instance_to_explain, feature_names=feature_names, plot_type="bar", show=False)
    
    plt.title(f"SHAP: Feature Importance for Sample {sample_idx}", fontsize=16)
    plt.xlabel("SHAP Value (impact on model output 'Fake Probability')")
    
    save_path = os.path.join(config.OUTPUT_DIR, f"sample_{sample_idx}_shap_plot.png")
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    print(f"  -> Saved SHAP analysis to {save_path}")


# =================================================================================
# 4. MAIN EXECUTION BLOCK
# =================================================================================
def main():
    """Main function to run the entire XAI analysis pipeline."""
    config = XAIConfig()
    
    try:
        model, test_dataset, model_config = setup_environment(config)
        
        # --- Select diverse samples for analysis ---
        print("\n--- [2/5] Selecting Diverse Samples for Analysis ---")
        correct_fakes = [i for i, s in enumerate(test_dataset.samples) if s['is_fake']]
        correct_reals = [i for i, s in enumerate(test_dataset.samples) if not s['is_fake']]
        
        # Note: Finding incorrect predictions might require running inference first.
        # For simplicity, we focus on correctly classified samples.
        random.shuffle(correct_fakes)
        random.shuffle(correct_reals)
        
        sample_indices_to_analyze = (correct_fakes[:config.NUM_SAMPLES_TO_ANALYZE//2] + 
                                     correct_reals[:config.NUM_SAMPLES_TO_ANALYZE//2])
        
        print(f"Selected {len(sample_indices_to_analyze)} samples: {sample_indices_to_analyze}")

        for i, sample_idx in enumerate(sample_indices_to_analyze):
            print(f"\n--- [{(i+3)}/5] Analyzing Sample {sample_idx} ({i+1}/{len(sample_indices_to_analyze)}) ---")
            
            sample = test_dataset[sample_idx]
            if sample is None:
                print(f"  -> Skipping sample {sample_idx} due to loading error.")
                continue
                
            batch = collate_fn([sample])
            details = get_prediction_details(model, batch, config.DEVICE)
            print(f"  Sample Info: GT='{details['ground_truth']}', Pred='{details['prediction']}' ({details['prob']:.3f})")

            # --- Run XAI Methods ---
            
            # 1. Attention Map Visualization
            analyze_attention_maps(details, sample_idx, config)
            
            # 2. Grad-CAM (Visual Saliency)
            analyze_grad_cam(model, test_dataset, sample_idx, config, model_config)
            
            # 3. Occlusion Sensitivity
            analyze_occlusion_sensitivity(model, sample, sample_idx, config)
            
            # 4. SHAP (Temporal Importance) - Optional, as it's slow
            analyze_with_shap(model, test_dataset, sample_idx, config)

        print("\n--- XAI Analysis Complete ---")
        print(f"All outputs have been saved to the '{config.OUTPUT_DIR}' directory.")

    except (FileNotFoundError, ValueError, ImportError) as e:
        print(f"\nAn error occurred: {e}")
    except Exception as e:
        import traceback
        print(f"\nAn unexpected error occurred: {e}")
        traceback.print_exc()

if __name__ == '__main__':
    main()