import torch
import torchaudio
import torchvision.transforms as transforms
import numpy as np
import cv2
import json
import os
from tqdm import tqdm

# =================================================================================
# UNIFIED PREPROCESSING SCRIPT FOR LAV-DF
#
# This script is designed to process the entire LAV-DF dataset (train, dev, and test splits)
# in a single, continuous run. This approach is critical to prevent data leakage by
# ensuring that all data samples are created in the exact same environment with the
# exact same code, eliminating any batch-specific artifacts that a model could exploit.
#
# HOW TO USE:
# 1. Update the BASE_DIR and OUTPUT_DIR paths in the Config class below.
# 2. Run the script once.
# 3. This will create a 'preprocessed_data' directory containing 'train', 'dev',
#    and 'test' subfolders with the processed tensors, along with a single
#    'unified_metadata.json' file that should be used for all subsequent training
#    and evaluation.
#
# Author: AI Assistant
# Date: October 27, 2023
# ---------------------------------------------------------------------------------


# =================================================================================
# 1. CONFIGURATION
# =================================================================================
class Config:
    # --- Input Paths ---
    # Point this to the root directory of the original LAV-DF dataset
    BASE_DIR = "/kaggle/input/localized-audio-visual-deepfake-dataset-lav-df/LAV-DF"
    METADATA_PATH = os.path.join(BASE_DIR, "metadata.json")

    # --- Output Path ---
    # All processed data and the final metadata file will be saved here.
    OUTPUT_DIR = "/kaggle/working/preprocessed_data"

    # --- Data Processing Settings ---
    NUM_FRAMES = 64
    VIDEO_SIZE = (128, 128)
    NUM_MFCC = 13

    # --- MODIFIED: Sample Count Limiter ---
    # Set to a dictionary to limit samples per split, or None for no limit.
    # This is useful for quick testing or if you have limited disk space.
    # Example for a full run: MAX_SAMPLES_PER_SPLIT = None
    # Example for a quick test: MAX_SAMPLES_PER_SPLIT = {'train': 100, 'dev': 50, 'test': 50}
    MAX_SAMPLES_PER_SPLIT = {'train': 8000, 'dev': 2000, 'test': 2000}
    
    # IMPORTANT: Normalization is NOT applied here. It is deferred to the training
    # script to be done on-the-fly. The video tensors are saved as uint8 to
    # significantly reduce storage space.
    NORM_MEAN = [0.485, 0.456, 0.406]
    NORM_STD = [0.229, 0.224, 0.225]

# Instantiate the config
config = Config()


# =================================================================================
# 2. CORE PROCESSING FUNCTION
# =================================================================================
def process_video_and_audio(video_path, config, video_transform):
    """
    Extracts frames and MFCCs from a single video file.
    Returns a video tensor and an audio tensor.
    """
    try:
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
             tqdm.write(f"Warning: Could not open video file: {video_path}")
             return None, None
             
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if total_frames < config.NUM_FRAMES:
             tqdm.write(f"Warning: Not enough frames ({total_frames}) in {video_path}, skipping.")
             cap.release()
             return None, None

        # Evenly space frame extraction
        frame_indices = np.linspace(0, total_frames - 1, config.NUM_FRAMES, dtype=int)
        frames = []
        for i in frame_indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, i)
            ret, frame = cap.read()
            if not ret: continue
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(video_transform(frame))
        cap.release()

        if len(frames) != config.NUM_FRAMES:
            tqdm.write(f"Warning: Could not extract the required number of frames from {video_path}")
            return None, None

        video_tensor = torch.stack(frames)
    except Exception as e:
        tqdm.write(f"Error processing video frames for {video_path}: {e}")
        return None, None

    try:
        waveform, sample_rate = torchaudio.load(video_path)
        mfcc_transform = torchaudio.transforms.MFCC(
            sample_rate=sample_rate, n_mfcc=config.NUM_MFCC
        )
        mfccs = mfcc_transform(waveform).squeeze(0).transpose(0, 1)

        if mfccs.shape[0] == 0:
            tqdm.write(f"Warning: Audio processing resulted in empty MFCCs for {video_path}")
            return None, None
    except Exception as e:
        tqdm.write(f"Error processing audio for {video_path}: {e}")
        return None, None

    return video_tensor, mfccs


# =================================================================================
# 3. MAIN EXECUTION
# =================================================================================
def main():
    print("--- Starting Unified Preprocessing for LAV-DF ---")
    print(f"This will process all splits (train, dev, test) in a single run.")
    print(f"Output will be saved to: {config.OUTPUT_DIR}")

    # Create the main output directory
    os.makedirs(config.OUTPUT_DIR, exist_ok=True)

    # Load the original, canonical metadata
    with open(config.METADATA_PATH, 'r') as f:
        all_metadata = json.load(f)
    
    print(f"Found {len(all_metadata)} total video entries in the original metadata file.")

    # --- MODIFIED: Apply sample limits if configured ---
    if config.MAX_SAMPLES_PER_SPLIT:
        print("\n--- Applying sample limits per split ---")
        
        # Group metadata by split
        metadata_by_split = {'train': [], 'dev': [], 'test': []}
        for item in all_metadata:
            split = item.get('split')
            if split in metadata_by_split:
                metadata_by_split[split].append(item)
        
        metadata_to_process = []
        for split, items in metadata_by_split.items():
            limit = config.MAX_SAMPLES_PER_SPLIT.get(split)
            print(f"Split '{split}': Found {len(items)} total items.")
            
            if limit is not None and len(items) > limit:
                print(f"  -> Applying limit: Selecting {limit} random samples.")
                import random
                random.shuffle(items) # Shuffle to get a random subset
                metadata_to_process.extend(items[:limit])
            else:
                print(f"  -> Using all {len(items)} items (limit not specified or not exceeded).")
                metadata_to_process.extend(items)
        
        # Replace the full list with our new, limited list
        all_metadata = metadata_to_process
        print(f"\nTotal videos to preprocess after applying limits: {len(all_metadata)}")


    # Define the video transformation pipeline.
    # We convert to tensor, which scales to [0.0, 1.0], and then resize.
    # We do NOT normalize here; that will be done during training.
    video_transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize(config.VIDEO_SIZE),
        transforms.ToTensor(),
    ])

    final_metadata = []

    # Process every single item in the metadata file
    for item in tqdm(all_metadata, desc="Preprocessing All Videos"):
        video_filename = item['file']
        video_path = os.path.join(config.BASE_DIR, video_filename)

        # Ensure the original video file exists
        if not os.path.exists(video_path):
            tqdm.write(f"Warning: Original file not found, skipping: {video_path}")
            continue

        base_name = os.path.splitext(os.path.basename(video_filename))[0]
        
        # Determine the output directory based on the 'split' from the metadata
        split_name = item.get('split')
        if not split_name:
            tqdm.write(f"Warning: No split specified for {video_filename}, skipping.")
            continue
            
        split_dir = os.path.join(config.OUTPUT_DIR, split_name)
        os.makedirs(split_dir, exist_ok=True)
        
        # Define output paths for the tensors
        video_out_path = os.path.join(split_dir, f"{base_name}_video.pt")
        audio_out_path = os.path.join(split_dir, f"{base_name}_audio.pt")

        # Process the video and audio
        video_tensor, audio_tensor = process_video_and_audio(video_path, config, video_transform)

        if video_tensor is None or audio_tensor is None:
            tqdm.write(f"Skipping {video_filename} due to a processing error.")
            continue

        # CRITICAL: Convert float tensor [0.0, 1.0] to uint8 [0, 255] for saving
        # This dramatically reduces file size and is a common practice.
        # The training script will convert it back to float and normalize.
        video_tensor_to_save = (video_tensor * 255).to(torch.uint8)
        
        # Save the tensors
        torch.save(video_tensor_to_save, video_out_path)
        torch.save(audio_tensor, audio_out_path)

        # Create the new metadata entry for this item
        new_item = item.copy()
        
        # Store paths relative to the OUTPUT_DIR for portability
        new_item['preprocessed_video_path'] = os.path.relpath(video_out_path, config.OUTPUT_DIR)
        new_item['preprocessed_audio_path'] = os.path.relpath(audio_out_path, config.OUTPUT_DIR)
        
        # Add a consistent 'label' field for easier use in the training script
        if new_item.get('n_fakes', 0) == 0:
            new_item['label'] = 'real'
        else:
            new_item['label'] = 'fake'
            
        final_metadata.append(new_item)

    # Save the single, unified metadata file
    output_metadata_path = os.path.join(config.OUTPUT_DIR, "unified_metadata.json")
    
    # Sort metadata by the original filename for consistency
    final_metadata.sort(key=lambda x: x['file'])
    
    with open(output_metadata_path, 'w') as f:
        json.dump(final_metadata, f, indent=4)

    print("\n--- Unified Preprocessing Complete ---")
    print(f"Successfully processed and saved {len(final_metadata)} items.")
    print(f"The new unified metadata file has been saved to: {output_metadata_path}")
    print("You should now use this directory and metadata file for training and evaluation.")


if __name__ == '__main__':
    main() 