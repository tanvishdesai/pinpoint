# ust XAI
import torch
import os
import random
from torch.utils.data import DataLoader

# --- Import necessary classes and functions from your main script ---
# This assumes 'PinPoint.py' is in the same directory.
from PinPoint import (
    Config,
    PinpointTransformer,
    LAVDFDataset,
    collate_fn,
    comprehensive_xai_evaluation,
    demonstrate_cav_tcav,
    QuantitativeXAIMetrics,
)

# ================================================================================
# SCRIPT CONFIGURATION (HARDCODED PARAMETERS)
# ================================================================================

# --- REQUIRED: PLEASE MODIFY THIS PATH ---
# Point this to the location of your trained model's .pth file.
MODEL_CHECKPOINT_PATH = "/kaggle/input/pp-xai-full-model-v1/best_pinpoint_model_antisocial.pth"

# --- OPTIONAL: Modify these if needed ---
# Directory where all XAI analysis results will be saved.
XAI_OUTPUT_DIR = "xai_evaluation_results"

# Number of samples to use for the detailed, per-sample XAI analysis.
# This generates visualizations for SHAP, LRP, IG, etc. for each sample.
NUM_SAMPLES_FOR_COMPREHENSIVE_XAI = 8 # (4 fake, 4 real if possible)

# Number of samples to use for calculating the quantitative XAI metrics.
# These metrics (sparsity, consistency) are averaged across this many samples.
# A larger number gives more stable metrics but takes longer.
NUM_SAMPLES_FOR_QUANTITATIVE_METRICS = 5

# ================================================================================

def main():
    """
    Main function to load a pretrained model and run a full suite of
    Explainable AI (XAI) analyses on the test dataset.
    """
    print("="*60)
    print("--- Starting Standalone XAI Evaluation Script ---")
    print("="*60)

    # 1. Initialize Configuration
    # We use the Config class from PinPoint.py to ensure all settings are consistent.
    config = Config()
    # Set TESTING to False to use the full test set, otherwise it will be truncated.
    config.TESTING = False
    
    print(f"Model Checkpoint Path: {MODEL_CHECKPOINT_PATH}")
    print(f"XAI Output Directory:  {XAI_OUTPUT_DIR}")
    print(f"Dataset Path:          {config.DATA_DIRECTORY}")
    print(f"Device:                {config.DEVICE}")
    print("-" * 60)

    # 2. Perform Pre-run Checks
    if not os.path.exists(MODEL_CHECKPOINT_PATH):
        print(f"FATAL ERROR: Model checkpoint not found at '{MODEL_CHECKPOINT_PATH}'")
        print("Please update the 'MODEL_CHECKPOINT_PATH' variable in this script.")
        return

    if not os.path.exists(config.DATA_DIRECTORY):
        print(f"FATAL ERROR: Data directory not found at '{config.DATA_DIRECTORY}'")
        print("Please ensure your preprocessed data is correctly located.")
        return

    # 3. Load Model
    print("\n--- [1/5] Loading pretrained model ---")
    try:
        model = PinpointTransformer(config).to(config.DEVICE)
        model.load_state_dict(torch.load(MODEL_CHECKPOINT_PATH, map_location=config.DEVICE))
        model.eval() # IMPORTANT: Set the model to evaluation mode
        print("Model loaded and set to evaluation mode successfully.")
    except Exception as e:
        print(f"FATAL ERROR: Failed to load the model. Error: {e}")
        return

    # 4. Load Test Dataset
    print("\n--- [2/5] Loading test dataset ---")
    try:
        # We only need the 'test' split for this evaluation.
        test_dataset = LAVDFDataset(config, split='test')
        if len(test_dataset) == 0:
            print("FATAL ERROR: The test dataset is empty. Cannot proceed.")
            return
        print(f"Found {len(test_dataset)} samples in the test set.")
    except Exception as e:
        print(f"FATAL ERROR: Failed to load the dataset. Error: {e}")
        return

    # 5. Run Comprehensive Per-Sample XAI Analysis
    # This generates the detailed visual reports (IG, LRP, SHAP, etc.) for a few samples.
    print("\n--- [3/5] Running Comprehensive XAI Evaluation (Qualitative Analysis) ---")
    try:
        comprehensive_xai_evaluation(
            model=model,
            test_dataset=test_dataset,
            config=config,
            num_samples=NUM_SAMPLES_FOR_COMPREHENSIVE_XAI,
            output_dir=os.path.join(XAI_OUTPUT_DIR, "comprehensive_analysis")
        )
    except Exception as e:
        print(f"An error occurred during comprehensive XAI evaluation: {e}")
        import traceback
        traceback.print_exc()


    # 6. Run Concept-based Analysis (CAV & TCAV)
    # This tests the model's sensitivity to specific, defined concepts.
    print("\n--- [4/5] Running Concept Activation Vectors (CAV/TCAV) Analysis ---")
    try:
        demonstrate_cav_tcav(
            model=model,
            test_dataset=test_dataset,
            config=config,
            output_dir=os.path.join(XAI_OUTPUT_DIR, "cav_tcav_analysis")
        )
    except Exception as e:
        print(f"An error occurred during CAV/TCAV demonstration: {e}")
        import traceback
        traceback.print_exc()

    # 7. Run Quantitative XAI Metrics
    # This computes numerical scores (sparsity, consistency) across many samples.
    print("\n--- [5/5] Running Quantitative XAI Metrics (Sparsity, Consistency) ---")
    try:
        # Create a subset of the test data for efficient metric calculation
        num_samples_for_metrics = min(NUM_SAMPLES_FOR_QUANTITATIVE_METRICS, len(test_dataset))
        metric_indices = random.sample(range(len(test_dataset)), num_samples_for_metrics)
        metric_subset = torch.utils.data.Subset(test_dataset, metric_indices)
        metric_loader = DataLoader(
            metric_subset,
            batch_size=config.BATCH_SIZE,
            collate_fn=collate_fn,
            num_workers=2
        )

        quant_xai = QuantitativeXAIMetrics(model, config)
        
        # Calculate and report each quantitative metric
        quant_xai.calculate_feature_agreement(
            metric_loader,
            output_dir=os.path.join(XAI_OUTPUT_DIR, "quantitative_metrics")
        )
        quant_xai.calculate_focus_and_sparsity(metric_loader)
        quant_xai.calculate_inter_technique_consistency(metric_loader)
        
    except Exception as e:
        print(f"An error occurred during quantitative XAI metrics calculation: {e}")
        import traceback
        traceback.print_exc()


    print("\n" + "="*60)
    print("--- XAI Evaluation Script Finished ---")
    print(f"All results have been saved to the '{XAI_OUTPUT_DIR}' directory.")
    print("="*60)


if __name__ == '__main__':
    # Ensure reproducibility for sample selection
    random.seed(42)
    
    # Run the main evaluation function
    main()