import torch
import os
import random
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from tqdm import tqdm

# --- Import necessary classes and functions from your main script ---
# This assumes 'PinPoint.py' is in the same directory.
from PinPoint import (
    Config,
    PinpointTransformer,
    LAVDFDataset,
    collate_fn,
    ConceptActivationVectors,
    ProbeDataset,
    T, # Import the torchvision transforms alias
)

# ================================================================================
# SCRIPT CONFIGURATION (HARDCODED PARAMETERS)
# ================================================================================

# --- REQUIRED: Point this to your trained model ---
MODEL_CHECKPOINT_PATH = "/kaggle/input/all-pp-xai-models/PP XAI no sync.pth"

# --- OPTIONAL: Modify if needed ---
# Directory where the corrected TCAV analysis results will be saved.
OUTPUT_DIR = "corrected_cav_tcav_analysis"


# ================================================================================
# MONKEY-PATCHING: CORRECTED TCAV SCORE CALCULATION
# ================================================================================

def fixed_compute_tcav_score(self, test_data, layer_name, target_class=1):
    """
    CORRECTED version of compute_tcav_score.
    This version computes gradients with respect to LOGITS instead of the
    post-sigmoid probabilities to avoid the gradient saturation problem.
    """
    if layer_name not in self.cavs:
        raise ValueError(f"CAV not trained for layer {layer_name}")

    cav = self.cavs[layer_name]['vector']
    directional_derivatives = []

    # Use a local, non-detaching hook for gradient calculation
    activations = []
    def hook_fn_for_tcav(module, input, output):
        activation_tensor = output[0] if isinstance(output, tuple) else output
        activations.append(activation_tensor)

    target_module = dict(self.model.named_modules()).get(layer_name)
    if target_module is None:
        raise ValueError(f"Layer {layer_name} not found in model.")

    hook = target_module.register_forward_hook(hook_fn_for_tcav)

    original_mode = self.model.training
    self.model.train()  # Required for GRU backward pass

    try:
        for batch in test_data:
            if batch is None: continue

            video = batch['video'].to(self.device)
            audio = batch['audio'].to(self.device)
            activations.clear()

            logits, _, _ = self.model(video, audio)

            if activations:
                layer_output = activations[0]
                
                # =================== START OF THE FIX ===================
                # We use the raw logits as the target score. The gradient of the logit
                # is non-zero even when the model is highly confident, which prevents
                # the saturation that caused the original TCAV scores to be zero.
                if target_class == 1:
                    target_score = logits
                else:
                    target_score = -logits # Pushes away from the positive class

                self.model.zero_grad()
                # Calculate gradients of the LOGIT w.r.t layer activations
                gradients = torch.autograd.grad(
                    outputs=target_score,
                    inputs=layer_output,
                    grad_outputs=torch.ones_like(target_score),
                    retain_graph=False
                )[0]
                # ==================== END OF THE FIX ====================

                # Pool gradients to match CAV vector dimension
                if gradients.ndim > 2:
                    grad_pooled = gradients.mean(dim=1)
                else:
                    grad_pooled = gradients

                grad_flat = grad_pooled.flatten(1).detach().cpu().numpy()
                for i in range(grad_flat.shape[0]):
                    dd = np.dot(grad_flat[i], cav)
                    directional_derivatives.append(dd > 0)
    finally:
        # Always clean up hook and model state
        hook.remove()
        self.model.train(original_mode)

    tcav_score = np.mean(directional_derivatives) if directional_derivatives else 0
    return tcav_score

# --- The actual Monkey Patch ---
# We are dynamically replacing the buggy method on the imported class
# with our new, corrected function. Any instance of ConceptActivationVectors
# created after this line will use our fixed logic.
ConceptActivationVectors.compute_tcav_score = fixed_compute_tcav_score
print("✅ Monkey-patch applied: `ConceptActivationVectors.compute_tcav_score` has been replaced with a corrected version.")


# ================================================================================
# HELPER FUNCTIONS (Copied from PinPoint.py for self-containment)
# ================================================================================

def _create_tcav_summary(tcav_scores, concept_name, output_dir):
    """Create a summary visualization for TCAV scores."""
    if not tcav_scores:
        return
    
    fig, ax = plt.subplots(figsize=(10, 7))
    
    layers = list(tcav_scores.keys())
    scores = list(tcav_scores.values())
    
    bars = ax.bar(range(len(layers)), scores, color='mediumseagreen', alpha=0.8)
    ax.set_xticks(range(len(layers)))
    # Use full layer names for clarity
    ax.set_xticklabels(layers, rotation=30, ha='right')
    ax.set_ylabel('TCAV Score')
    ax.set_title(f'Corrected TCAV Score\nConcept: "{concept_name}"')
    ax.set_ylim(0, 1)
    ax.axhline(y=0.5, color='r', linestyle='--', alpha=0.7, label='Random Baseline')

    for bar, score in zip(bars, scores):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height, f'{score:.3f}', ha='center', va='bottom')
    
    ax.legend()
    plt.tight_layout()
    
    tcav_path = os.path.join(output_dir, f"corrected_tcav_summary_{concept_name}.png")
    plt.savefig(tcav_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Corrected TCAV summary saved to: {tcav_path}")

# ================================================================================
# MAIN EXECUTION
# ================================================================================

def main():
    """
    Main function to load the model and re-run only the TCAV analysis.
    """
    print("="*60)
    print("--- Starting Corrected TCAV Analysis Script ---")
    print("="*60)

    config = Config()
    config.TESTING = False

    # --- Pre-run Checks ---
    if not os.path.exists(MODEL_CHECKPOINT_PATH):
        print(f"FATAL ERROR: Model checkpoint not found at '{MODEL_CHECKPOINT_PATH}'")
        return
    if not os.path.exists(config.DATA_DIRECTORY):
        print(f"FATAL ERROR: Data directory not found at '{config.DATA_DIRECTORY}'")
        return
    
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    print(f"Results will be saved to: {OUTPUT_DIR}")

    # --- Load Model ---
    print("\n--- [1/3] Loading pretrained model ---")
    model = PinpointTransformer(config).to(config.DEVICE)
    model.load_state_dict(torch.load(MODEL_CHECKPOINT_PATH, map_location=config.DEVICE))
    model.eval()
    print("Model loaded successfully.")

    # --- Load Test Dataset ---
    print("\n--- [2/3] Loading test dataset ---")
    test_dataset = LAVDFDataset(config, split='test')
    if len(test_dataset) == 0:
        print("FATAL ERROR: The test dataset is empty.")
        return
    print(f"Found {len(test_dataset)} samples in the test set.")

    # --- [3/3] Re-run CAV/TCAV Demonstration ---
    print("\n--- [3/3] Running Concept Analysis with Corrected TCAV Logic ---")

    # This instance will now use our `fixed_compute_tcav_score` method
    cav_analyzer = ConceptActivationVectors(model, config)
    layer_names = ['gated_attention_layers.2.ffn.0', 'classification_head']

    # --- Define concept artifact-generating functions ---
    def apply_visual_blurring(sample):
        video_tensor = sample['video']
        blur_transform = T.GaussianBlur(kernel_size=7, sigma=(3.0, 5.0))
        sample['video'] = torch.stack([blur_transform(frame) for frame in video_tensor])
        return sample

    def apply_lip_sync_error(sample):
        audio_tensor = sample['audio']
        offset = random.randint(5, 15)
        sample['audio'] = torch.roll(audio_tensor, shifts=offset, dims=0)
        return sample

    concepts = {
        'visual_blurring': apply_visual_blurring,
        'lip_sync_error': apply_lip_sync_error,
    }

    real_indices = [i for i, s in enumerate(test_dataset.samples) if not s['is_fake']]
    if not real_indices:
        print("FATAL ERROR: No 'real' samples found for baseline.")
        return

    random_indices = random.sample(real_indices, min(50, len(real_indices)))
    random_subset = ProbeDataset(test_dataset, random_indices)
    random_loader = DataLoader(random_subset, batch_size=config.BATCH_SIZE, collate_fn=collate_fn)
    
    all_tcav_scores = {}

    for concept_name, transform_fn in concepts.items():
        print(f"\nAnalyzing concept: '{concept_name}'...")
        concept_indices = random.sample(real_indices, min(50, len(real_indices)))
        probe_dataset = ProbeDataset(test_dataset, concept_indices, transform_fn=transform_fn)
        concept_loader = DataLoader(probe_dataset, batch_size=config.BATCH_SIZE, collate_fn=collate_fn)

        concept_activations, random_activations = cav_analyzer.extract_concept_activations(
            concept_loader, random_loader, layer_names
        )
        
        current_concept_scores = {}
        for layer_name in layer_names:
            cav_analyzer.train_cav(concept_activations, random_activations, layer_name)

            test_indices = random.sample(real_indices, min(20, len(real_indices)))
            test_dataset_concept = ProbeDataset(test_dataset, test_indices, transform_fn=transform_fn)
            test_loader = DataLoader(test_dataset_concept, batch_size=config.BATCH_SIZE, collate_fn=collate_fn)

            if len(test_loader) > 0:
                # This now calls our corrected function
                tcav_score = cav_analyzer.compute_tcav_score(test_loader, layer_name, target_class=1)
                print(f"  -> CORRECTED TCAV Score for '{layer_name}': {tcav_score:.4f}")
                current_concept_scores[layer_name] = tcav_score
        
        all_tcav_scores[concept_name] = current_concept_scores
        
        # We can still visualize the CAV accuracy and vectors
        cav_save_path = os.path.join(OUTPUT_DIR, f"cav_analysis_{concept_name}.png")
        cav_analyzer.visualize_cav_analysis(layer_names, concept_name, cav_save_path)

    # Generate final summary plots with the new, non-zero scores
    for concept_name, scores in all_tcav_scores.items():
        if scores:
            _create_tcav_summary(scores, concept_name, OUTPUT_DIR)
            
    print("\n" + "="*60)
    print("--- Corrected TCAV Analysis Finished ---")
    print(f"All new results have been saved to the '{OUTPUT_DIR}' directory.")
    print("="*60)

if __name__ == '__main__':
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)
    main()