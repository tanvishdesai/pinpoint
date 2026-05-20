# Technical Dump

## Plain contribution framing

Deepfake videos are hard to trust because fake speech and fake face motion can each look plausible by themselves. The harder clue is whether the sound and face move together in time. PinPoint turns that clue into a training objective rather than hoping the model discovers it by accident.

PinPoint uses audio-to-video cross-attention, then trains the attention map itself to look diagonal and smooth for real aligned samples. The model is evaluated not only by accuracy but also by whether several explanation methods point to mouth-region, audio-feature, and lip-sync evidence.

## Model

- Name: PinPoint.
- Task: binary audio--visual deepfake detection.
- Inputs:
  - Video: 30 RGB frames, resized to 128 x 128, ImageNet normalization.
  - Audio: 13-dimensional MFCC sequence.
- Visual encoder:
  - ImageNet-pretrained ResNet-18.
  - In-place ReLU and residual operations patched for XAI compatibility.
  - Final convolutional features adaptively pooled and projected to 256 dimensions.
  - Early feature extractor layers frozen.
- Audio encoder:
  - Conv1D 13 -> 64, kernel 3, padding 1.
  - Conv1D 64 -> 128, kernel 3, padding 1.
  - LayerNorm(128).
  - GRU hidden size 256.
- Fusion:
  - 3 gated cross-attention blocks.
  - 8 attention heads.
  - Dropout 0.1.
  - Audio is query; video is key/value.
  - Learned sigmoid gate filters attended audio features.
  - Audio self-attention and feed-forward network after gating.
- Outputs:
  - Binary fake logit.
  - Auxiliary offset head with 11 classes, corresponding to -5 through +5 video-frame offsets.
  - Final audio-to-video attention map.

## Training

- Framework: PyTorch.
- Epochs: 15.
- Batch size: 4.
- Optimizer: AdamW.
- Learning rate: 1e-4.
- Weight decay: 1e-4.
- Scheduler: cosine annealing.
- Automatic mixed precision: enabled.
- Gradient clipping: max norm 1.0.
- Class imbalance:
  - BCEWithLogitsLoss uses pos_weight = num_real / num_fake from the training split.
- Loss weights:
  - Classification: 1.0.
  - Offset: 0.5.
  - Synchronization: 3.0.
- Synchronization loss components:
  - Direct MSE target weight: 1.0.
  - Diagonal dominance weight: 0.5.
  - Smoothness weight: 0.2.
  - Bandwidth: 2.
  - MFCC frames per video frame: 2.
- Curriculum:
  - Phase 1: epochs 1-2, mask 50-80% of video frames.
  - Phase 2: epochs 3-5, synchronization focus; offset loss down-weighted by 0.1.
  - Phase 3: epochs 6-15, full multimodal training.
- Data augmentation:
  - JPEG compression.
  - Random noise.
  - Gaussian blur.
  - Resize degradation.
  - Color jitter.
  - Random erasing.
  - Modality dropout probability 0.15.
- Missing detail before final submission:
  - Hardware model and VRAM are not recorded in the available logs. Add this if known.
  - Random seed is not recorded in the available code path. Add or report before final submission.

## Data

- Datasets: LAV-DF and FakeAVCeleb in a unified preprocessed metadata format.
- Splits:
  - Train: 78,703 total; 21,254 real; 57,449 fake.
  - Validation: 31,501 total; 8,271 real; 23,230 fake.
  - Test: 26,097 total; 6,906 real; 19,191 fake.

## Results

- Full model:
  - Accuracy: 97.47% (25,436 / 26,097).
  - Real precision / recall / F1: 0.93 / 0.98 / 0.9535.
  - Fake precision / recall / F1: 0.99 / 0.97 / 0.9825.
  - Macro F1: 0.9680.
  - Confusion matrix:
    - Real predicted real: 6,767.
    - Real predicted fake: 139.
    - Fake predicted real: 522.
    - Fake predicted fake: 18,669.
- Ablations:
  - No synchronization loss: 75.47% accuracy, fake F1 0.8067.
  - No gate: 58.35% accuracy, fake F1 0.6115.
  - No curriculum: 81.44% accuracy, fake F1 0.8639.
- XAI metrics:
  - Full model video Gini: 0.631.
  - Full model audio Gini: 0.928.
  - Full model consistency, Spearman: 0.514.
  - No curriculum consistency: 0.450.
  - No gate consistency: 0.372.
  - No synchronization loss consistency: 0.436.

## Figures used

- Figure 1: architecture and curriculum overview.
- Figure 2: gated cross-attention block.
- Figure 3: multimodal XAI suite example.
- Figure 4: LIME explanations across full model and ablations.
- Figure 5: real versus fake attention maps.
- Figure 6: attribution case study with original, Grad-CAM, and SHAP.
- Figure 7: counterfactual explanation.
- Figure 8: TCAV lip-sync concept scores.
