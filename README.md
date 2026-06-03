# PinPoint: High-Precision Deepfake Detection

PinPoint is a state-of-the-art multi-modal deepfake detection architecture that leverages Curriculum Learning, Modality Gating, and Synchronization features to precisely identify audio-visual manipulations.

## Core Concepts

- **Curriculum Learning (Curr)**: The model is trained on progressively harder examples, significantly improving its generalization against unseen deepfake generation methods.
- **Modality Gating (Gate)**: An intelligent gating mechanism that dynamically weights the importance of audio versus visual features depending on the scene context.
- **Audio-Visual Sync**: Explicitly extracts synchronization offsets as a feature to detect subtle lip-sync manipulations.

## Project Structure

- `PinPoint.py`: The core model architecture and training loops.
- `full-lavdf-testing.py`: Evaluation script for benchmarking the model on the full LAV-DF dataset.
- `xxai.py` / `correct-tcav.py`: eXplainable AI (XAI) scripts to interpret model decisions using techniques like TCAV.
- `Case Study.py`: Scripts for generating specific failure/success case studies.

## Ablation Studies

The repository includes extensive ablation study scripts to prove the efficacy of each component:
- `no curr-model.py` (Curriculum disabled)
- `no gate - model.py` (Gating disabled)
- `no loss.py` (Custom loss disabled)
The corresponding `*-results.txt` and `*-graphs/` directories contain the empirical proof of PinPoint's superiority.
