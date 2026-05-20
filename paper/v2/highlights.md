# Highlights

- PinPoint learns audio--visual deepfake evidence with gated cross-attention and an explicit synchronization loss.
- The synchronization loss shapes attention maps toward diagonal, smooth, and concentrated structures for aligned real samples.
- A three-stage curriculum reduces shortcut learning by first masking video, then emphasizing synchronization, then training full fusion.
- PinPoint reaches 97.47% accuracy and 0.9680 macro F1 on a unified LAV-DF and FakeAVCeleb test split.
- Multimodal XAI analysis links predictions to mouth-region, acoustic-feature, attention, counterfactual, and lip-sync concept evidence.
