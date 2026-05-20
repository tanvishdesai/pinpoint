# Literature List

This list follows the research-paper-writing skill categories. The v2 manuscript cites a focused subset rather than copying every reference from the old draft.

## Direct predecessors

1. Chugh et al., "Not made for each other: Audio-visual dissonance-based deepfake detection and localization," ACM MM 2020.
   - Directly motivates the use of audio--visual mismatch rather than unimodal artifacts.

2. Haliassos et al., "Lips don't lie: A generalisable and robust approach to face forgery detection," CVPR 2021.
   - Establishes lip-region temporal dynamics as robust forensic evidence.

3. Raza and Malik, "Multimodaltrace: Deepfake detection using audiovisual representation learning," CVPR Workshops 2023.
   - A close multimodal representation-learning baseline.

4. Gao et al., "Temporal feature prediction in audio-visual deepfake detection," Electronics 2024.
   - Uses temporal prediction to learn audio--visual inconsistency.

5. Javed et al., "Audio-visual synchronization and lip movement analysis for real-time deepfake detection," International Journal of Computational Intelligence Systems 2025.
   - Strong recent synchronization-focused predecessor with LAV-DF and FakeAVCeleb results.

## Methodological foundations

6. He et al., "Deep residual learning for image recognition," CVPR 2016.
   - Backbone citation for ResNet-18.

7. Ribeiro et al., "Why should I trust you? Explaining the predictions of any classifier," KDD 2016.
   - LIME foundation.

8. Lundberg and Lee, "A unified approach to interpreting model predictions," NeurIPS 2017.
   - SHAP foundation.

9. Sundararajan et al., "Axiomatic attribution for deep networks," ICML 2017.
   - Integrated Gradients foundation.

10. Bach et al., "On pixel-wise explanations for non-linear classifier decisions by layer-wise relevance propagation," PLOS ONE 2015.
    - LRP foundation.

11. Kim et al., "Interpretability beyond feature attribution: Quantitative testing with concept activation vectors," ICML 2018.
    - TCAV foundation.

## Gap-exposing papers

12. Khalid et al., "FakeAVCeleb: A novel audio-video multimodal deepfake dataset," arXiv 2021.
    - Shows the need for paired audio/video manipulation benchmarks.

13. Cai et al., "Do you really mean that? Content driven audio-visual deepfake dataset and multimodal method for temporal forgery localization," DICTA 2022.
    - Motivates temporal localization and content-driven audiovisual forgery.

14. Baldassarre et al., "Quantitative metrics for evaluating explanations of video DeepFake detectors," BMVC 2022.
    - Exposes the lack of quantitative explanation evaluation.

15. Tsigos et al., "Towards quantitative evaluation of explainable AI methods for deepfake detection," MAD 2024.
    - Supports the need for perturbation-oriented XAI validation.

16. Nguyen-Le et al., "Passive deepfake detection across multi-modalities: A comprehensive survey," arXiv 2024.
    - Summarizes generalization, robustness, attribution, and interpretability gaps.

17. Croitoru et al., "MAVOS-DD: Multilingual audio-video open-set deepfake detection benchmark," arXiv 2025.
    - Exposes open-set and multilingual robustness gaps not yet solved by PinPoint.

18. Kaya and Alhajj, "An integrated explainable framework for multimodal deepfake detection across image, audio, and video data," Ain Shams Engineering Journal 2026.
    - Closest target-venue paper; shows venue fit for explainable multimodal deepfake detection.
