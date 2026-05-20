# Technical dump — [PAPER TITLE / PROJECT NAME]

**Project:** [Your paper title or working title]
**Date compiled:** [Date]
**Code location:** [Path or repo link]
**Results location:** [Path to results files / logs / notebooks]

---

## Instructions

Fill every section of this file by reading the code and results directly. Do not
fill from memory. The goal is that the methodology and experiments sections of the
paper can be written entirely from this file — no going back to the code mid-draft.

Mark any field as `[NOT APPLICABLE]` if it genuinely does not apply to your work.
Mark any field as `[TO VERIFY]` if you are unsure of the value and need to check.
Never leave a field blank without one of these markers.

After completing this file, cross-check it against the experiments sections of the
five venue papers in `VENUE_NOTES.md`. Add any category of reported information they
include that this file does not yet capture.

---

## 1. Problem and task definition

**Task type:**
[e.g., binary classification / multi-class classification / regression / detection /
segmentation / generation / retrieval — be specific]

**Input modalities:**
[What does the model take as input? e.g., RGB video frames + audio MFCCs, image + text]

**Output:**
[What does the model produce? e.g., binary real/fake label + confidence score]

**Evaluation setting:**
[e.g., supervised, zero-shot, few-shot, cross-dataset, in-distribution only]

---

## 2. Datasets

### Dataset 1

**Name:**
**Citation:** [Paper that introduced this dataset — must be cited]
**Access:** [URL or official source]
**Total samples:** [Total number of samples/clips/images]
**Train split:** [Count or percentage]
**Validation split:** [Count or percentage]
**Test split:** [Count or percentage]
**Class distribution:**
  - Class A: [count and percentage]
  - Class B: [count and percentage]
  - [Add more if multi-class]
**Imbalance handling:** [How did you handle class imbalance, if at all?]
**Preprocessing applied:**
  - [Step 1: e.g., resize to 128×128, normalize to [0,1]]
  - [Step 2: e.g., extract 8 frames per clip at equal intervals]
  - [Step 3: e.g., resample audio to 16kHz, pad/trim to 10s]
**Augmentation applied (training only):**
  - [e.g., random horizontal flip p=0.5, temporal jitter ±2 frames]
**Notes:** [Any unusual aspects of this dataset worth flagging]

---

### Dataset 2 (if applicable)

**Name:**
**Citation:**
**Access:**
**Total samples:**
**Train / Val / Test split:**
**Class distribution:**
**Preprocessing applied:**
**Augmentation applied:**
**Notes:**

---

## 3. Architecture

### 3.1 High-level description

**Model name (what you call it in the paper):**
**One-sentence description:**
[What does this model do, in plain language — this becomes your first methodology sentence]

**Novel components (list here first):**
[Name every component you designed or modified that is not taken directly from prior work.
These are your architectural contributions.]
- [Component 1 name]: [one sentence on its function]
- [Component 2 name]: [one sentence on its function]

---

### 3.2 Input processing

**Video / image input:**
- Resolution: [H × W × C]
- Frames per sample: [ ]
- Normalization: [mean=[], std=[] OR scale to [0,1] OR other]
- Any special preprocessing: [ ]

**Audio input:**
- Sample rate: [ ] Hz
- Feature type: [raw waveform / MFCC / mel-spectrogram / other]
- Feature dimensions: [ ]
- Window size / hop length: [ ]
- Duration per sample: [ ] seconds

**Other modalities (if applicable):**
- [Modality name]: [describe input format]

---

### 3.3 Feature extractors / backbones

**Visual backbone:**
- Architecture name: [e.g., ResNet-18, DenseNet-169, EfficientNet-B4]
- Pretrained on: [e.g., ImageNet-1K]
- Frozen layers: [e.g., all layers except last 2 blocks / fully fine-tuned / frozen]
- Output feature dimension: [ ]
- Citation for this architecture: [paper reference]

**Audio backbone / encoder:**
- Architecture name: [e.g., TCN + GRU, CNN-BiLSTM, Wav2Vec2]
- Pretrained on: [or "trained from scratch"]
- Frozen / fine-tuned: [ ]
- Output feature dimension: [ ]
- Citation: [ ]

**Other backbone (if applicable):**
- Architecture name:
- Pretrained on:
- Output feature dimension:
- Citation:

---

### 3.4 Fusion / novel components

Describe every novel component in detail. For each:

**Component name:**
**Purpose:** [Why is this component needed — what problem does it solve?]
**Inputs:** [What tensors / features go in, with dimensions]
**Outputs:** [What tensors / features come out, with dimensions]
**Mechanism:** [Step-by-step description of what this component computes]
**Key equations (if any):**
```
[Write equations here in plain text or LaTeX notation]
```
**Hyperparameters specific to this component:**
- [param name]: [value]
- [param name]: [value]
**Citation (if based on prior work):** [or "original contribution"]

---

### 3.5 Prediction heads

**Primary head:**
- Type: [e.g., FC layer + sigmoid, FC layer + softmax]
- Input dimension: [ ]
- Output: [e.g., scalar probability, class logits]
- Activation: [ ]

**Auxiliary head (if any):**
- Type:
- Purpose: [e.g., offset prediction for regularization]
- Input dimension:
- Output:

---

### 3.6 Full layer-by-layer summary (for key modules)

[For each major module, list layers in order with their dimensions. This is what
goes in the implementation details section or supplementary material.]

**Module: [name]**
```
Layer 1: [type, input dim → output dim, activation]
Layer 2: [type, input dim → output dim, activation]
...
```

---

## 4. Training procedure

**Hardware:**
- GPU model: [e.g., NVIDIA Tesla P100, RTX 3090]
- GPU VRAM: [ ] GB
- Number of GPUs: [ ]
- Total training time (approximate): [ ] hours

**Software:**
- Framework: [PyTorch / TensorFlow / JAX] version: [ ]
- CUDA version: [ ]
- Python version: [ ]
- Key library versions: [e.g., torchvision 0.15, librosa 0.10]

**Optimizer:**
- Name: [AdamW / Adam / SGD / etc.]
- Learning rate: [ ]
- Weight decay: [ ]
- Beta1, Beta2 (if Adam/AdamW): [ ]
- Momentum (if SGD): [ ]

**Learning rate schedule:**
- Type: [cosine annealing / step decay / warmup + decay / constant]
- Warmup steps / epochs: [ ]
- Minimum LR: [ ]
- Decay factor (if step decay): [ ]

**Training parameters:**
- Batch size: [ ]
- Total epochs: [ ]
- Actual epochs trained (if early stopping triggered): [ ]
- Random seed: [ ]

**Loss function(s):**

Loss 1:
- Name: [e.g., Binary Cross-Entropy with Logits]
- Purpose: [what it supervises]
- Weight in total loss: [ ]
- Any special parameters: [e.g., pos_weight for class imbalance]

Loss 2 (if applicable):
- Name:
- Purpose:
- Weight:
- Parameters:

Loss 3 (if applicable):
- Name:
- Purpose:
- Weight:
- Parameters:

**Total loss formula:**
```
L_total = [write out the weighted combination here]
```

**Regularization:**
- Dropout rates: [list per location, e.g., "0.4 after BiLSTM, 0.3 after Dense"]
- Weight decay: [if separate from optimizer setting]
- Batch normalization: [where applied]
- Label smoothing: [value, or N/A]

**Model selection criterion:**
[e.g., best validation accuracy, best validation F1, lowest validation loss]

**Early stopping:**
- Metric monitored: [ ]
- Patience: [ ] epochs
- Was it triggered?: [yes / no, and at which epoch if yes]

**Curriculum / phased training (if applicable):**
[Describe each phase: what data was used, which losses were active, how many epochs]
- Phase 1: [ ]
- Phase 2: [ ]
- Phase 3: [ ]

---

## 5. Evaluation metrics

For each metric used, fill one entry:

**Metric 1:**
- Name: [ ]
- Formula or definition: [ ]
- Why this metric was chosen: [ ]
- Higher is better: [yes / no]

**Metric 2:**
- Name:
- Formula or definition:
- Why chosen:
- Higher is better:

**Metric 3:**
- Name:
- Formula:
- Why chosen:
- Higher is better:

[Add more as needed]

---

## 6. Results

### 6.1 Main results table

[Fill this table with your results and every baseline you compare against.
Add or remove rows as needed.]

| Method | Accuracy | Precision | Recall | F1 | AUC | Notes |
|--------|----------|-----------|--------|----|-----|-------|
| [Baseline 1] | | | | | | [citation] |
| [Baseline 2] | | | | | | [citation] |
| [Baseline 3] | | | | | | [citation] |
| **[Your method]** | | | | | | |

**Takeaway sentence for this table:**
[One sentence: what does this table show, and what does it mean?]

---

### 6.2 Ablation study results

[One row per variant. Each variant should differ from the full model by exactly one
component. Fill in the component removed and all metrics.]

| Variant | Component removed / changed | Accuracy | F1 | Notes |
|---------|-----------------------------|----------|----|-------|
| Full model (ours) | — | | | |
| Ablation 1 | [what was removed] | | | |
| Ablation 2 | [what was removed] | | | |
| Ablation 3 | [what was removed] | | | |

**Takeaway sentence:**
[Which component's removal caused the largest drop, and what does that tell you?]

---

### 6.3 Per-class or per-condition results (if applicable)

[Fill if your paper reports results broken down by class, dataset, or condition]

| Class / Condition | Metric 1 | Metric 2 | Notes |
|-------------------|----------|----------|-------|
| | | | |
| | | | |

---

### 6.4 Statistical validation (if applicable)

| Test used | Comparing | p-value | Conclusion |
|-----------|-----------|---------|------------|
| | | | |

**Confidence intervals (if computed):**
[Method] 95% CI: [ ]

---

### 6.5 Cross-dataset / generalization results (if applicable)

| Train dataset | Test dataset | Metric | Value | Notes |
|---------------|-------------|--------|-------|-------|
| | | | | |

---

## 7. XAI / interpretability results (if applicable)

[Fill if your paper includes explainability analysis]

**XAI methods applied:**
- [Method 1]: [what it shows in your paper]
- [Method 2]: [what it shows in your paper]

**Quantitative XAI metrics (if any):**
| Metric | Value | What it means |
|--------|-------|---------------|
| | | |

**Key qualitative findings:**
[What do your XAI visualizations show? What patterns appear in real vs. fake samples?
What concepts does the model demonstrably rely on?]

---

## 8. Figures and diagrams needed

For every figure the paper requires, fill one entry. These become DIAGRAM PLACEHOLDERS
in the draft. Be specific — someone other than you should be able to produce the figure
from this description alone.

**Figure 1:**
- Type: [architecture diagram / workflow / results chart / heatmap / confusion matrix / attention map / comparison figure]
- Proposed title: [ ]
- What it shows: [Describe all elements: components, arrows, labels, axes, color coding]
- What the reader should conclude from it: [ ]
- Data source: [which results / which model output / which code file produces this]
- Priority: [essential / recommended / optional]

---

**Figure 2:**
- Type:
- Proposed title:
- What it shows:
- What the reader should conclude:
- Data source:
- Priority:

---

**Figure 3:**
- Type:
- Proposed title:
- What it shows:
- What the reader should conclude:
- Data source:
- Priority:

---

[Add more figure entries as needed]

---

## 9. Venue-specific checks

After filling all sections above, check the experiments sections of the five papers
in `VENUE_NOTES.md`. For each category they report that this file does not yet cover,
add a new section below.

**Additional category 1 (from venue paper analysis):**
[Category name and content]

**Additional category 2:**
[Category name and content]

---

## 10. Claim verification checklist

Before writing the draft, verify that every claim you intend to make in the paper
has supporting data in this file. Check each:

- [ ] Main accuracy / performance claim — supported by section 6.1
- [ ] Ablation claim — each component's contribution supported by section 6.2
- [ ] Any "state-of-the-art" claim — baseline comparison in section 6.1 is complete
- [ ] Any "novel" architecture claim — component described in section 3.4
- [ ] Any statistical significance claim — p-values in section 6.4
- [ ] Any qualitative / interpretability claim — XAI findings in section 7
- [ ] Any deployment / efficiency claim — hardware and training time in section 4
