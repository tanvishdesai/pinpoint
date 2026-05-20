---
name: research-paper-writing
description: >
  Use this skill whenever the task is to write, draft, or improve a research paper.
  Triggers include: "write a paper about my project", "draft a research paper", "help me
  write a conference paper / journal paper", "turn my code and results into a paper",
  "write the related works section", "improve my draft", or any similar request that
  involves producing academic writing from code, results, or an idea document.
  This skill covers the complete pipeline: venue selection → literature review →
  technical extraction → full draft → revision. Do NOT use for blog posts, reports,
  or non-academic writing.
---

# Research paper writing — complete pipeline

## Overview

This skill takes you from a raw idea document plus code/results all the way to a
polished, submission-ready research paper draft. It is designed to avoid every common
failure mode: opaque writing, undersold contributions, oversold claims, incoherent
related work, and AI-generation artifacts that survive into submission.

**What you start with:**
- An idea document (can be rough notes, a problem statement, experiment logs)
- Code for the project and its outputs (results, plots, tables, metrics)

**What you produce by the end:**
- `VENUE_NOTES.md` — structure and style observations from five target-venue papers
- `LITERATURE_LIST.md` — 12–15 papers selected for literature review
- `literature_review.csv` — filled structured literature review table
- `TECHNICAL_DUMP.md` — all reportable technical content extracted from code/results
- The paper draft itself, section by section

**Reference files in this skill:**
- `VENUE_NOTES_TEMPLATE.md` — blank template for venue analysis
- `LITERATURE_LIST_TEMPLATE.md` — blank template for the paper list
- `literature_review_template.csv` — blank CSV for structured literature review
- `TECHNICAL_DUMP_TEMPLATE.md` — blank template for technical extraction
- `WRITING_RULES.md` — the complete writing rules referenced throughout this skill

---

## Phase 0 — orientation (do this before anything else)

Before opening any template or writing any sentence, spend time with the idea document
and the code. Your goal in this phase is to be able to answer five questions in plain
English, without technical jargon:

1. What problem exists in the world that motivates this work?
2. Why do current methods fail to solve it adequately?
3. What did you do differently?
4. What specific evidence shows it works?
5. Who benefits from this, and how?

Write out those five answers in a scratch paragraph — conversational, plain, no LaTeX,
no citations. This is not for the paper. It is for you. If any answer is fuzzy or
requires a paragraph of caveats to state, that is a signal: the contribution framing
needs sharper thought before writing begins.

**If you cannot state the core contribution in two sentences, stop here and resolve
that before proceeding.**

---

## Phase 1 — venue selection and style analysis

### 1.1 Choose the target venue

Decide whether this work targets a conference or journal, and name it specifically.
Do not leave this as "a good venue." Examples: CVPR, NeurIPS, IEEE TPAMI, Pattern
Recognition, Ain Shams Engineering Journal. The venue determines page limits, section
structure expectations, citation density, notation conventions, and formality level.

Record the choice at the top of `VENUE_NOTES.md`.

### 1.2 Find five strong accepted papers from that venue

Select five published papers from the exact venue that:
- Are thematically close to your work (same problem domain or same methods domain)
- Were clearly accepted on merit (look for highly cited papers, best paper nominees,
  or papers from strong research groups)
- Were published recently (within 3 years unless the venue moves slowly)

Avoid selecting papers that are weak in their related work or writing — you will use
these as quality benchmarks.

### 1.3 Read each paper and fill in `VENUE_NOTES.md`

For each of the five papers, record the following by reading the actual paper, not
its abstract alone:

**Structure observations:**
- How many sections does it have, and what are they named?
- Does the introduction end with a bullet-point list of contributions, or prose?
- How long is the related work section relative to the paper total?
- Is related work organized by method type, chronologically, or by research gap?
- Does the methodology section use numbered subsections? Pseudocode? Algorithm boxes?
- How are tables formatted — do they use booktabs style (top/mid/bottom rules)?
- Are figures numbered and captioned with full standalone descriptions?

**Writing style observations:**
- What is the average sentence length — short and direct, or long and clause-heavy?
- How are claims hedged — "we show", "results suggest", "we demonstrate"?
- How dense are the citations — inline grouped `[1,2,3]` or footnoted?
- Does the paper use first person ("we") or passive voice?
- How does the related work end? Does it explicitly state the gap?
- How does the conclusion read — summary-only or does it add future work?

**Contribution framing observations:**
- How are the core contributions stated? Numbered list, paragraph, or mixed?
- Are the contributions stated as specific claims with evidence pointers, or vague?
- Does the paper restate its contributions in the conclusion with matching language?

After completing all five papers, write a single "synthesis note" at the bottom of
`VENUE_NOTES.md` summarizing the pattern: what the typical structure looks like, the
typical tone, and two or three specific things to consciously mirror in your paper.

---

## Phase 2 — literature review

### 2.1 Identify 12–15 papers for the review

Using the idea document and Phase 0 answers as guidance, identify the papers that
belong in your literature review. The goal is not to cite everything tangentially
related. The goal is to map the landscape of prior work in a way that makes your
contribution's necessity visible.

Think in three categories:

**Direct predecessors** (4–5 papers): Papers that do the closest thing to what you do.
Your paper must acknowledge and differentiate from every one of these.

**Methodological foundations** (4–5 papers): Papers that introduce the methods you build
on (backbone architectures, loss functions, evaluation frameworks). These establish
credibility and show you understand the tools you are using.

**Gap-exposing papers** (3–5 papers): Papers whose limitations directly motivate your
work. When you describe what these papers cannot do, it explains why your work exists.

Record all 12–15 in `LITERATURE_LIST.md` with title, authors, venue/year, and one
sentence on why it belongs in the review.

### 2.2 Fill in `literature_review.csv`

For each paper in the list, fill one row of the CSV using the template columns. Read
the actual paper — do not fill from abstract alone.

The CSV has the following columns (see `literature_review_template.csv`):

| Column | What to fill |
|--------|-------------|
| `paper_id` | Short identifier, e.g. `smith2023deepfake` |
| `title` | Full title |
| `authors` | First author et al. |
| `venue` | Conference or journal name |
| `year` | Year of publication |
| `category` | `direct_predecessor`, `methodological_foundation`, or `gap_exposing` |
| `core_method` | One sentence: what technique or approach they use |
| `dataset_used` | Dataset(s) they evaluate on |
| `key_metric` | Their best reported number and metric name |
| `key_strength` | What they genuinely do well |
| `key_limitation` | The specific weakness most relevant to your work |
| `how_we_differ` | One sentence: what your work does that this paper does not |
| `cite_in_sections` | Comma-separated list: `intro`, `related_work`, `method`, `experiments` |

**Do not skip the `how_we_differ` column.** It is the single most important column.
Every paper you cite must have a clear answer there. If you cannot differentiate from
a paper, that paper is either not in your review or it has exposed a gap in your own
contribution framing that needs to be resolved before writing.

### 2.3 Structure the related work section from the CSV

Group all 12–15 papers by their natural subsection themes (not by your three
categories — those were for your thinking, not for the reader). Typical groupings:
one subsection per major direction of prior work, each ending with a gap statement.

The structure of each subsection follows this template:
1. One framing sentence introducing this direction
2. 3–5 paper summaries, each 2–3 sentences: method, strength, limitation
3. One closing sentence stating the gap that remains — the gap your paper fills

The last subsection or a standalone closing paragraph should synthesize across all
subsections and state explicitly what is missing from all prior work together.

**Do not write this section yet.** Just plan the subsection groupings now. The actual
writing happens in Phase 4.

---

## Phase 3 — technical extraction

### 3.1 Read the code and produce `TECHNICAL_DUMP.md`

Open the codebase and the results files. The goal of this phase is to extract every
piece of technical information that needs to appear somewhere in the paper, organized
and pre-formatted so that writing the methodology and experiments sections does not
require going back to the code.

Fill `TECHNICAL_DUMP.md` using the template. Required sections:

**Architecture and method:**
- Model name(s) and what they do at a high level
- Input format: dimensions, preprocessing steps, normalization values
- All backbone/pretrained model choices with their original citations
- Layer configurations (depths, widths, activation functions)
- Any novel components: name them, describe their function, write the math if applicable
- Output format: what the model produces, how it is converted to a prediction

**Training procedure:**
- Hardware used (GPU model, VRAM)
- Framework (PyTorch / TensorFlow / JAX, version)
- Optimizer name and all its hyperparameters (lr, weight decay, betas, etc.)
- Learning rate schedule (type, warmup steps, decay factor)
- Batch size
- Number of epochs / steps trained
- Loss function(s) with all component weights
- Regularization: dropout rates, weight decay, data augmentation used
- Any early stopping or model selection criteria
- Random seed(s) used

**Datasets:**
- Name of every dataset used
- Number of samples in train / validation / test splits
- Class distribution (balanced or imbalanced — note the ratio)
- Any preprocessing beyond normalization
- Source/citation for each dataset

**Results:**
- Primary metric(s) — for each: name, value, whether higher-is-better
- All comparison baselines: their method name, their reported number, the source
- Ablation study results: each variant name, what was changed, the resulting metric
- Any secondary metrics (per-class, per-dataset, per-condition)
- Any statistical validation (p-values, confidence intervals, significance tests)

**Figures and diagrams needed:**
- List every figure you believe the paper needs, with a one-sentence description of
  what it should show. These become diagram placeholders in the draft.

After filling this file, go back to the five venue papers from Phase 1. Check what
they report in their experiments sections. Add any category of information they report
that you have not yet captured.

---

## Phase 4 — writing the draft

Now you have everything. Begin writing. Follow the order below — this is not the
section order of the paper, it is the writing order, which is different.

### Writing order

1. **Method section** — write this first because you know it most precisely
2. **Experiments section** — write immediately after, while method is fresh
3. **Related work section** — write now using the CSV and Phase 2 structure plan
4. **Abstract** — write after method, experiments, related work are complete
5. **Introduction** — write last, using language already established in the above
6. **Conclusion** — write after introduction

### Why this order

The introduction must promise exactly what the paper delivers. If you write the
introduction first, you are promising things you have not yet formulated precisely.
If you write it last, you are simply narrating what you already wrote, which is far
easier and far more accurate. The abstract is a compression of the introduction, so
it too comes late.

### How to begin each section

Before writing any section, answer these questions in one sentence each:
- What does this section need to communicate to the reader?
- What does the reader need to know coming in, and what must they know leaving?
- What is the one thing they must not miss?

Write those three sentences as a comment at the top of the section draft. Delete them
when the section is done. They keep you anchored.

### Diagram placeholders

Whenever, while writing, you judge that a figure would communicate something better
than prose alone, do not skip it or leave a vague note. Insert a formatted placeholder
at that exact point in the draft:

```
[DIAGRAM PLACEHOLDER]
Type: architecture diagram / workflow / results chart / comparison table / attention map
Title: (proposed figure title)
Shows: (one paragraph describing exactly what this figure contains, what each element
        represents, what the reader should conclude from looking at it)
Priority: essential / recommended / optional
[END PLACEHOLDER]
```

This serves two purposes: it forces you to think clearly about what the figure must
communicate, and it gives whoever produces the figure an unambiguous brief.

---

## Phase 5 — revision

Never revise the whole paper in a single pass. Each pass has one job only.

### Pass 1 — contribution audit

Read only the abstract and introduction. Answer: is the core contribution stated in
one or two sentences, specifically and without hedging? Does the introduction end with
a clear list or statement of contributions? Are those contributions verifiable against
the experiments section? Any contribution not backed by a specific table row or figure
must be either backed or removed.

### Pass 2 — related work audit

Read only the related work section. For each cited paper, check: is it summarized
accurately in your own words? Does the summary explain why it matters to your work?
Does each subsection end with a gap statement? Does the section as a whole end with
a statement of what is missing from all prior work?

### Pass 3 — methodology audit

Read only the methodology section. Test: could a competent researcher in your field
reproduce your approach from this section alone, without seeing your code? Every
architecture choice, every hyperparameter, every loss weight, every dataset split must
be present. If `TECHNICAL_DUMP.md` has something that the methodology section does not,
add it.

### Pass 4 — logic flow audit

Read the full paper start to finish, slowly. At the end of every paragraph, ask: does
the next paragraph follow logically from this one? At the end of every section, ask:
does the next section follow logically from this one? A broken logic flow is usually a
missing transition sentence. Add it.

### Pass 5 — sentence-level audit

Read the full paper out loud. Every sentence that makes you pause, sounds clunky, or
loses you mid-clause gets rewritten. This pass catches: run-on sentences, passive voice
where active is clearer, hedged claims that need to be direct, and AI-generation
artifacts (see `WRITING_RULES.md` section 5).

---

## Quick reference

| Phase | Input | Output file | Template |
|-------|-------|-------------|----------|
| 0 | Idea document | Scratch paragraph (informal) | — |
| 1 | 5 venue papers | `VENUE_NOTES.md` | `VENUE_NOTES_TEMPLATE.md` |
| 2 | 12–15 papers | `LITERATURE_LIST.md` + `literature_review.csv` | both templates |
| 3 | Code + results | `TECHNICAL_DUMP.md` | `TECHNICAL_DUMP_TEMPLATE.md` |
| 4 | All above | Paper draft (section by section) | `WRITING_RULES.md` |
| 5 | Draft | Revised draft | `WRITING_RULES.md` |

---

## See also

- `WRITING_RULES.md` — the complete set of writing rules, sentence-level to structural
- `VENUE_NOTES_TEMPLATE.md` — blank template to copy per project
- `LITERATURE_LIST_TEMPLATE.md` — blank template for paper list
- `literature_review_template.csv` — blank CSV for literature review
- `TECHNICAL_DUMP_TEMPLATE.md` — blank template for technical extraction
