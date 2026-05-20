# Research Paper Writing Skill

**Version:** 1.0
**Built for:** Producing polished, publication-ready research paper drafts from raw
idea documents and code. Designed to avoid every common failure mode in academic
writing — particularly the specific issues that appear in AI-assisted drafts.

---

## Files in this skill

| File | Purpose |
|------|---------|
| `SKILL.md` | **Start here.** The complete pipeline description — all phases, all rules, the writing order, revision passes, and quick reference. |
| `WRITING_RULES.md` | The complete writing rulebook — structural, section-level, paragraph-level, sentence-level, and AI artifact avoidance rules. Final checklist lives here. |
| `VENUE_NOTES_TEMPLATE.md` | Template to fill when analyzing five papers from the target venue. Copy per project. |
| `LITERATURE_LIST_TEMPLATE.md` | Template for identifying and categorizing 12–15 papers for literature review. Copy per project. |
| `literature_review_template.csv` | Structured CSV for detailed per-paper analysis. Copy per project. |
| `TECHNICAL_DUMP_TEMPLATE.md` | Template for extracting all technical content from code and results before writing begins. Copy per project. |

---

## The five phases at a glance

```
Phase 0 → Write the story (2 paragraphs, plain English, no citations)
    ↓
Phase 1 → Venue selection + read 5 papers → VENUE_NOTES.md
    ↓
Phase 2 → Identify 12–15 papers → LITERATURE_LIST.md + literature_review.csv
    ↓
Phase 3 → Extract code and results → TECHNICAL_DUMP.md
    ↓
Phase 4 → Write the draft (method first, introduction last)
    ↓
Phase 5 → Five dedicated revision passes (one job per pass)
```

---

## The non-negotiable rules (read these first)

1. **Write the introduction last.** Always.
2. **One paragraph, one idea.** No exceptions.
3. **Every claim: ask "so what?"** The answer is the next sentence.
4. **Every result needs an interpretation sentence.** Not "Table 2 shows results."
5. **Read the paper out loud before submitting.** Every time.
6. **Every citation must be personally verified.** No phantom references.

---

## How to use this skill per project

1. Copy `VENUE_NOTES_TEMPLATE.md` → rename to `VENUE_NOTES_[PROJECTNAME].md`
2. Copy `LITERATURE_LIST_TEMPLATE.md` → rename to `LITERATURE_LIST_[PROJECTNAME].md`
3. Copy `literature_review_template.csv` → rename to `literature_review_[PROJECTNAME].csv`
4. Copy `TECHNICAL_DUMP_TEMPLATE.md` → rename to `TECHNICAL_DUMP_[PROJECTNAME].md`
5. Follow `SKILL.md` phase by phase
6. Reference `WRITING_RULES.md` during Phase 4 and all Phase 5 revision passes

---

## What this skill is designed to prevent

These are the specific failure modes it was designed to eliminate, observed in
real paper drafts:

- **Related work that lists facts without synthesis** — addressed in Phase 2 + WRITING_RULES §2.1
- **Contributions that are vague or unverifiable** — addressed in WRITING_RULES §1.3
- **Introduction written before the paper is known** — addressed by writing order in Phase 4
- **AI-generation artifacts: phantom citations, cross-contaminated text, confident vagueness** — addressed in WRITING_RULES §5
- **Missing hyperparameters, unreproducible experiments** — addressed in Phase 3
- **Overselling or underselling the contribution** — addressed in WRITING_RULES §4.4
- **Broken paragraph logic, decoration-only transitions** — addressed in WRITING_RULES §3
- **Results tables with no interpretation** — addressed in WRITING_RULES §2.3
- **Diagram placeholders forgotten or left vague** — addressed by placeholder format in Phase 4
