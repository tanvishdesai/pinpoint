# PinPoint paper v2

This folder contains the rewritten Ain Shams Engineering Journal draft.

## Files

- `main.tex`: anonymized Elsevier-style manuscript for double-anonymized review.
- `title_page.tex`: separate title page with author details, declarations, CRediT roles, and generative-AI disclosure draft.
- `highlights.md`: Elsevier-style highlights.
- `VENUE_NOTES.md`: venue requirements and style notes from the paper-writing skill.
- `LITERATURE_LIST.md`: curated literature list.
- `literature_review.csv`: structured literature review table.
- `TECHNICAL_DUMP.md`: extracted technical details from code/results.
- `images/`: copied figures from the original draft.

## Compile

From this folder:

```powershell
pdflatex main.tex
pdflatex main.tex
pdflatex title_page.tex
```

The manuscript intentionally keeps author information out of `main.tex` because the target journal uses double-anonymized review.
