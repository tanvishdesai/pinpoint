# results/ — source of truth

Trusted record of PinPoint numbers for the paper draft. Cite only from here.

- **[`RESULTS.md`](RESULTS.md)** — annotated, human-readable. Read this first.
- **[`consolidated_results.json`](consolidated_results.json)** — machine-readable, same numbers.

Every entry has a **trust tag**: `trusted` (cite it), `invalid` (degenerate run,
do not cite — kept to document the failure), `not_run` (planned/pending).

Two test sets are in play and must not be conflated: the **merged** LAV-DF+FakeAVCeleb
set (n=26,097, original run + ablations) and the **LAV-DF-only** set (n=1,550,
elevation metrics). See the warning box at the top of `RESULTS.md`.
