# Cycle 3 — Demand-Planner Findings (reconstructed note)

> ⚠️ The original `testinput3.md` was lost (accidentally deleted during the run, then
> not recoverable from the workflow transcript — the planner agent returned its findings
> structurally rather than via a file write). This is a reconstruction note, not the
> verbatim original.

The cycle-3 demand-planner pass surfaced **no new actionable P0/P1/P2 planner findings**
(no `F3.x` items appear in the ledger). All cycle-3 fixes were usability-driven — see
[usability3.md](usability3.md) and the **## Cycle 3** block in [LEDGER.md](LEDGER.md):

- **U3.1** (P1) → FIXED — CA chart-panel toggle pills + tab clear-× themed (legible in Dark).
- **U3.3** (P2) → FIXED — CA toggle buttons expose `aria-pressed` + non-color active cue.
- **U3.2** (P2) → FIXED — Customer Demand Map footer total uses compact K/M/B formatting.
- **U3.4** (P3) → FIXED — removed fabricated backend `delta: 0.0`; UI renders "— no prior period".
- **U3.5 / U2.7** (P3) → FIXED — Item Analysis breadcrumb renders "Item &lt;id&gt; — &lt;desc&gt;".

The applied fixes + red→green evidence are intact in
[cycle3/fixes-applied.md](cycle3/fixes-applied.md); the code changes are in the working tree.
Lost beyond recovery: `cycle3/capture-digest.md` and `cycle3/screens/` (binary, gitignored anyway).
