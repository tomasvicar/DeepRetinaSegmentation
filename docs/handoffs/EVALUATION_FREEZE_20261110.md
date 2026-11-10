# Evaluation freeze — 2026-11-10, 18:03 UTC

This supplements the timestamped [continuation](../CONTINUATION_2026-11-10_1752.md).
The goal remains incomplete. No actual new-model TEST evaluation has run.

- Genuine clean Ada checkout: `/var/home/tom/code/DeepRetinaSegmentation_evaluation_8c56594`,
  detached `8c56594f36a92e569e901fcce0ed99ab2179629f`.
- Evaluation package/project hash: `3779e1b5e9577d6870cf5e9a3ee48fb2271941dbbca12891f7a7f002075c4c9c`.
  Actual imported paths/dependencies/default numerical flags are recorded in
  `reports/unified/evaluation-runtime-8c56594-prepared.json`. CUDA remained uninitialized.
- Entire genuine checkout passed687 tests; exact log, runner and before/after
  source/runtime observations in `reports/heldout/`. Four earlier failed tests
  and their environment/fixture causes are retained in that proof. Subsequent
  synthetic fixture isolation restores its own runtime flags.
- Use the existing frozen87 uv Python environment with explicit `PYTHONPATH`
  pointing at this new checkout's `src`; do not silently import frozen87 code.
  Do not modify either execution checkout or the still-running3157 training source.
- First wait for terminal60 QA and independent copies of best/last. Best may be
  earlier than60. No evaluation of a moving live checkpoint for final selection.
- Threshold plan `reports/unified/global-threshold-val-20261110/plan.json`
  predeclares six VAL thresholds and keeps fresh0.5 as reference. It still needs
  a concrete completed-run checkpoint/device/runtime addendum before execution.
- Produce fresh enriched VAL evidence with this frozen source, inspect every
  head/source and the stronger original references, then freeze selection and
  inference protocol. Only afterward use the explicit reviewed TEST dispatcher.
- AVweight2 E8 is now terminal-QA verified and independently copied; see
  `reports/unified/old-avweight2-e008-verified.json` and its comparison. Small AV
  gains came with a vessel regression; no final superiority claim.
- One Meta paired E8 allocation was reviewed for submission in the PRIVATE repo.
  This note does not assert actual submission: inspect its current PBS/ledger.
  No arrays, retries or concurrent project GPU allocations are authorized there.

Optional MATLAB FOV search/binning work continues in its own branch and must not
change this frozen evaluation source. Heavy Pascal training remains stopped.
