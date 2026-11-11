# Curated fd44 final-eligibility proposal validation

Implementation base: `82939411e8afa6936effce2f9cb084f4eb0a31a5`.
See [the contract and boundary notes](../../docs/CURATED_FD44_SELECTION_PROPOSAL.md).

308 distinct focused tests passed across bounded CPU runs: original curated
checker/adapter and negative cases (47), synthetic curated context (14), existing
TTA/photometry/typed eligibility/V3/held-out regressions (137), and remaining
selection/producer/pool-adapter regressions (110). These counts include existing
tiny synthetic CPU forwards, never actual models or actual dataset/GT reads.

Recorded runs:

- Original checker/adapter: 42 passed in 11.85s.
- Context and old-family regressions: 151 passed in 37.54s.
- After formatting, 179 core tests passed in 43.15s; 14 context tests had setup
  errors because lint removed a pytest fixture import. The import was restored
  using the repository's explicit fixture-alias pattern.
- Corrected context plus complementary selection/producer/pool tests: 124 passed
  in 23.71s (all 14 affected context tests included).
- Strengthened partial-BEST check plus four explicit old-family rejection cases:
  5 passed in 1.88s. Partial BEST removes an actual annotation visit and cannot
  borrow terminal LAST coverage.
- Original-chain mutation after the checker: 1 passed in 2.00s.

An initial collection used the native environment's editable package rather than
this checkout; `PYTHONPATH=src` corrected that before any tests ran. An earlier
legacy policy assertion was updated to require the distinct curated index, while
preserving rejection of untyped curated TTA. No operational run was retried.

All pytest commands used CUDA hidden, one pinned CPU core, nice17/idle IO and one thread
per math backend, with each pytest invocation capped at60 seconds. Changed Python
files pass Ruff and `git diff --check`.

No actual model/checkpoint load, canonical dataset/GT read, GPU query/context,
actual VAL/TEST evaluation, final freeze, default promotion or operational source
change was performed. The proposal still requires independent review and fresh
source-bound evidence before actual curated final eligibility can be claimed.

Evaluation-source SHA256 at original proposal commit `13c3ae9`, before the
action-scoped checker-message clarification follow-up:
`3665df1838168e46ec6fb21dbf3af2d6318be4a11082ce21b42b45cb0b5ec613`.
