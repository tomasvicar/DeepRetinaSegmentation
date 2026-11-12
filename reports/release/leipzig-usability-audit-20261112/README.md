# Leipzig prediction usability audit

Base: `2350bbf` (`for_Leipzig`). Documentation-only corrections:

- The first original-model commands now supply the actual source and matching FOV,
  show CPU operation, and use a copyable composed `uv run` command.
- The historical 45-degree/automatic-FOV fallback is disclosed; it is not camera
  calibration. The stricter composed command requires known calibration and FOV.
- The original guide identifies an existing DRIVE TRAIN21 photo/FOV pair in the
  shared archive, explains FOV versus task annotations, and gives Python-only
  extraction and path-based API examples.
- The unified preview card gives its array-FOV Python API and explains automatic
  saved photometry, avoiding double preprocessing.

Verification used the existing Python environment and this checkout's source:
`predict-original --help`, `predict --help`, and `models --help` returned 0 without
importing Torch. Python API examples passed AST syntax checks; argument contracts
were checked against source. ZIP member names were checked from archive metadata,
without extracting/decoding images. `git diff --check` passed.

No prediction, model tensor loading, GT decoding, training, TEST evaluation or new
package/wheel validation was performed. Previously completed installed-wheel
validation remains separate evidence. No CLI code, weights, aliases, inference
semantics or defaults changed. Actual predictions remain covered by the linked
model-specific evidence, not by these documentation checks.
