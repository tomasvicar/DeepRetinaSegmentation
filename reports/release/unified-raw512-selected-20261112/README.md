# CPU deployment copy of the VAL-selected raw512 BEST50

This adds one explicit model, `unified-raw512-selected`, with rot4/.35/512/64
prediction defaults. Original521b…2cea and artifact292e…9b29 are different
identities; the frozen selection and any held-out evaluation bind the original.
The exporter sidecar records the full mapping and unchanged model-state digest.
No weights were trained, quantized or selected by this packaging operation.

CPU export on the existing Python3.12.14/Torch2.6 runtime completed with observed
exit0 in2.027s, peak RSS461,604KiB, core4, one thread, CUDA hidden, no swap.
The original SHA/stat remained unchanged. Actual product safe loaders compared
all tensor bytes/configuration. Output8,892,466B uses ordinary Git binary storage,
matching the repository's existing `.pt` strategy; no LFS/ZIP is introduced.

Focused tests on the same CPU runtime (`PYTHONPATH` set to the staged package,
OMP/MKL/OpenBLAS1, core4):

```
python -m pytest -q test_bundled_models.py test_bundled_inference_defaults.py test_inference_export.py
53 passed in 2.77s
```

The registry test checks all13 checksums, safe loading and actual selected coverage;
default tests preserve the old12 models/aliases and check all four new defaults.
Export tests exercise state/config preservation, CPU storage, strict load and
failure cases. These are focused packaging checks, not a new full-suite claim.

The [model card](../../../docs/UNIFIED_RAW512_SELECTED.md) retains the dense-vessel
VAL deficit and other adverse evidence. No TEST, all-VAL equivalence, universal
quality improvement or clinical validity is inferred from packaging checks.

Real TRAIN integration uses the preidentified DRIVE21 RGB and official FOV, with
the decoded photograph ID pinned to its canonical TRAIN record; it reads no GT.
Original research6d low-level rot4 → FOV/geometry restore → decoding was compared
with the product CPU artifact using API and CLI, **omitting all four options**.
All three actual children returned0 (23.09/20.26/20.86s), outer observed0;
probabilities were bitwise equal and all five masks had zero changed pixels.
The maximum combined owned-process RSS was bounded to3GiB, CUDA hidden/core4.
This verifies this one integration input, not quality or equality on all770 VAL.
[Numbers and exact input/source identities](TRAIN-parity.json) remain separate
from the frozen original quality report.

An initial research-call attempt exited1 **before any model forward** because
frozen6d's high-level `predict_unified` has no `tta` argument. The corrected separate
attempt uses its actual existing low-level rot4 function and literal surrounding
preparation/restore/decoding operations; the frozen source was never changed.
The first controller wheel build also failed with disk quota in temporary build
storage; the successful isolated build/install ran on Old's existing HOME runtime.
Failures and original detailed launch/output receipts remain in private operations
evidence; neither is represented as a successful evaluation.

The wheel build and `uv pip install --no-deps --target` both returned0 with the
existing runtime; no dependency/environment upgrade occurred. From HOME outside
both source and build checkouts, the installed package verified all13 model
checksums, unchanged aliases, the selected four defaults and safe CPU reload
**without map_location**. Its API then predicted the same TRAIN photo with omitted
options: probability max absolute difference0 and five unchanged masks versus the
frozen original. Actual outer/child exit0; CUDA stayed uninitialized. The build
artifact (69,830,696B) and observed execution are pinned in
[installed-wheel.json](installed-wheel.json) and its separate outer receipt.
The wheel itself remains outside Git; the new8.89MB model is a normal Git binary.
This is an integration check, not an assertion of new-device latency or TEST quality.
