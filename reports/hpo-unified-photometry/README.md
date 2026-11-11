# HPO photometry port — isolated review candidate

Base: `7718e94d7e1bb4bc907ff9a71e15f2ad780cded3`. The reviewed product1fb operator,
strict cache reader, Dataset, train/resume and descriptor validation are ported
selectively. HPO safe immutable byte-stream loading is retained, with only saved
photometry validation inserted. HPO A/V-residual architecture, model bytes, joint
FP32 TTA math and strict `inference_tta` descriptors remain unchanged.

Fresh VAL can receive an externally pinned processed cache. Nonraw protocol and
metrics carry the exact saved operator/backend descriptor; header and metric
binding use canonical JSON types. Default historical raw identity/schema are
unchanged, and original fields are retained whenever present. Cache runtime
provenance is portable and does not include host paths. Full prepared-image
photometry precedes crops/augmentation/TTA/tiling, with image-derived FOV only.

No new completion family, curated admission, TEST path, actual model/source freeze,
model promotion or evaluation is created. `completion_profiles.py`,
`freeze_eligibility.py`, `v3_completion.py`, `heldout.py`, `inference_tta.py`, model,
A/V-residual implementation and legacy files are byte-unchanged. The existing
source72-only TTA producer guard is preserved. Prior evidence cannot be relabelled
with this branch's new evaluation identity.

## Verification

CPU6, one thread/inter-op where used, nice17/idleIO, CUDA hidden, external60s cap
per test command; generated fixtures use a private home basetemp.

- 106 focused operator/cache/HPO-TTA tests passed in15.48s.
- 92 producer/selection/TTA regressions passed in18.73s after adding canonical
 NONE-mode nonraw header binding; existing tiny CPU producer fixtures only.
- 33 final HPO integration checks passed in3.40s, including both existing benchmark
 callers mocked to ensure they never feed raw RGB to a nonraw saved model.

Ruff and diff whitespace checks pass. Tests include exact default raw crops/RNG,
tiny d028 raw weight/optimizer/RNG parity, cache/resume relocation, all-profile TTA
preprocessing once, preserved photo/support traces, descriptor/backend/FOV/pin and
corruption failures, boolean/integer header negatives, cache forwarding/runtime
identity and rejection of a new curated training identity from TTA admission.

Nine core HPO model/joint/TTA/decoder function ASTs remain exact. Photometry and
cache module bytes plus the legacy operator dependency match reviewed1fb; therefore
operator fingerprint remains196da8626c87d6b07bed6508252dd37328ee79846c4040c1403f9a090002708e.
The earlier product real TRAIN parity belongs to product1fb; it is not retagged as
an actual new-HPO checkpoint forward. No real images/GT/operational checkpoint/GPU
were accessed here, and no nonraw real-model quality or performance is asserted.

See [implementation contract](../../docs/UNIFIED_PHOTOMETRY_PREPARATION.md) and
[verification.json](verification.json) for pins and the exact unchanged scope.
