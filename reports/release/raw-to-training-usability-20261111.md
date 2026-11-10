# RAW-only colleague workflow audit

Audited stable source `4ec7344` with only the repository and shared RAW archives
assumed available. No prepared corpus is assumed available to the colleague.

## Concrete gaps and fixes

The quickstart generated a fresh 17-source parent, while the V2 append CLI
accepted only the authors' historical `61032be2…` identity. No byte-reproduction
proof connected those steps. The new `prepare_training_data.py` performs the
existing preparation, known patient regrouping, full parent audit, independent
copy/append, and full combined audit. It passes the **actual newly audited**
parent fingerprint. The historical append default remains unchanged, with an
explicit audited-parent override exposed for advanced use. Positive TRAIN
supervision is required for every selected source before a complete-workflow
proof is written. Source data and existing output directories are not overwritten.

The default final manifest includes all 17 base sources plus DRIONS, GAMMA,
IDRiD disc and BIOMISA, with explicit reviewed BIO filename roles and the pinned
nominal-camera policy. Parent holdouts are preserved; IDRiD official TEST labels
are converted mechanically, without model evaluation.

The original universal backbone has an existing trainable five-output extension.
A complete scratch preset now exposes it without requiring manual config edits.
This uses the modern objective and boundary padding. Native three-output
single-net and four-input VesClass training entrypoints were absent in the audited
source; this is an explicit remaining implementation gap, not resolved by the
five-output extension. Historical replay is not claimed.

## Bounded verification

CPU-only, one numerical thread, no model training/forward on real data and no
writes to shared RAW or prepared production data. Focused suite:

```bash
PYTHONPATH=src OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  CUDA_VISIBLE_DEVICES='' uv run --frozen pytest -q \
  tests/test_prepare_training_data.py tests/test_prepare_corpus_v2.py \
  tests/test_prepare_unified_data.py
```

25 tests passed. The new bounded fixture reads actual-format synthetic DRIVE,
RITE and Drishti archives, then appends synthetic DRIONS expert contours through
the real writer/copy/audit path. It checks paired photo identity, all five label
channels (vein valid-negative in the artery-only fixture), invalid missing cup,
parent byte preservation/distinct inodes, and final schema2 content identity.
Other new-source adapters are empty only in this bounded test. Production has no
subset option and checks every required source's positive TRAIN support.

Other checks cover fail-before-write path/policy/archive errors, missing-source
rejection, actual audited parent pin propagation, explicit CLI pin routing and
the scratch five-output original-topology preset. CLI help was executed.

No complete 21-source rebuild was repeated during this usability audit. The new
workflow is tested composition, not a claim of historical canonical identity.
Platform copying currently requires GNU `cp`; output size and CPU duration are
estimates, not a new measured full-build result. Existing skimage deprecation
warnings remain; no label or geometry algorithms were changed here.
