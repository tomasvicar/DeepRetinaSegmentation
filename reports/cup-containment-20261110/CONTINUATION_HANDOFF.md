# Fresh paired cup continuation — completed and audited

Prepared helper: `scripts/continue_cup_containment.py`.
SHA256: `0379fdfc146a693b26f0623986e5cb8593356cd5289c4929a87caf8696334c47`.
Original proposal stays unchanged, SHA256
`3f82e625b62f089bc7a3a36a9cfb3072e8367821147fba48e1f9831826d5f4ee`.
Root review passed in commit `1002a00`,
[`continuation-root-review.json`](https://github.com/tomasvicar/DeepRetinaSegmentation/blob/1002a00/reports/cup-containment-20261110/continuation-root-review.json).
One bounded continuation was authorized and started18:36:58 UTC on2026-11-10.
Child PID/process group1303824, external supervisor1303822. Staged helper/proposal
and root review are read-only outside immutable b413. Original prefix/reference
files remain unchanged. Launch artifacts and the review copy are in `continuation/`.

Actual Ada output:
`~/code/DeepRetinaSegmentation_cup_containment_runs/fresh-paired-20261110T183601Z/`.
`continuation.log` is the live log, `result/per-photo.jsonl` holds paired counts,
and `terminal.json` will record the external supervisor result. Its fixed wall
deadline is19:56:58 UTC. Expected completion is approximately19:36–19:45 UTC.
The supervisor checks child RSS every2s and signals only its own isolated process
group on4800s/2GiB limits; no automatic retry. Original training/other jobs are
untouched.

First actual progress:157 reused records plus2 new records. Photo158 truthfully
logs cup FP+1 versus historical counts with `exact_counts=false`; both dedicated
disc counts remain exact. GT/source/input pins passed and fresh cup/clips come
from the same arrays. RSS796MiB after42s. This is startup/progress proof, not a
completed validation result.

## Preserved earlier work and timing

The failed full run executed immutable `b413416` from 17:44:59 to 18:05:09 UTC,
1210.0573 seconds. It completed157 paired records and attempted photo158 before
the old exact native-count gate rejected1FP drift. Neither its output nor
historical references are changed. The completed prefix is byte-pinned:

- `402aa0524f07031babb08b1a4c01c30e38a14050c569205cc8edc9f1244705ca`
- Terminal proof `9f34b971e1a3becc8a4a3cf001a68ed77b3aa1c814e9ba67e707a7dba2756348`
- Source/data package `4e24ddf235bd108bebe35437320ad168d332df3274fbd2a8c98567ef07181bfa`
- Composer helper `28654e8c43973167031279f60a247cc6e298f1541e47828947d2657a0f12acdc`

Ada original run directory:
`~/code/DeepRetinaSegmentation_cup_containment_runs/full-20261110T174434Z/`.
Prefix is `result/per-photo.jsonl`; proof is `terminal.json`. Original native
default references, failed run, five-photo canary and one-photo forensics remain
available independently. The old success watcher correctly skipped composition.

The continuation report will disclose **prior aborted wall time + continuation
wall time**, rather than claim only continuation time for the612-photo study.
This sum explicitly excludes earlier canary/forensic probes. Failed-run per-model
timings were not flushed, so they remain unknown, not imputed from the canary.
New per-model inference timing is reported separately.

## Gates and outputs

Before loading models, the helper verifies its external SHA, original proposal,
b413 Git/source package, composer, canonical manifest/content, original reference
hashes, exact prefix bytes and terminal proof. It checks the sorted first157 of
612 supported VAL IDs, sources, record inventories, original exact native proofs
and all count/subset invariants. A metadata-only preflight passed for the actual
157 records; see `continuation-prefix-metadata-preflight.json`. This preflight
did not decode any image/GT array or load model weights.

On authorized execution, decoded prefix GT support, first-observer traces and
image geometry are rechecked with **zero prefix model forwards**. Original prefix
per-photo input hashes were not saved; exact canonical image bytes plus frozen
data code bind those inputs, and current input hashes are explicitly labelled as
observed at continuation. The455 remaining photos get fresh model input hashes,
one universal cup forward plus each dedicated disc forward. Native cup and its
two intersections are counted from these **same arrays**.

GT positive/valid support and source/annotation identity remain hard exact gates.
Fresh-versus-historical TP/FP/FN differences are logged as differences, with exact
booleans per model; they are never relabelled exact or hidden by tolerance. Prefix
results retain their original-phase exact proof separately. No threshold,
normalization, geometry, annotation, weight or model-source change is introduced.

Fresh output contains all612 paired per-photo count records, a complete report,
and four explicitly named770-photo composites: native cup in each D0/D1 reference
context, and cup clipped by the corresponding fresh dedicated disc. Every
vascular/disc count and AV accuracy is copied unchanged from the corresponding
historical strong reference. Fresh standalone disc drift is diagnostic only.
The cup baseline for containment effect is **fresh native**, while historical
native drift is a separate comparison. Reports include pooled/per-source TP
removal, FP reduction, FN increase and all5 pooled/source-macro metrics. None is
an independent final superiority claim.

CPU runtime is fixed to2 intra/1 interop, nice10 or lower priority, CUDA hidden,
OMP/MKL/OpenBLAS2, original uv Python executable and Torch2.6.0+cu124. Internal
checks enforce4800s and observed maxRSS below2GiB after each head; the authorized
launcher should additionally supervise the whole process externally at the same
limits. A fresh output directory is mandatory and protected source/canonical/
ownCloud/old-run trees are refused. Failure preserves partial data in the fresh
directory; an existing output cannot receive even a new failure sidecar.

## Tests and execution status

**16 tiny CPU tests passed in0.05s**, Ada CPU1/nice10/CUDAhidden, without real
images/checkpoints or forwards. Fixtures reject changed/skipped/duplicate prefix,
incorrect source/observer/GT support, false historical exact proofs, nonfinite
outputs and non-subset clipping. They demonstrate fresh-native versus clipping
coupling under simulated historical1FP drift, preservation of non-cup fields,
empty/ignored support handling and no overwrite/protected-tree/failure-sidecar
writes. A wrong frozen source is rejected before data/model operations.

Actual continuation is running under the separately recorded root approval.
No original failure gate was relaxed in place. The new experiment is a separately
labelled fresh paired comparison. A complete report is still pending.

## Reviewed command contract

Use the existing uv-frozen Python; no installation is required. Keep b413
immutable and deploy this reviewed helper outside it. Set `HELPER`, `PLAN` and
fresh `OUTPUT` to reviewed artifact paths. The exact executed expansion is saved
in `continuation/external-launch.json`; this template is documentation only.

```bash
SOURCE="$HOME/code/DeepRetinaSegmentation_cup_containment"
OLD="$HOME/code/DeepRetinaSegmentation_cup_containment_runs/full-20261110T174434Z"
PYTHON="$HOME/code/DeepRetinaSegmentation_unified_run_87cf045/.venv/bin/python"
env CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 \
  PYTHONPATH="$SOURCE/src:$SOURCE/scripts" nice -n10 "$PYTHON" "$HELPER" \
  --source "$SOURCE" --plan "$PLAN" \
  --manifest "$HOME/code/DeepRetinaSegmentation_for_Leipzig/data/unified25/manifest.json" \
  --reference-dir "$SOURCE/reports/disc-reference" \
  --prefix "$OLD/result/per-photo.jsonl" --prefix-terminal "$OLD/terminal.json" \
  --output "$OUTPUT" \
  --expected-helper-sha256 0379fdfc146a693b26f0623986e5cb8593356cd5289c4929a87caf8696334c47
```

No TEST, GPU job, training migration or automatic scheduling is part of this helper.

## Completed result, 2026-11-10 19:34:35 UTC

The bounded supervisor returned0. All612 cup-supported photos (157 explicitly
reused+455 new) and four770-photo composites are complete. Post-run count/artifact
audit passed; `complete/artifact-audit.json` verifies byte hashes, exact GT support
and unchanged non-cup counts/AV accuracy. Original references remain untouched.
Full report SHA:
`ccb0ec383488e50966caa5fa1268ad95251f4118909bf09d41d9d1784c0a74f8`.

Continuation wall3454.931s, prior aborted full attempt1210.057s, combined4664.988s
(77.75min). This sum excludes earlier canary/forensic probes and is not claimed
as a clean from-scratch runtime. External peak observed RSS1043512KiB (1019MiB).

| Cup Dice, pooled within source | Fresh native | Clip by fresh D0 | Clip by fresh D1 |
|---|---:|---:|---:|
| Drishti | .663362 | .692782 | .693229 |
| G1020 | .333952 | .360485 | .366654 |
| REFUGE | .432987 | .461106 | .458413 |
| RIM-ONE | .073239 | .239833 | .155999 |
| Pooled | .401057 | .435399 | .434542 |
| Source macro | .375885 | .438552 | .418574 |

D0 removes80891 true positives and1450920 false positives; D1 removes26129 true
positives and1227657 false positives. GT positive/valid counts stay exact. D0
cup recall declines from.981858 to.954932; D1 to.973160. No universal benefit is
claimed; even with higher aggregate Dice, genuine cup pixels are removed.
RIM-ONE's earlier experimental physical-calibration limitation still applies.

All5 source macro changes from.654498 to.667032 in the unchanged D0 reference
context and.653996 to.662534 in D1 context. The paired difference comes solely
from cup containment. Historical native drift is separate: two cup photos have
one extra FP each (total+2), no TP/FN/support change. Dedicated D0/D1 native
counts match their historical references exactly on all612 supported photos.
The two fresh-native controls, both containment variants, all per-photo counts
and source-level removals/drift remain available under `complete/`. This optional
VAL postprocessing study changes no public default, model weight or native route,
and does not establish independent or test performance.
