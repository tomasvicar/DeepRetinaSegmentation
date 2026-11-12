# Final prediction benchmark recipe — prepared, not executed

Use `scripts/benchmark_strong_pipeline.py` after the checkpoint and inference
protocol have been selected separately. This script does not select a model,
change aliases/defaults, run a quality evaluation or access TEST/segmentation GT.
No real model benchmark was run for this preparation.

## Reuse and exact original composition

The strong script loads **four** networks: dedicated vessel, dedicated raw Disc0,
universal three-head network used for CUP, and one AV classifier for both A/V.
It shares single-net preparation, retains each network's native tiles, calibrates
provided-FOV AV from predicted vessels, and applies the explicitly selected cup
policy. The fixed product composition uses `disc0/raw` plus `inside-disc`, using
that SAME disc forward. Universal vessel/disc outputs are discarded; its complete
network still contributes parameters, storage and compute. Five output masks do
not mean five networks. No extra bundled checkpoint is loaded.

The original checkpoints are pinned by the supplied completed calibrated
reference; cup/disc policy must match that reference. Historical composed
reference528233fe… used the fixed inside-Disc0 recipe and four model SHAs
`dae07480…`/`7d6ddc3b…`/`9fa4f9ff…`/`82428903…` for vessel/disc/universal/AV.
It is a different complete evidence envelope from original182 and must not
borrow that report's full quality metrics silently. Existing product TRAIN
parity evidence confirms the five composed masks match the historical3a03
benchmark operation; this preparation does not repeat a model forward.

The current `frozen_original_reference` VAL schema (including original182) is
also accepted explicitly. Its complete `policy_spec` must match the fixed
four-network policy SHAeed8e889…07705, including all four model identities; the
actual files must match the unchanged top-level hashes. Its original family,
whole report SHA and GT policy are recorded directly. No synthetic historical
cup-component fields are inserted into the report. Changed cup/AV policy,
checkpoint identity, disc variant, TEST/preview or incompatible schema is rejected.
The [schema check](../reports/final-benchmark-preparation-20261112/frozen-reference-schema-check.json)
records51 targeted test cases and an actual original182 metadata/four-weight-byte
check, with CUDA uninitialized and no model deserialization/forward. This is not
a timing or memory-fit result. Earlier staged timing script bytes are historical;
use the current script and matching helper for the eventual final measurement.

Do not use `benchmark_latency.py`'s old benchmark entrypoint for this comparison:
it measures a universal+AV **two-network proxy**, with uncertified raw AV and,
for nonraw unified, unlike model-only timing. Only its shared CUDA-synchronized
`measure` helper is reused by the strong script.

## Select inputs and protocol before timing

Keep an immutable small list of real **TRAIN raw RGB photographs and official
original-frame FOV assets** with original byte hashes and their TRAIN provenance.
Use the same paired input bytes in both pipelines. The existing DRIVE TRAIN21
example has RGB SHA`e7af55375ba088ac63db120377eaf94b2405fe80b7874d6e1d236cda4f8c4c83`
and official-FOV SHA`acf23eeef91bd7c59624b9c2bb014703bcb37ef65cc8da2c0d221f3ca43dadc7`.
It is a repeatable single-geometry control, not a representative corpus. Add
other TRAIN geometries only through a separately fixed input list, never by
selecting photographs after seeing speed. No mask NPZ, targets, VAL or TEST
image reads are needed. TRAIN provenance is an operator input contract; this
single-photo benchmark does not infer a split from an arbitrary filename.

Bind the exact already-selected checkpoint SHA, original reference SHA/component
files, saved photometry/full configuration, TTA mode, scalar threshold and resolved
tile size/overlap. The explicit byte-SHA option prevents timing another checkpoint;
it does not confer completion/freeze eligibility. The ordinary loader retains
strict saved descriptors/backends. Full-frame photometry happens once per call,
before TTA/tiling. Whole-view models reject explicit tiling overrides.

## Three distinct timing phases

1. `--phase-timings` records model deserialization/device transfer separately for
   the four original networks and unified. Input/model SHA checks precede loading
   and warm filesystem caches: these are **not cold disk or process-startup times**.
2. Every warmup call is recorded separately, including the first call after load.
   One warmup and three measured calls per pipeline is the historical minimum.
   Keep all samples; initialization/cache effects belong to warmup, not a faster
   fabricated cold prediction. CUDA is synchronized around every recorded phase.
3. Default measured calls include full-frame geometry/photometry, model inference,
   original-frame restoration and five-mask decoding on resident decoded input.
   `--include-input-io` additionally reads/decodes/hashes the same pinned RGB/FOV
   inside each call. Run these as separate clearly labeled invocations if both
   scopes are needed. Never compare an input-IO-inclusive arm to an exclusive arm.

Both scopes exclude model loading, output serialization/overlays and CLI process
startup. They are complete **five-mask pipeline** clocks, not complete product API
clocks. `predict_original` additionally builds continuous probability, conditional
AV/support arrays and metadata, and loads models per API call. Public APIs/CLI
also perform their own input/output work. A future claim about those interfaces
must time their actual entrypoints separately; do not sum medians from these
phases or label these mask-only clocks as product API latency.

The final retained mask shape/dtype/hash/foreground counts are checked outside
clocks. Verification makes no extra NN call. Parameters are counted from actual
loaded models. Artifact byte size and model tensor bytes/dtypes are separate;
LAST may include optimizer state. No optimizer stripping, compression, weight
transformation or BF16-inference assumption is made. Training precision metadata
is preserved; this benchmark enforces FP32/no autocast and records TF32/backend
settings rather than claiming identical cross-device arithmetic.

## Reusable invocation, still requiring a separately approved resource packet

Use a fresh output, immutable evaluation source and the existing owned external
observer/cap. The script does not acquire a shared GPU lock or provide a scheduler.
Choose CPU affinity/thread counts, memory admission and a whole-run time cap for
the actual selected model/cohort. CPU example flags below match the historical
one-thread comparison; replace device/resources only through the reviewed packet.
`$TTA/$THRESHOLD/$TILE/$OVERLAP` come from the selected protocol, not this recipe.

```sh
CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
 PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src \
 taskset -c "$APPROVED_CPU" nice -n 17 ionice -c 3 \
 timeout --signal=TERM --kill-after=10s "$APPROVED_CAP" "$PYTHON" \
 scripts/benchmark_strong_pipeline.py \
 --input "$TRAIN_RGB" --fov "$OFFICIAL_TRAIN_FOV" --dataset "$DATASET" \
 --vessel "$ORIGINAL_VESSEL" --disc "$ORIGINAL_DISC0" \
 --universal "$ORIGINAL_UNIVERSAL" --av "$ORIGINAL_AV" \
 --reference-report "$EXACT_COMPOSED_REFERENCE" \
 --original-disc-variant disc0/raw --original-cup-policy inside-disc \
 --unified "$SELECTED_CHECKPOINT" --expected-unified-sha256 "$SELECTED_SHA" \
 --unified-tta "$TTA" --unified-threshold "$THRESHOLD" \
 --unified-tile-size "$TILE" --unified-overlap "$OVERLAP" \
 --device cpu --cpu-threads 1 --interop-threads 1 --opencv-threads 1 \
 --phase-timings --warmup 1 --repeats 3 --output "$NEW_OUTPUT"
```

Add `--include-input-io` only for the separately labeled IO-inclusive run. Omit
explicit tile/overlap flags for whole-view models. All loaded checkpoints,
reference and input snapshots are identified by encoded SHA. Reports record
runtime versions, actual affinity/threads/nice, device/physical UUID, precision,
backend flags, process RSS, CUDA allocation and package/helper source hashes.
CUDA allocation peaks cover measured calls after warmup, not load/warmup or
external allocations; whole-process RSS is not a per-pipeline measurement.
Sanitize operational paths/UUIDs before publishing.

Fixed original-then-unified order and warm caches remain qualifications. The
historical [one-TRAIN CPU TTA example](../reports/benchmark/tpe72-tta-cpu-drive21)
found original59.59s, TPE NONE8.75s, rot433.20s and dihedral865.53s. Those medians
belong only to that checkpoint/protocol/photo/runtime; NONE speed cannot be
attached to TTA quality or a later selected model. Report per-photo samples and
paired within-run ratios, device/precision and all scopes. A speed observation
is not model quality, training-seed uncertainty or evidence of convergence.

Preparation validation:34 lightweight tests passed2.70s; Ruff check passed. These
use synthetic tiny RGB/FOV arrays, fake checkpoint bytes, mocked predictions and
mocked CUDA timing calls. They verify exact TTA/threshold/tiling forwarding,
full-frame photometry ordering, phase/IO scopes and call counts, model-reference
release, CLI/resource metadata and that CUDA remains uninitialized. No real model,
NN forward, retinal/GT/TEST data or GPU was used for this preparation.
