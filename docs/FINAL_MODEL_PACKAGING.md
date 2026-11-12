# Package a selected unified model without changing its frozen identity

This procedure has been applied to the separately named
[`unified-raw512-selected`](UNIFIED_RAW512_SELECTED.md) release: the registry now
contains 13 models, and all 12 prior blobs/entries/aliases remain unchanged. The
CPU export utility itself does not select a model or add a registry entry; those
are separate reviewed release steps. Formal training-completion/freeze/TEST
evidence and ordinary product prediction remain separate contracts.

## Keep original selection/TEST bytes

The research `selection.check_selection` binds the selected role, exact original
checkpoint SHA, original family/QA/protocol/corpus and full inference protocol.
`heldout.evaluate_reviewed_heldout` reloads its checked immutable original snapshot
and enforces threshold, TTA and tiling from that frozen protocol, with no new
inference overrides. A converted CPU file has a different byte SHA even when all
weights are identical. **Do not substitute it into the original gate, edit the
selection JSON, retag historical VAL or claim its file SHA was TEST-evaluated.**

Keep original BEST/LAST/CSV/completion inputs and the selection/TEST reports with
their original pins. The export records an explicit original→artifact relationship;
it does not make or authenticate a selection decision. A selected LAST must use
its own epoch/coverage/score, not borrow BEST metadata, and vice versa.

## CPU-only inference export

From the installed checkout, after the actual original checkpoint SHA is reviewed:

```bash
CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
uv run --no-sync python -m deepretina.inference_export \
  --source /immutable/original-selected.pt \
  --expected-original-sha256 REVIEWED_ORIGINAL_CHECKPOINT_SHA256 \
  --output /new/export/inference.pt
```

Python equivalent:

```python
from deepretina.inference_export import export_unified_inference

proof = export_unified_inference(
    "/immutable/original-selected.pt", "/new/export/inference.pt",
    expected_original_sha256="REVIEWED_ORIGINAL_CHECKPOINT_SHA256",
)
```

The placeholder must be replaced by the actual reviewed64-digit SHA, not a
filename hash or an invented value. The utility reads one checksum-verified byte
snapshot with `weights_only=True,map_location="cpu"`, accepts unified format1,
validates finite model tensors and actual product loader grammar, and clones model
storage onto CPU. Tensor name/dtype/shape/bytes and original configuration,
architecture, normalization, preprocessing/photometry, support/data/source/geometry
and selected epoch metadata are preserved. No quantization, field normalization,
training, inference or dataset access occurs. CPU constructors/strict state loads
are used to verify compatibility, with CPU RNG restored afterward.

Training-only top-level `training_state`, optimizer, scheduler-state and RNG
fields are removed. The original historical `config.device="cuda:0"` may remain:
it describes training and is **not** a GPU tensor/storage. The product still maps
weights safely to CPU before choosing the prediction device. Unexpected non-JSON
metadata/tensors outside the model state are refused instead of silently dropping
inference metadata. Already-exported inputs are refused; start from the original.

Outputs are a new `inference.pt` plus `inference.pt.export.json`. The payload is
marked `inference_only`; its internal provenance retains the original SHA and
model-state digest. The separate sidecar records the actual new artifact SHA,
original/artifact sizes, removed fields and epoch/source identities. These hashes
have different meanings and must stay separately named. The new file cannot be
used as an optimizer/RNG resume checkpoint. File size reduction from removing
Adam/RNG is **not model compression or a smaller parameter count**.

Neither output may already exist, including dangling symlinks; exclusive writes
refuse races without overwriting other files. If an I/O error occurs between the
two writes, a valid artifact may remain without the sidecar: treat that as an
incomplete operation, preserve the failure, and review a new destination. The
utility does not remove user files or claim atomic two-file publication.

## Minimal final release steps after genuine freeze/review

1. Preserve the original selected model/role and its original completion/freeze/
   measured VAL/TEST evidence. Record exact source/runtime/corpus/protocol pins.
   Do not package an incompletely eligible BEST because LAST is complete.
2. Export once with the reviewed original SHA into a new directory. Independently
   check exported tensors/state/config against the origin and verify no optimizer/
   RNG/training state or non-CPU model storage remains. Keep the export sidecar.
3. Check product architecture/profile compatibility **before choosing a release
   file**. Current compact UNet/raw or supported saved CLAHE profiles load without
   a new adapter; width24 itself is supported by the same architecture. A future
   `compact_unet_av_residual` winner would need its own verified compatibility
   port: current product allows only `compact_unet`/`legacy_universal`. Never strip
   unknown config fields or silently relabel that architecture.
4. Compare the original and exported model under the same actual prepared input,
   device, FP32 activation/joint-A/V decoding, threshold/TTA/tiling and once-only
   preparation. A small verified TRAIN input can prove integration, not new whole
   corpus quality. Compare the product port to the frozen research implementation
   when their source identities differ. Record actual numerical differences.
5. Add **one separately named** registry entry/file only after root review. Its
   `sha256` is the exported artifact SHA; retain original checkpoint SHA, selected
   role/epoch, model-state digest, selection SHA and public-safe provenance links
   distinctly. Keep all previous blobs/aliases; do not repoint `vessels`, `av` or
   `disc_cup`, or relabel an experimental preview as the final model.
6. Copy the frozen inference protocol's **four actual values** into this new
   model's `inference_defaults`: `tta`, `threshold`, `tile_size`, `overlap`.
   For a tiled rot4/.35/512/64 decision that means exactly those four values,
   not merely a suggested command in the model card. Explicit user overrides
   remain possible and are a different protocol from the advertised measurements.
   The CLI and `predict_bundled_unified` share resolution; direct custom checkpoint
   prediction retains ordinary defaults. See [defaults](BUNDLED_INFERENCE_DEFAULTS.md).
7. On a small explicitly identified TRAIN raw photo/FOV, verify CLI and public API
   with omitted options reproduce the reviewed resolved protocol and outputs;
   inspect `prediction.json` actual threshold/TTA/tiling and exported artifact SHA.
   Keep quality and cost measurements matched: do not combine none latency with
   rot4 quality, or transplant a calibrated original-FOV claim to another domain.
8. Build one wheel with `uv build --wheel`, install with `uv pip install --no-deps
   --target` into an owned isolated HOME directory using the existing runtime,
   then run **outside the checkout**. Verify actual module origin, all old/new
   weight hashes, aliases/registry, default resolution, CLI/API imports/help and
   one matched small functional prediction. This updates package-source proof;
   previous wheel proofs do not cover a later new module or model file. No sdist
   or source ZIP is needed for the repo handoff.
9. Update model card/README/status/test inventory with exact role, original and
   exported identities, full protocol, all head/source regressions and TEST state.
   Link original measured quality evidence and conversion/installed-wheel checks
   separately. If TEST has not actually run, state it remains pending.

No extra selection-loader change is needed merely to ship a CPU inference copy:
strict selection/TEST continues using original bytes, and product prediction
validates its exported artifact checksum. The mapping is explicit release evidence,
not permission to bypass a gate. Rebuilding a formal selection around a different
artifact would be a separately reviewed scientific operation, not packaging.

## Existing evidence and limits

The [TPE preview release](../reports/release/unified-v3-tpe-preview/README.md)
verifies a verbatim genuine BEST58 with no optimizer state; its original bytes
were intentionally unchanged. That is not an inference-only conversion proof.
The training writer stores `model.state_dict()` directly on its training device;
a CPU `map_location` audit alone does not prove that its original serialization
has CPU-only storage. The new exporter removes this uncertainty for a new artifact.

The existing [V2 control](UNIFIED_V2_CONTROL.md) intentionally distributes the
original LAST with optimizer/RNG state. Do **not** advertise the entire current
wheel as optimizer-free, or silently change that older blob while adding a new
inference-only model. Original5 reference-model conversions are separate preserved
safe artifacts. Existing wheel/parity reports are linked in [TESTING.md](TESTING.md).

Initial implementation used tiny synthetic CPU export/load/forward parity tests:
[verification](../reports/release/CPU-inference-export-20261112/verification.json).
Subsequent [actual existing bundled BEST58 export preflight](../reports/release/CPU-inference-export-20261112/actual-bundled-preview-preflight.json)
completed onCPU with external exit0. All128 model tensors/config were preserved,
original SHA/stat unchanged, and safe reload without map_location confirmed CPU
storage with CUDA uninitialized. Original20d4caab…4d4 and artifact696038a0…5ce
remain separate identities; the private artifact is outsideGit. BEST58 already
has no optimizer fields, so this primarily checks real format/config and storage
compatibility. No forward/realimage/GT/TEST or finalselection/release occurred;
all12 original packaged blobs, registry entries and defaults remain unchanged.

## Selected raw512 deployment copy

The subsequently frozen raw512 BEST50 is now separately named
[`unified-raw512-selected`](UNIFIED_RAW512_SELECTED.md). Its four defaults match
rot4/.35/512/64, and the original521b…/CPU artifact292e… identities are explicitly
separate. The [release checks](../reports/release/unified-raw512-selected-20261112/README.md)
record actual CPU export, preserved old12 blobs/aliases, focused tests, exact
single-TRAIN probability/mask parity and installed-wheel checks. TEST was pending
at packaging; this evidence does not add a new quality or clinical claim.
