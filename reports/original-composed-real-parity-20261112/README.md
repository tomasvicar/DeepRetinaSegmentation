# Original composition: first actual TRAIN identity preflight

The one-shot CPU operation on Ada at 2026-11-12 00:35:37 UTC stopped before
model loading/forward. Child and external shell both returned **1**; no timeout,
RSS violation or prediction directory occurred. The owned group peak was
452,624 KiB and elapsed time was 1.27 seconds. No GPU, GT, VAL or TEST was read.
No automatic retry occurred.

The pinned encoded DRIVE21 RGB/FOV files matched the earlier timing evidence;
the exact corpus manifest SHA `17a2277c39cb37b7eee1b2063cfd13f2cf4adf4ebdf86cf449a1f51367c5a767`
contained the expected `drive_21_training_vessels` TRAIN row. The driver incorrectly
compared a pixels-only SHA against canonical `raw_rgb_sha256`.
`prepare.decoded_photo_id` actually hashes original uint8 RGB **shape as three
little-endian int64 values, followed by contiguous decoded pixels**. This
convention is identical in frozen source 6b3dfad and the product source. Applying
it to this same image gives the exact canonical ID
`sha256:77d1a4436a1cac7f3397c9a2f2e84bf1e2d2b13647a833a3d7559aa6aa0d6925`.

The original failure and receipts are preserved; this is not successful parity
evidence. A corrected separately authorized attempt must retain the encoded
input pins, manifest pin, row ID/split/source-name checks and compare BOTH the
shape-prefixed hash and canonical photo ID before any model load.

`stage-manifest.json` identifies every actual staged source/input/helper file.
The code was staged from product commit a045bb1, with exact four original model
copies under a new namespace, separate from frozen training sources.
`driver.py`, `observe.py` and `launch-once.sh` are the actual failed invocation.
External byte-pinned dependencies were the frozen timing benchmark script
SHA3a0324e09c128b36381b1ab219c88c8f10e06090918c10023901588da7ac1aae
and owned process-group helper SHA382a39797a2695bfc5a4d55e7df1eb6e01e23db9d4fced30ca6021e67607a8a7.

Parent independently reran the eleven toy integration tests: PASS in 4.37s.
No composite real-photo success is claimed by that test result.

## Second preflight: runtime tiling API mismatch

After explicit root review of the shape-prefixed identity convention, a new
namespace `DeepRetinaSegmentation_original_composed_parity_20261112_attempt02`
was authorized. It passed the corrected exact identity guard, then stopped before
the first forward: checkpoint `tiling` also contains descriptive `blend`/`window`
fields, which `tiled_logits` does not accept. Both actual exits were 1; elapsed
1.49 seconds, peak owned-group RSS 555,420 KiB. Source a045bb1 and all original
artifacts remain unchanged. No automatic retry occurred.

The actual benchmark main filters runtime tiling to `crop_size`/`border` before
calling `original_operation`. Product fix e249b81 now selects those exact two
arguments while preserving the complete payload in exported provenance. The toy
fixtures now contain the real descriptor fields and the literal reference setup
mimics benchmark main's normalization. All eleven focused tests passed in 2.29s;
Ruff passed. This is a runtime argument fix, not a change of tiling math.

## Explicitly approved corrected attempt03

A third separately authorized namespace started 2026-11-12 00:38:35.624918UTC on
Ada, source e249b81, observer2744783/child2744784. CPU1/intra1/inter1, affinity4,
nice17, idle IO, CUDA hidden, 590s work/600s whole cap and 8GiB owned-group RSS
limit. All58 stage pins passed before launch; MemAvailable was40,467,924KiB.
The original benchmark function is compiled directly from its hash-pinned AST;
its main's two runtime tiling arguments are reproduced. API loader instrumentation
captures the actual four loaded objects once and reuses them for the reference;
no extra model or operator is substituted. Child and external shell receipts,
full finite-score/mask comparison and post-child source hashes determine success.

## Actual attempt03 result

**PASS** at2026-11-12 00:39:18UTC: observed child0 and external observer0, no timeout,
no RSS limit, elapsed43.1243s. The owned-group sampled peak was2,902,100KiB;
process `ru_maxrss` separately reported2,944,968KiB. All five `(5,584,565)` boolean
masks agree exactly with the actual frozen benchmark operation, differences
`[0,0,0,0,0]`, both raw-mask SHA
`60054453a1f5522187c1c476567e7fbe747d3d2e42339126adf22b950868872c`.
All exported scores are finite, all four loaded component identities agree,
all58 staged source/input/helper files were unchanged after both operations,
and CUDA was not initialized. Torch intra/inter threads were1/1. Candidate21.89s
includes actual weight loading/export; reference19.85s reuses the exact model
objects. These times are **not** a speed comparison. No GT arrays, VAL/TEST or
GPU were used; no quality or arbitrary-camera claim follows.

The successful original JSON/log bytes were copied with exact remote SHA checks
(`attempt03/remote-artifact-sha256.json`). No weights, RGB/FOV pixels or predicted
image arrays are published here. All three actual Ada namespaces were made
read-only after completion; the unsuccessful ones remain separate and unchanged.
The actual external source/helper dependencies are identified in stage manifests;
full duplicate helper/benchmark code is retained at the immutable Ada namespace,
not copied into this product report. `artifact-sha256.json` pins the copied
small artifacts and invocation scripts.
