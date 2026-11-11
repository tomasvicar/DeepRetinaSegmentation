# Independent processed RGB cache review

Reviewed prototypef030be53ba2cecee327c888aa54a6db7d5abfd20 and guard follow-up
fd44cdf. This is a source review and bounded synthetic preprocessing/cache audit;
no model forward/training, GPU, actual GT scoring, TEST evaluation or full-corpus
build was run by the reviewer.

One concrete P2 was found in f030: cache/profile validation occurred only during
Dataset construction. Its mutable config could subsequently switch to `raw` or
a different physical scale and `_image` would still serve processed cache pixels,
including from its populated Tensor LRU. Reported directly to the author. fd44
snapshots the admitted profile/scale and checks them **before LRU lookup**;
the two post-population mutation negatives pass. No observed training corruption
is claimed: the training entrypoint does not normally mutate those fields.

The reviewed final paths have these properties:

- Builder selects only eligible TRAIN/VAL photos from manifest metadata, uses
  image-derived/official `photos.json` FOV assets, verifies raw RGB aliases before
  transformation, and records encoded source RGB/FOV/alias hashes. It never
  derives a photometric FOV from target validity or labels.
- Full build checks canonical encoded bytes before/after; explicit TRAIN subset
  proofs are typed separately and rejected as full-cache admission. The index
  binds canonical/manifest/photos, descriptor/backend, operator and construction
  source, exact selected IDs, output byte hashes, shape and quantization.
- Canonical images and old raw derivatives remain unchanged. New output is staged
  outside canonical/ownCloud, has exclusive build locking, readonly completed
  files/directories, and a final completion marker/index fingerprint.
- Reader validates externally expected identity, current descriptor/operator,
  canonical metadata, aliases and coverage. Every accessed NPY's encoded hash is
  checked before decoding. It returns a private uint8 array; Dataset copies into
  its bounded128MiB float Tensor LRU. No persistent RGB mmap/dictionary is kept.
- Cached `_image` bypasses `apply_photometry` entirely. Uncached `_image`, VAL and
  raw-photo inference retain the same once-only full prepared-image operation.
  Labels/validity use the existing canonical/raw-derived reader separately.
- Train/resume and frozen HPO plans bind the portable cache identity. The new
  cache module enters the explicit training-source fingerprint; runtime paths
  may relocate while identity remains pinned. Missing photos do not fall back.

Twelve cache-focused pure CPU cases passed in4.65seconds, with the one tiny
training/resume case deliberately deselected. They include cached/uncached input,
target and crop-RNG parity, no double application, external/profile/source alias
mismatches, bit flips, subset/output guards, bounded array release and the fixed
mutable-config negatives. The author's broader202/35-case evidence is separate.
The earlier benchmark RAW bypass is also resolved: the final prototype explicitly
applies shared full-image photometry once in the complete strong operation; its
primitive benchmark labels preprocessing versus model-only scopes separately.

No remaining concrete blocker was found in the reviewed prototype. This does
not admit a whole build or GPU arm by itself. The actual three-TRAIN operator/
reader proof belongs to sourcef030; fd44 changes only the Dataset mutation guard,
not builder/operator bytes. Preserve that provenance rather than relabeling the
old measurement. Whole3299-photo capacity, complete byte audit and measured epoch
throughput remain subsequent evidence, not conclusions from this review.
