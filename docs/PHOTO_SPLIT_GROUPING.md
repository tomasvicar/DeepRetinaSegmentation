# Portable preparation: exact-photo split grouping

New modern `deepretina prepare` manifests identify each original photograph with
`photo_id=sha256:<digest>`. The digest is SHA256 of its RGB uint8 shape encoded as
little-endian int64 values, followed by contiguous decoded RGB bytes. Identity is
computed **before** Lanczos resizing and is independent of source filename, task,
annotator, supplied FOV mask or field-angle metadata. The hash is cached per
original source resource so observer/task variants do not repeatedly hash a large
photograph. This is a decoded-photo identity, not an archive/file identity.

Split assignment first constructs connected components over patient/source IDs,
photo IDs and shared prepared image paths, globally across datasets. For example,
if photo X belongs to patient A and also appears under another dataset's source B,
all photos of those source groups share a partition. PatientA's distinct left/right
photographs still have different photo IDs and image resources. This operation
does not merge their images or annotation geometry.

A known official test alias places the whole component in test; otherwise an
official validation alias places it in validation. Test is the stricter outcome
when validation/test aliases disagree. If any train-only source without a patient
crosswalk joins an official heldout component, preparation fails explicitly.
It cannot silently move that heldout photo into training. Remaining components
use deterministic source-family seed ranking, preserving the existing 20% validation
and 20% test policy for unsplit groups. Singleton source groups retain their prior
ranking key. `split_group_id` records each connected source component; it is not
a recovered clinical patient identifier.

The supplied DRHAGIS files `24.jpg` and `32.jpg` provide a real regression case:
both decode to 3168×4752 RGB and the same photo ID:
`sha256:5c819ce505bfd367cd316d1a437b4f0f182997439e92ab1e0ccf1d4ed1b43a3d`.
A read-only check of the actual archive verified their shared component despite
different source names. No source photographs were copied into these notes.

Tests additionally cover cross-dataset aliases, patient/photo transitive grouping,
separate patient eyes/images, official validation/test protection, train-only
conflict rejection, differing FOV masks, original-size hashing, geometry-sensitive
hashes and per-resource observer caching. Sixty targeted preparation and adapter
tests passed. The running unified experiment uses its separate immutable source
snapshot and corpus; this change does not rewrite that dataset or any existing
modern manifest. Reprepare to a new output directory to obtain photo grouping.

Exact decoded equality does not detect cropped/rescaled/recompressed near-duplicates
or establish patient identity for unrelated files. Supplied patient crosswalks and
conservative policies remain necessary; see [DATASETS.md](DATASETS.md) and the
patient-policy documentation. No new accuracy or universal patient-disjointness
claim follows from this grouping fix.
