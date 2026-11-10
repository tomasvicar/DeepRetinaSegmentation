# Optional BIOMISA fixed export-role policy

This is a separate, explicitly selected native annotation policy. The default
Python API remains quarantine and no BIOMISA selector is automatically included
in preparation, v1, current HPO, or any training-ready v2 manifest.

## Evidence and exact scope

The local `BIOMISA.zip` has45 labelled fundus-eye folders, with44 named Disc and45
named Cup files. They contain actual binary PNGs despite `.jpg` extensions.
Every complete pair has the **named Disc strictly inside named Cup**. Three local
native overlays spanning glaucomatous and healthy eyes independently show the
named Cup as the outer disc-like boundary and named Disc as the inner cup-like
region. Encoded resource hashes, areas and all nesting counts were retained by
[the preceding archive audit](CORPUS_V2_ARCHIVE_AUDIT.md). No color threshold,
area sorting, component filtering, contour filling or clipping is used to fix it.

The source matches the [authors' composite fundus/OCT repository](https://data.mendeley.com/datasets/trghs22fpg/4),
not the different50-image OCT/fundus collection. The [authors'2022 paper abstract](https://repository.adu.ac.ae/items/5005dc3f-8011-49ef-88de-54b103156e1e)
confirms an annotation application for fundus/OCT and Topcon3D OCT2000 acquisition.
These primary descriptions support the modality/annotation context, **but do not
explicitly acknowledge the filename inversion**. The role correction is a local
source-wide interpretation supported by all44 pairs and the three visual overlays,
not an author-certified erratum, recovered exporter semantics, clinical review,
or a universal claim about every BIOMISA release.

## Explicit opt-in API

```python
from deepretina.adapters.biomisa import biomisa, FIXED_POLICY

biomisa(source, emit)  # default: retain raw refs, all supervision validity false
biomisa(source, emit, label_policy=FIXED_POLICY)  # explicit locally audited policy
```

The fixed policy is `named_cup_is_disc_named_disc_is_cup`:

- Source filename ending`_Cup.jpg` supplies the inclusive **disc** channel.
- Source filename ending`_Disc.jpg` supplies the **cup** channel.
- A missing named Disc file means unknown cup, with cup validity false. The actual
  P10/Left case retains its named Cup as disc-only.
- Containment is checked **after** applying those fixed roles. A contradictory
  future sample fails; the decoder never sorts, swaps by size, clips, or repairs it.
- Pixels and original geometry are retained exactly; both raw resources and the
  explicit policy remain in metadata. Default quarantine emits all validity false,
  so an absent/counterintuitive label does not become background supervision.

Patient/paired-eye grouping uses the source cohort/P_N directories. Cross-cohort
identities are not established. These candidates stay train-only unless a reliable
crosswalk and new split review exist. A native exact decoded-photo check against
frozen v1 protects its heldouts; it does not detect re-encoded or unknown-patient
aliases.

## Physical scale remains qualified

The [manufacturer3DOCT2000 specification](https://topconhealthcare.com/wp-content/uploads/sites/8/2020/07/3DOCT-2000.pdf)
lists45° fundus coverage. Together with the authors' camera description this
supports an **inferred device-based45° proposal**, not a measured per-image angle,
verified local archive version, or proof of an acquisition zoom setting. The
adapter still sets `physical_calibration.field_of_view_degrees=None` and records
that explicit proposal separately. No silent legacy25 geometry is applied.

Before admission to a training-ready v2, review the policy, camera inference and
cross-source aliases, then freeze a new corpus/plan. Existing v1 data, annotations,
model defaults, checkpoint weights and running trials remain untouched.
