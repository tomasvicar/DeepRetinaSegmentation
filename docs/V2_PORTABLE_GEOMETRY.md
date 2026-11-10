# Portable geometry policy for unified corpus-v2 models

A newly trained unified model can carry the exact preparation policy in its
checkpoint. Prediction then needs the source name, not a remembered camera angle:

```bash
uv run deepretina predict --checkpoint runs/v2/best.pt \
  --input original_gamma.jpg --dataset gamma --output predictions/gamma
```

The same works for `drions`, `idrid_disc`, and `biomisa` when the checkpoint
declares their rules. No new bundled model alias or trained weights are added by
this change. A checkpoint without the policy keeps the existing behavior,
including the requirement to supply an explicit angle for unknown legacy names.

## Captured before training

`train_unified` detects new-source rows/geometry policy declarations in its
manifest snapshot. It reads the corpus-local `geometry-policy.json`, verifies
its exact bytes against both row policy hashes, validates schema/version25px/degree
and nominal/reported qualifications, and checks each row's selected rule against
its original native shape and field angle. Invalid/missing/mixed policies fail
before device/RNG/model setup or run-output creation. No GT-derived camera angle
or GT mask read is introduced by this policy capture.

The existing corpus-v2 producer contract is supported directly:

- `metadata.geometry_policy_sha256` and `metadata.policy_sha256` bind the sidecar.
- `metadata.physical_calibration` holds the selected rule and original native H,W.
- `metadata.original_geometry.original_shape` must agree with those native H,W.
- The top-level row angle must match the selected document rule.

The whole original JSON text, including its exact encoding/newline, is embedded
as a small safe primitive value in `run.json`, `best.pt` and `last.pt`:

```json
{"geometry_policy":{"format_version":1,"sha256":"original-file-SHA256","document_json":"original JSON text"}}
```

The portable wrapper has version1; the document has its own schema_version1.
Hash, size, source-scope, coordinate contract, rule and nominal-qualification
checks also run when loading a checkpoint. No external policy path, network
request, custom pickle class or new package dependency is needed for inference.
The supported document is limited to the four new source names; it cannot
replace the historical DRIVE/HRF/etc defaults.

Resume checks compare the embedded policy with the current manifest-bound policy
and with best/last provenance. The helper itself is included in the executable
training-source fingerprint. Existing source-fingerprint checks remain enforced;
this is not permission to migrate a running3157 checkpoint into new code. Do not
retrofit a policy onto an old model as if its training provenance had been known.

## Rule selection and honest overrides

The current reviewed document is
`corpusv2_nominal_fov_20261110_v1`, SHA256
`2591d8461b56433529105532c20203e0b17a48c781c1a74fd3a2901bcb301e3d`.
The angles come from that **embedded document**, not a newly hardcoded legacy
default table:

| Source/rule | Policy angle | Qualification |
|---|---:|---|
|DRIONS|30°|Explicit research assumption; camera/actual field unverified|
|GAMMA native H1934×W1956|30°|Resolution-associated Topcon inference, not EXIF/per-image calibration|
|GAMMA native H2000×W2992|45°|Resolution-associated KOWA inference, camera model unverified|
|GAMMA other native H,W|45°|Explicit document fallback, flagged as unseen native shape|
|IDRiD disc|50°|Dataset-reported acquisition; not per-eye metric calibration|
|BIOMISA|45°|Device-nominal angle; export/capture geometry unverified|

GAMMA matching uses the original decoded raw image H,W **before crop, scaling or
rotation**. Dimensions are never sorted or transposed. A resized/rotated export
can therefore enter the explicitly flagged fallback rather than silently being
identified as a known camera. Supplying no dataset does not infer GAMMA from its
shape. FOV-mask selection/crop and the original maximum-dimension scale anchor
continue to use `prepare_single_net`; no FOV-circle radius calibration is added.

An explicit angle always wins over a valid applicable policy:

```bash
uv run deepretina predict --checkpoint runs/v2/best.pt \
  --input original_gamma.jpg --dataset gamma \
  --field-of-view-degrees 30 --output predictions/gamma-explicit
```

`prediction.json` records the actual effective angle and policy ID/SHA, native
shape, selected nominal/reported rule, camera-attribution method, qualifications
and evidence. `policy_applied` distinguishes policy application from an explicit
override. An unknown native shape is recorded even under override; the override
does not claim that the nominal fallback or an actual measured calibration was
used. Broken version/hash metadata cannot be bypassed by providing an angle.
Other dataset names keep their legacy/explicit preprocessing behavior.

## Verification and scope

87 tests passed on isolated Ada CPU1/nice10/CUDA-hidden source: policy version/hash
binding, row/rule/native-shape mismatches, source-scope protection, exact GAMMA
shape and flagged fallback, explicit overrides, old checkpoint behavior, actual
raw prediction from a tiny generated safe checkpoint with native80×96 restoration,
and synthetic functionality training run/best/last persistence/resume guards.
The existing unified and vessel-loss suites were included. No real model training
or quality evaluation ran. An original read-only bundled weight is used only by
an existing synthetic historical-resume test; no source model was changed.

Selection agrees with the unmodified `prepare_corpus_v2.angle_for` at fa1cf03 on
seven known/unknown/flat-source cases, using a vendored project-owned reference.
The exact reviewed policy JSON is a small test fixture; no photographs, GT arrays
or weights are committed here.

A metadata-only check on the actual completed v2 manifest SHA
`17a2277c39cb37b7eee1b2063cfd13f2cf4adf4ebdf86cf449a1f51367c5a767`
also passed:7811 rows,446 newly policy-bound rows (DRIONS220/GAMMA100/IDRiD81/
BIOMISA45),7365 untouched parent rows. GAMMA rules selected30° for6 records and
45° for94. No actual RGB/GT array was decoded or model/CUDA imported by that check.
Reports are under `reports/v2-geometry/`. These checks establish compatibility and
coordinate-policy reproduction, not historical per-image camera certainty or a
segmentation-quality improvement. Stable defaults, corpus,3157 and live studies
remain unchanged.
