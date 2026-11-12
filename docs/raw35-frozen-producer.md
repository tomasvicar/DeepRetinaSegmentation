# Experimental frozen native RAW35 producer

This branch implements a reusable producer, not a prepared corpus or a new training arm. No actual data preparation, native image/GT/model read, remote query, GPU use or training was performed. The existing delivered for_Leipzig artifacts, native25 scratch60 run, historical canary, shared source branches and gates are unchanged. Current trainer/config admission continues to reject35.

The accepted scope is frozen17a TRAIN2529photos/5700records and VAL770photos/993records:3299photos/6693records,21TRAIN/15VAL row sources and612VAL groups. The metadata planner reproduced these counts and retains1,118 TEST records only inside a separate `test-inventory-unmaterialized.json` inventory. TEST paths never enter the35 manifest/photos or resource closure. The original TEST record is nested as `frozen25_record`, not claimed as materialized35.

## API and safe metadata planner

`deepretina.raw35_producer` supplies `FrozenPlan`, `dependencies`, `FilteredSource`, `Replay`, `load_plan`, `validate_profile` and `run`. The API shares the original21 source decoders and `prepare_single_net`; it does not call the old build/append execution flows or recompute splits. `FilteredSource` permits only the compiled closure before enumeration/find/bytes/read, including G1020 patient CSV, native FOV, implicit core labels, and the HRF/DRHAGIS extra A/V archive. The original source reader's TAR duplicate checking can read payloads before filtering, so this producer deliberately accepts indexed ZIP/directories only. Existing21 sources use those backends. Ambiguous transport aliases fail unless ZIP CRC/size metadata establishes the original reader's deterministic shortest identical alias. No byte-based fallback search runs outside the retained closure.

The optional `retains(stem, task)` hook added to the legacy segmentation adapters checks ARIA, ONHSD, RIGA, RIM and STARE observer identities before annotation decode. Without that hook, existing adapters behave as before. Ordinals and stems remain original, including observer4 when observers1–3 are excluded. All other adapters retain their existing algorithms and label role policies. Every requested emission must be present exactly once; unexpected or missing rows fail.

Metadata-only command, from this worktree using the existing HPO environment:

```sh
mkdir -p /var/home/tom/.raw35-producer-tmp
TMPDIR=/var/home/tom/.raw35-producer-tmp PYTHONPATH=src \
  /var/home/tom/code/DeepRetinaSegmentation_hpo/.venv/bin/python \
  -m deepretina.raw35_producer plan \
  --canonical /var/home/tom/code/DeepRetinaSegmentation_proposed_storage/corpusv2_2591_fa1cf03 \
  --geometry-policy tests/reference/geometry_policy/policy.json
```

This reads local JSON/source metadata only and prints membership plus exact resource requests. It neither opens source archives nor decodes frozen25 references. The source requests form the union of retained rows' RGB, annotation, backend and sidecar dependencies. The request list is not an assertion that remote archive members have been resolved or verified.

The immutable metadata pins are:

| Input | SHA256 |
|---|---|
| manifest25 |17a2277c39cb37b7eee1b2063cfd13f2cf4adf4ebdf86cf449a1f51367c5a767|
| photos25 |765709a1ddc08035565ebf3d54d414dd1f53a9882a5cfa9c0fd03811b8e48a4d|
| preparation25 |e7899af17ac94153a64752a25b52fba6001da361ae7a2893750d4bf2493389e2|
| prepared25 content receipt |00d9ef19d262eef74f0ca1350fec260c687dff5cc3a53f8911cbe3403e821e52|
| parent25 geometry policy |2591d8461b56433529105532c20203e0b17a48c781c1a74fd3a2901bcb301e3d|
| retained membership signature |08ef48ec32109c8971d5690066ac48781b768e2b3b8241ee1880513f66b828b5|

## Replay and output contract

Every retained observer/alias emission hashes native uint8 RGB, even if the same resource already emitted. It checks row/source/patient/observer/native-resource provenance and source split against frozen metadata. The frozen canonical owner emits first. Alias annotations use the owner's verified geometry/FOV; aliases cannot select a new owner by iteration order. Owner FOV is recomputed and25-replayed on each owner emission. Cross-resource owner grids remain in a bounded cache until their final retained annotation; unexpected owner ordering or the profile's cache ceiling fails for review.

For each emission, native RAW→25 RGB (both canonical and row alias), FOV, target and validity must equal frozen decoded arrays exactly before any35 files for that emission are written. Float32 targets remain float32; nearest/order0 transforms preserve soft values and unknown/partial validity. Source-specific RIM >127 semantics, RIM stereo left crop with native maximum dimension scale anchor, UoA watermark suppression, official/automatic FOV and BIOMISA fixed filename role mapping stay in the original decoders/preparer. Effective validity intersects image/annotation FOV. If a formerly supported valid/positive head or conditional A/V scoring population becomes empty, execution fails for review; rows never disappear or move partitions.

The35 arrays are prepared independently from native RGB/labels with `pixels_per_degree=35, normalized=False`. The output retains raw photo/record/source/group IDs, aliases, source ownership and patient provenance, with truthful35 shape/geometry/preprocessing metadata. `geometry-policy35.json` gets policy ID `raw35_frozen2591_20261112_v1`, a new35 hash, `target_pixels_per_degree=35`, `parent_pixels_per_degree=25`, and `parent_geometry_policy_sha256=2591…`. It preserves2591 separately as its parent25 source-rule document. Preparation and every photo carry the actual35 policy SHA and explicit parent lineage; photo `geometry_policy` carries `sha256`, `pixels_per_degree=35` and `parent_geometry_policy_sha256`. No old25 manifest/content/cache hash is retagged. The producer writes new manifest/photos/preparation, per-row replay/support report, integrity/content fingerprint and `_SUCCESS` only after complete membership succeeds. A failed attempt remains incomplete, never auto-resumed or overwritten. No cache or CLAHE/photometry artifact is built.

## Disabled execution profile and future command

`configs/experimental/raw35-frozen-disabled.json` is disabled and intentionally incomplete. Its ceilings are example execution bounds, not estimates. It lacks backend/Python pins, output destination and exact native encoded-byte identities. Changing a boolean alone cannot make it execute. A future reviewed profile must supply the frozen preparation's exact Python/package environment, a new exclusive local output parent, finite time/native-read/output/RSS/cache bounds, storage reserve and every archive closure:

```json
{
  "resources": {
    "ARCHIVE.zip": {
      "request_to_member": {"requested/suffix": "exact/indexed/member"},
      "members": {
        "exact/indexed/member": {"size": 123, "sha256": "64 lowercase hex characters"}
      }
    }
  }
}
```

The mapping keys must equal the planner's requests exactly. Member values must equal its resolved retained closure, without extra members. Resource sizes/hashes must come from approved existing inventories or an independently authorized bounded resource audit; this task did not obtain them by opening source data. All byte reads verify the exact encoded hash before image/GT decoding. Missing members, ambiguity, unconsumed resources, changed bytes/backend or lost support fail closed.

For a later explicitly approved execution, the CLI is:

```sh
TMPDIR=/var/home/tom/.raw35-producer-tmp PYTHONPATH=src \
  /var/home/tom/code/DeepRetinaSegmentation_hpo/.venv/bin/python \
  -m deepretina.raw35_producer produce \
  --canonical /var/home/tom/code/DeepRetinaSegmentation_proposed_storage/corpusv2_2591_fa1cf03 \
  --geometry-policy tests/reference/geometry_policy/policy.json \
  --profile /absolute/reviewed-profile.json --profile-sha256 REVIEWED_PROFILE_SHA256 \
  --data-root /var/home/tom/ownCloud/fundus_databases \
  --output /absolute/new-exclusive-local-Ada-namespace/raw35-train-val
```

Do not run this command under the present code-and-fixtures authorization. The producer checks time/native-read/RSS/free-space cooperatively before resource reads and after emissions; output byte totals are checked periodically and before completion. These guards cannot interrupt one already-running decoder, allocator or write; a future operator still needs external process/time/storage containment. It never writes into ownCloud/canonical data or an existing output. A local HOME TMPDIR is deliberate: one metadata print redirected to this controller's `/tmp` hit per-user quota (`Errno122`); only that task-owned partial file was removed. This was not evidence that Ada's storage is full.

## Verification and remaining readiness gaps

The19 new synthetic checks and78 existing adapter/policy checks pass (97 total). Python replay is pinned to3.12.13, matching the recorded actual canary environment. The synthetic suite covers frozen identity/split/owner ordering, per-emission RGB drift, exact25 RGB/target/valid/FOV replay failures before output, soft/partial labels, support loss, resource filtering before underlying reads, sidecar closure, encoded-byte drift, observer skipping without renumbering, path escape, disabled admission, full synthetic ZIP/profile/output lifecycle and truthful new35 fingerprints. Existing adapter and policy fixtures also pass unchanged. The runner integration uses a fixture decoder and tiny synthetic source bytes; it is not real RIM coverage.

Run CPU fixtures without installing dependencies:

```sh
TMPDIR=/var/home/tom/.raw35-producer-tmp PYTHONPATH=src \
  /var/home/tom/code/DeepRetinaSegmentation_hpo/.venv/bin/pytest -q \
  tests/test_raw35_producer.py tests/test_adapters_segmentation.py \
  tests/test_adapters_additional.py tests/test_adapters_auxiliary.py \
  tests/test_geometry_policy.py -p no:cacheprovider
```

Only the historical three-TRAIN-photo/seven-annotation canary established actual25 replay for DRIVE/RITE, RIM and UoA. Full21 source closure identities, representative native BIOMISA/HRF/IOSTAR/addition geometry coverage and every retained emission's replay remain unmeasured. The complete producer can fail for review on previously unseen inconsistencies; passing synthetic fixtures does not certify all21 sources or authorize a full build. Native35 trainer admission, common25 probability bridge evaluation and QA/gate contracts remain separate future work; no guards in72/6d were relaxed.

Metadata predicts16,826,751,303B uncompressed uint8 RGB35 and99,920,393,124B per-record float32-target/bool-valid/FOV representation for TRAIN+VAL. Compressed size and full runtime are unknown. There is no three-photo extrapolation and no quality claim.
