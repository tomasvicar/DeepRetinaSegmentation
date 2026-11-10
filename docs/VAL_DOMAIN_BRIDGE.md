# Explicit unchanged-VAL domain bridge

A changed TRAIN/TEST corpus changes the global manifest and canonical identity, even when its entire validation domain is preserved. Do not edit old measurement headers to make them appear measured on the new corpus. `scripts/bridge_val_domain.py` creates an explicit, versioned **VAL770-only** proof and a new typed reference wrapper. Original measurements remain unchanged inside that wrapper, including their original global pins. Generic `compare_reports` retains its exact manifest guard and does not implicitly unwrap this artifact.

## Actual v1→v2 result

The [actual encoded-byte verification](../reports/data/val-domain-bridge-v1-v2/verification.json) passed on Ada CPU1/nice10 in4.36seconds without importing Torch, model forward, asset decoding, GT scoring or hashing TEST assets:

- 770 validation photographs,993 annotation rows and612 known split groups match exactly.
- 3303 referenced RGB/mask/FOV asset paths match byte-for-byte; the inventory sums1,417,162,870 encoded bytes including aliases.
- Parent manifest `529edba7743eb5b9175da429c2124d53f48c419b090a284b5650fd68f1fdcf69`, canonical `61032be218c623a910c0f3f5e2df3bfdc58e978fe494433a4e6f18d55e41693e`.
- Target manifest `17a2277c39cb37b7eee1b2063cfd13f2cf4adf4ebdf86cf449a1f51367c5a767`, canonical `00d9ef19d262eef74f0ca1350fec260c687dff5cc3a53f8911cbe3403e821e52`.
- Strong measured clipped-disc0 original report SHA `528233fe50876b15f127f8567fc43d7b8a34a8713cfceff54f02956a0a0bfa2c`.

The canonical [bridge](../reports/data/val-domain-bridge-v1-v2/bridge.json) SHA is **`016ad76e7472cd3e64f846a3f7aeac3369e60732fd9a59c70e46fc082ec86ddd`**. The new [reference wrapper](../reports/data/val-domain-bridge-v1-v2/reference-for-target.json) SHA is **`547a95afb94a6714c028907e575ed8523b57bd31d60c43614ecf49b1232dff73`**. Resolving the wrapper against that explicit bridge and the current target assets succeeded; the measured per-photo body remains identical. This produces no new model score or v2 candidate comparison.

The producer's complete schema2 audit supplies the independently pinned global canonical identities. The bridge rehashes **only VAL assets**, not all TRAIN/TEST assets; it binds the expected global audit declarations and completion markers rather than claiming an independent whole-corpus audit. It does not bridge the856 TEST photographs or their27 newly converted IDRiD photos. The [target training-domain metadata check](../reports/data/val-domain-bridge-v1-v2/target-training-domain.json) separately confirms2529 eligible TRAIN photo IDs using the frozen reader's filters, without opening those assets.

## Verification and trust boundaries

The verifier compares all VAL row values and types, record IDs, observer/source/group/preparation metadata and photo-index geometry/FOV/source metadata. No extra/missing VAL row or photo is admitted. Both contexts must have matching schema2 audit/manifest completion declarations and caller-supplied expected canonical pins. Source/photo/group identities cannot cross partitions; explicit official TEST and unknown-patient TRAIN-only membership are also checked from metadata. Asset paths must stay under their resolved corpus roots and cannot traverse symlinks. Streaming hashes and before/after file identities, including ctime, detect replacement or mutation during the operation. Metadata is checked unchanged after the hash pass.

The wrapper family is `val_domain_recontextualized_original_reference`. It retains the complete measured reference body, original artifact SHA and separate parent/target domain declarations. Only explicit `compare_completed_val.py` opt-in with **all three** of bridge file, externally trusted bridge SHA and original measured reference file can resolve it. The resolver checks canonical bridge bytes, reconstructs the exact wrapper, verifies the original raw/semantic hashes, and rehashes the target VAL assets. A renamed file, wrapper's own hash, altered count body or self-declared context cannot replace those checks. Once verified, only an internal temporary comparison view receives the target global pins; normal exact-pair validation still runs. The output distinguishes original measurement, wrapper and derived-view hashes.

A trusted SHA means the caller deliberately accepts a reviewed proof artifact. It is not proof of prior TEST non-exposure, original-model training independence, clinical annotation correctness, unknown patient independence, or future filesystem immutability. The old reference's historical exposure, calibrated preprocessing and backend qualifications remain intact. No GT array is decoded to create the bridge; exact encoded equality and metadata support the domain claim.

## Commands

Create a new artifact directory outside both corpora; existing outputs are refused:

```bash
uv run python scripts/bridge_val_domain.py \
  --parent /ABS/immutable-v1 --target /ABS/immutable-v2 \
  --expected-parent-content 61032be218c623a910c0f3f5e2df3bfdc58e978fe494433a4e6f18d55e41693e \
  --expected-target-content 00d9ef19d262eef74f0ca1350fec260c687dff5cc3a53f8911cbe3403e821e52 \
  --original-reference reports/cup-containment-20261110/complete/fresh-cup-clipped-by-disc0-in-disc0-reference-val.json \
  --output /ABS/new-val-bridge
```

There is no TEST/split override. The production path requires770 VAL photos. Small synthetic fixtures exercise the lower-level functions with a smaller explicitly supplied count; they cannot resolve as production VAL770.

For a **future actual completed v2 model**, use its enriched VAL evidence and its own terminal60 QA, not retagged v1 candidate evidence:

```bash
uv run python scripts/compare_completed_val.py \
  --reference reports/data/val-domain-bridge-v1-v2/reference-for-target.json \
  --val-bridge reports/data/val-domain-bridge-v1-v2/bridge.json \
  --trusted-val-bridge-sha256 016ad76e7472cd3e64f846a3f7aeac3369e60732fd9a59c70e46fc082ec86ddd \
  --original-measured-reference reports/cup-containment-20261110/complete/fresh-cup-clipped-by-disc0-in-disc0-reference-val.json \
  --candidate /ABS/actual-v2-completed-val-evidence.json \
  --completion-qa /ABS/actual-v2-terminal60-qa.json \
  --manifest /ABS/immutable-v2/manifest.json --output /ABS/new-v2-comparison.json
```

Completion checking derives eligible TRAIN photo identities from the bound target manifest using task/split/usable/exclusion filters. Both candidate evidence and terminal QA must agree with that independently derived count; the ID-set hash is retained. A subset run cannot satisfy the check merely by self-consistently declaring a smaller training set. Legacy v1 default completion checks remain2221, and standard same-corpus comparisons need no bridge.

Sixty-four synthetic CPU-only tests passed in0.88seconds, covering changed GT/RGB/FOV bytes, geometry/observer/group/photo/membership, numeric metadata types, symlink escape, ineligible original reference, canonical/global pins, modified wrapper counts, filename-hash substitution, mid-operation rewrites, no TEST-asset hashing, explicit opt-in, generic rejection of wrappers, and subset completion claims. Actual v2 preparation and bridge verification are separate receipts; no v2 quality promotion follows from either.
