# Final selection / freeze / TEST audit: source0b and source72

Audited HPO commit `317c3030afbb25dcb23b65090a712c5a92faa7c5`.
**No TEST evaluation, training, NN forward, real GT decoding, selection artifact
creation or implementation change occurred.** CPU probes in
`reports/selection-family-audit-20261111/probes.json` exercise actual rejection
paths and synthetic metadata subgates, not merely string searches.

## What actually blocks the new families

| Stage | Actual behavior | Required action |
|---|---|---|
| `selection.checkpoint_snapshot` → `unified.load_unified` | Current HPO `UnifiedConfig` rejects `vascular_sampling_weight` (0b and72) and independently `mixed_precision` (72). Genuine owned0b BEST59 and72 LAST3 were rejected before model construction. | Minimal source-compatible model/config support with exact saved field values; never drop/filter fields. Stable2139555 already supports vascular sampling; BF16 port903c929 is separately pending review. |
| `selection._context` | No fixed6b commit or pool-QA allowlist. It consumes raw BEST/LAST, CSV, summary and freshly enriched VAL, **not** an external completion QA or original process terminal. | Bind externally pinned typed completion provenance and original execution outcome; loader compatibility alone is insufficient. |
| `completion_evidence` | Requires actual terminal60 and CSV-selected macro BEST, but permits an annotation-visit subset, total photo visits <=budget, and never checks selected BEST annotation coverage. | Require manifest-derived exact eligible photo/annotation ID sets, full terminal budget and **own selected-weight** coverage. Preserve BEST versus LAST roles. |
| `completion_roles.verify_terminal60` | Truly fixed to source6b/executioncb5c/training9075, `trusted_pool_completion_qa`,2529photos/5700records. | This is the existing fixed-LAST **VAL comparison** adapter, not the final selection engine. Keep its historical meaning; add an explicit new-family adapter rather than aliasing0b/72 to6b. |
| Enriched VAL source lease | `_context` requires evidence.source equal the actual evaluation package/runtime fingerprint plus the checkpoint's original training fingerprint. | Freeze a compatible evaluation toolchain first, then produce new enriched VAL under it. Never rewrite old source hashes or thin reports. Training and evaluation commits may differ honestly. |
| Frozen selection | `create_selection` creates schema1 with decision timestamp and full source/config/data/protocol/completion/VAL pins; `check_selection` rebuilds the same context. | Extend explicit provenance/role schema together with its checker. Arbitrary extra fields currently fail exact-field/context checks. No filename-only approval. |
| Explicit execution | General unified CLI rejects `--split test` before loading weights. `heldout.evaluate_reviewed_heldout` rejects missing explicit stage2 request before source/data reads. | Preserve both boundaries; stage1 receipt stays `test_evaluation_enabled=false`. Nothing in this audit authorizes stage2. |
| `compare_reviewed_test.py` | Metadata/count-only consumer. It has no6b training-source/QA allowlist and accepts additional config fields when all hashes/bindings agree. It expects known schema1 wrappers, completed60/8192 and macro-BEST completion semantics. | No broad source allowlist is needed here. Update only an explicitly revised selection/report schema or checkpoint-role semantics later; never use it as an execution gate. |

## Executed scope probes

1. Actual `checkpoint_snapshot` on bundled0b BEST81113 and actual72 partial LAST723da
   rejects unsupported vascular configuration. An independent in-memory config
   probe shows that adding only vascular support would still leave BF16 unsupported.
   No original config/checkpoint was modified; no NN model was constructed/forwarded.
2. Public evaluate with `split=test` and stage2 with execution flag false reject
   before nonexistent input paths can be read.
3. Existing **fabricated** two-photo metadata fixture passes the comparison's
   checked-candidate function with0b or72 training fingerprints and their new config
   fields. No real TEST report, label or evaluator was involved.
4. A synthetic completion subgate with60 contiguous CSV rows, terminal optimizer
   step122880 and consistent headers is accepted despite **zero** annotation
   visits, only **one** total photo visit and no selected BEST coverage supplied.
   This is a demonstrated weakness of the completion subgate, not a claim that a
   complete selection artifact was created or any real invalid model was tested.
   A synthetic terminal epoch3 is correctly rejected.

The raw completion check also does not perform the standalone verifier's complete
AdamW parameter/moment/RNG binding. The standalone verifier should remain the
source of that stronger typed provenance, not be replaced by a casual boolean.

## Minimal safe adaptation

Use an isolated evaluation branch/toolchain, not a wholesale retrospective HPO
merge into the production app. Preserve exact dataclass field order and defaults,
original checkpoint bytes/config, FP32 stored state and FP32/no-autocast inference.
A BF16 **training** configuration does not authorize BF16 stored weights or BF16
held-out inference. Preserve finite/architecture/lineage checks, immutable safe
byte snapshots and operation-wide source/canonical leases.

Add an explicit family-aware freeze-eligibility check before artifact creation:

- **6b pool:** retain the genuine `trusted_pool_completion_qa` family/source contract.
- **0b /72 standalone:** require genuine `standalone_unified_completion_qa`, original
  bounded supervisor terminal and reviewed protocol, all externally byte-pinned.
  The existing verifier4b568b is source-generic through reviewed source pins;
  it still binds the fixed corpusv2/budget and requires imports from the actual
  reviewed training checkout. It is not a replacement TEST executor.
- **72 V3 pool:** its real `v3_rung_evidence`/worker/task/lease/PBS provenance is a
  different family. Do not relabel it as standalone. Native V3 validation enforces
  all5700annotations/all2529photos at rung60; partial3/9/27 is not completion.
  A later genuinely standalone72 retraining may use the standalone verifier.

For every eligible selected weight, verify manifest-derived actual TRAIN IDs,
5700positive annotation visits and2529photographs under this corpus profile,
full60/491520 run budget, optimizer/scheduler/RNG consistency and role-specific
coverage. Terminal LAST coverage does not transfer to earlier BEST. Current0b
BEST59 happens to have all5700; future earlier BEST may not. Dynamic corpus
counts must come from the trusted manifest, not a candidate self-declaration.

The current final gate is **training macro-BEST only**: `completion_evidence`
requires selected score equal CSV max. Passing LAST60 as “BEST” fails whenever
last is not the macro maximum. V3 trial ranking instead uses LAST worst6ratio;
its saved BEST remains macro-selected. Freeze an explicit checkpoint-role/objective
policy: either genuinely select macro-BEST with its own coverage and fresh VAL,
or add a reviewed terminal-LAST role that retains the true BEST separately.
Never copy scores/hashes between roles. Bind the actual study/decision provenance
when selecting a TPE winner; raw CSV maximum alone does not record that choice.

Keep derived BN artifacts separate: their stale training-selection scores and
optimizer state were intentionally removed. They are research/predict-only weights,
not ordinary original-training completion evidence; do not fabricate missing scores.

## Correct final sequence and original reference

1. Finish training and preserve original immutable terminal/protocol/raw artifacts;
   verify the appropriate genuine family and each selected role's own coverage.
2. Freeze the compatible evaluation code/environment, objective/role and inference
   threshold/tiling policy; produce freshly enriched **VAL only** for exact weights.
   Stage1 selection checks decode VAL support to verify counts; “metadata gate”
   means no NN forward there, not “no VAL GT is read”. This audit did not call it.
3. Record/recheck selection with external byte pins and explicit decision timestamp.
   Unsigned reviewer strings/hashes do not authenticate a person or prove past blindness.
4. Freeze the original reference policy against that exact unified selection SHA.
   `evaluate_original_frozen.freeze` requires its original VAL source/policy identity
   to equal its **own** current runtime. Use its own preserved frozen source/runtime;
   do not retag old original VAL to the new unified toolchain. Its saved-count freeze
   reads metadata/counts, not a new NN prediction. Comparison permits independently
   pinned source identities for the two pipelines.
5. Only a separately authorized, explicit reviewed stage2 invocation may decode real
   TEST GT or run inference. It rechecks selection before CUDA/TEST and evaluates
   the checked in-memory checkpoint bytes with no protocol overrides, preserving
   source/canonical checks through report emission. A VAL770 bridge does not certify
   TEST856/new official27. Keep the generic test CLI blocked.
6. After actual authorized execution, the metadata comparator validates policy,
   selection, chronology, canonical inventory, roles and paired per-photo support.
   It performs no model inference or GT-array decoding itself.

Secondary hardening: `selection.data_identity` presently checks source/photo aliases,
while the final comparison's inventory also checks `split_group_id`. Preserve known
patient/group separation in the eligibility layer; do not infer complete patient
independence from opaque identifiers. No claim here substitutes for the existing
corpus producer audit or changes the fixed train/val/test memberships.
