# Independent adaptive V3: implementation prepared, not launched

Protocol schema3/kind `adaptive_v3_unified` is a new study/DB/profile/output/queue
family. Existing V1 and finite V2 modes and records retain their behavior. No actual
study or running source was changed. This coordinator package can be deployed from
its own reviewed commit while ML training source remains frozen72fb321, execution
347f78ffd2f4a352c0870faad34327945f7d1c64b865dacb7b7d86e5ca4e0aaa,
training3725ad32a76449b2a3383485cbf14ef050a75f945b4c25e98f4b5b511a332f28.
BF16 benchmark/staging, native-GPU memory/runtime proof and final resource deadline
remain required before root creates executable contracts. No GPU/remote execution
or actual study creation was performed here.

`v3.make_protocol(deadline_utc, control_worker_id, profiles)` builds a draft using
explicit supplied deadline/control/profile bindings. It defaults to24 new configs
(accepted positive upper bound24),8 TPE startup observations,24 EI candidates,
seed42,multivariate/constant_liar. Source/data/geometry/cache are exact72 /2529TRAIN
photos/5700records/21sources/770VAL, manifest17a2/content00d9/policy2591/cache18c3;
all declared profiles require pinned cache with no fallback. Meta cap<=5, own<=3
(default0); root may include a future idle Ada binding before freezing the new DB.
No workers may be added to existing immutable bindings later. Repeated asynchronous
suggestion order and unclean restarts do not imply bitwise TPE history replay.

Actual seed control0 is enqueued and bound to the reviewed control worker first.
It has vascular priority2,mild,base12,batchnorm,lr3e-4,wd1e-5. It consumes a slot and
is protected60 except actual failure/deadline. Six searched parameters:

| Parameter | Bounds |
| --- | --- |
| lr | log1e-4..8e-4 |
| weight_decay | log1e-6..1e-3 |
| vascular_sampling_weight | log1..3 |
| base_channels |12 or16 |
| norm |batch or group |
| augmentation |basic,mild,strong |

Other training settings are exact vascular2 BF16 config: compact_unet/depth4,
384×batch4,60×8192,seed42,raw25,coverage_first,cosine60,loss weights1,no teacher,
no max_steps. Task schema1 stays zero-based trial0000-e003 and carries study_kind,
is_control,candidate_role and actual suggested_params. Reconstruct exact candidate
config from reviewed template+params+profile worker/device override at the worker.

Actual milestone rungs3/9 always promote (unless deadline/failure). First native
ASHA comparison is27,min_resource27,factor3. Control records its native suggestion
but continues. Only actual60 can COMPLETE. Native TPE uses actual completed/pruned
history after startup; failures are not fabricated observations. Same worker/GPU/
plan/config/cache stay pinned across rungs. No job allocation, scheduling, transport
or queue polling is implemented by this package.

## Prespecified six-task LAST objective

Maximize the minimum of actual-LAST ratios: vessel Dice, artery Dice, vein Dice,
disc Dice,cup Dice,conditional AV accuracy, each divided by its actual original
reference value. Reference report exact SHA
182ad346b7bda554ae58e76acbc3f17d99228a30bd19cfa335c89534e9f66836,
reviewed context6ecb69b4029915c3b81ffa0c133c7f540c4da262473418bf4e39d26d40ac9d16.
Reference integer counts are stored in `src/deepretina_optuna/v3_reference.json`;
values are recomputed as2TP/(2TP+FP+FN), conditional correct/scored. The six reference
values are .7701233997938133/.689521829787506/.6972863687090058/
.8948079084718797/.4353989841007063/.748566415513326. A score>=1 means all six
point estimates reach reference; independent quality/generalization is unproven.
No threshold tuning or TEST is part of this search.

Each rung uses current LAST, and COMPLETE uses terminal60 LAST. Per-epoch macro
CSV and macro-selected best.pt remain separate actual provenance. The frozen
producer does not save every epoch's task-balanced optimum; no such history is
invented. Selection/release of full-annotation terminal artifacts and fresh float32
VAL/final reviewed TEST are later separate gates.

Existing receipt identity/budget/scheduler/finite/checkpoint/CSV/source/profile
checks still apply. V3 `checkpoint_qa` additionally requires:

- `kind="v3_rung_evidence"`, `last_checkpoint_sha256==checkpoint_sha256`;
- `last_validation_metrics` from actual loaded LAST and its
  `last_validation_metrics_sha256` using core.digest (sorted compact JSON+newline);
- original `macro_best_score`, `macro_best_epoch`/best_epoch verified against CSV;
- q.metric/r.metric equal derived six-task LAST score; metrics contain fixed.5,
  unified/full_data/VAL770, unique photo IDs, exactly pinned GT/source/record support,
  consistent per-photo/pooled integer counts and derived Dice/conditional accuracy;
- terminal `coverage` actual LAST photo_visits/annotation_visits: exact5700 positive
  eligible record IDs, exact2529 photos each>=60 visits,total491520,5700 headercounts.
  Counts alone cannot pass an altered-ID inventory. Macro-best coverage must be
  disclosed separately; it is not terminal coverage.

Eligibility ID contract comes from manifest17a2's independently pinned TRAIN flags
SHA62a4dd1497b8455ac499952b1191fd7fee1e72f4913f8c2ae492ced088049c17,
cross-checked against original all5700 control coverage SHA
d89114dd3dabdee8b998c62eb5464dfc140040006de0739562dd757eecfaeb4b.
VAL support contract is derived from the exact reviewed original report and matched
actual fresh control-LAST metadata. No GT array decode or model import is used.

## Crash durability and verification

A V3-only journal commits issuance intent before ask, then recovers the same trial,
parameters, binding and lease through pending-task creation. Already drawn params
are retained. If a partial draw changed history past its RNG snapshot, existing
explicit recorded RNG migration is used; original stream recovery is not claimed.
Promotion/terminal decisions are journaled before replacing pending, receipt,
decision or tell. Duplicate original receipts replay the committed decision after
restart, including crashes after terminal tell. V1/V2 paths retain their own rules.

TEMP-DB tests cover unchanged V1/V2,seed/control/caps,24 genuine suggestions,native
TPE sampling after8 completed observations,3/9 protection/27 pruning,all6 params,
partial adaptive draws,issuance/promotion/terminal fault seams,artery-only bottleneck,
reference/source/count/support tampering and exact full annotation identity.
Actual model/data/GPU/study runs are outside these tests. Private Meta worker/SIF/
PBS envelope and LAST-QA adapters must independently match this new contract;
passing coordinator tests alone does not establish staging or cluster readiness.
