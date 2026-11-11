# V3 adaptive HPO: bounded implementation proposal, not an approved launch

Assessment against actual code,2026-11-11. New isolated full worktree/branch starts
at72fb321. No running code, DB, profile, worker, source, GPU or remote host was
changed/queried. Existing V2 deadline07/source6b and independent Ada source0b remain
untouched. Public ML source stays72fb321 (training3725ad32…, execution347f78ff…).
Coordinator implementation can have its own later commit; it need not modify that
frozen training checkout. Actual BF16 speed/quality and five Meta slots are unproven.

## What already exists, and why this needs an explicit new mode

The older coordinator checkoutc6acd7c is V1 adaptive TPE, but its validation pins
3157/2221photos/17sources, mandatory imported baseline, macro objective and E3 ASHA.
It cannot be reused by changing JSON alone. Source72 includes newer V2 core/v2
files byte-identical to the running frozen V2 copy (core91e11b9c…, v2a0e8df7e…).
That copy has seven fixed queued candidates, no adaptive extras, protected control0,
corpus/cache/policy validation and finite-only transition journals. Keep both modes.

Reuse the existing single writer/localSQLite lock, Optuna5.0.0 TPE sampler, two
persisted RNG streams and explicit interrupted-history migration, GPU/profile caps,
same-worker/physical-GPU rung promotion, lease tokens, immutable protocol/binding
hashes, and actual checkpoint/CSV receipts. Add `study_kind="adaptive_v3_unified"`,
protocol schema3, new DB/study/output/queue roots. Trial numbering stays zero-based:
trial0=`trial0000`, task=`trial0000-e003`; never retag existing records.

## Proposed search and resource policy (root review required)

| Parameter | Proposed bounds | Seed control0 |
| --- | --- | --- |
| lr | float log[1e-4,8e-4] | 3e-4 |
| weight_decay | float log[1e-6,1e-3] | 1e-5 |
| vascular_sampling_weight | float log[1,3] | 2.0 |
| base_channels | categorical[12,16] | 12 |
| norm | categorical[batch,group] | batch |
| augmentation | categorical[basic,mild,strong] | mild |

Fix compact_unet,depth4,patch384,batch4,raw25,seed42,coverage_first,
8192samples/epoch,cosine60,all existing loss weights1,foreground_probability.5,
no teacher/initialization,max_steps=None,`mixed_precision="bf16"`. Cache18c3 is
required initially; all workers need native BF16, supported memory and verified
source/data/runtime/container bindings. No fallback to FP32/older GPU on failure.
Worker count/thread settings remain explicitly bound per approved profile.

Suggest initial max24 new configurations (hard cap<=48), sampler startup8 finished
observations,24 EI candidates,seed42,multivariate=True,constant_liar=True. These are
review parameters, not an authorization. Startup is genuinely random exploration;
with5 simultaneous workers, several waves may precede usable TPE adaptation.
Record asynchronous order; seed42 does not guarantee identical suggestions after
history changes. No historical V1/V2 trial is imported as same-source/control.
Actual fresh control0 is enqueued first, protected through60, and consumes one
of the24 slots. Fix its approved worker before DB creation, without requiring Ada
or interrupting its existing protected run. Other slots can issue once control0
has been durably bound; they need not wait for its full completion.

Retain reporting/resume milestones3/9/27/60. E3 and E9 always promote unless actual
failure/deadline; first native SHA comparison at27 (`min_resource=27`,factor3).
Control0 always60 except failure/deadline; register/log its native prune suggestion.
All completed candidates require actual60; pruned histories are resource-labelled,
never fabricated terminal values. Initial Meta cap<=5, own cap0 unless root approves
other idle BF16-native bindings before the new DB. Each allocation must accommodate
measured full60+VAL/cache/setup/cleanup margin because current promotions pin the
same worker/physical GPU. No walltime, throughput or deadline is assumed from Ada;
new Meta staging/runtime/memory/benchmark evidence must precede an executable plan.

## Objective: preserve72 training, prevent cup/disc hiding vessel regression

Concrete preferred scalar for a separately pinned original-reference VAL report:
`min(vessel_dice/V0, conditional_AV/A0, disc_dice/D0, cup_dice/C0)` at fixed.5.
This maximizes the worst task recovery ratio; >1 means all four pooled task values
exceed that same pinned reference. Reference values must be recomputed from actual
report counts, with positive denominators, exact canonical VAL770 inventory/policy
and original-source provenance. Do not hardcode rounded baseline numbers or equate
original and unified package source identities. Report all task values, macro,
per-source supports/counts and deltas; the scalar alone is not a quality claim.
Tie handling stays ordinary scalar TPE; task/source details remain available for
review, not a hidden post-hoc objective. Root must approve this formula/reference.

Use the **actual last checkpoint at each rung** for its current task ratio; actual
terminal60 score is the completed-trial objective. Best completed configuration is
selected by this prespecified terminal scalar, not max early-checkpoint performance.
This is the smallest honest path retaining frozen72. Its CSV only records macro/
mean/selection scores, and best.pt is selected by macro. It cannot prove a true
per-epoch best task-balanced history. Both last.pt and best.pt contain their actual
validation_metrics, but last gives only the current rung. If root instead requires
best task-balanced checkpoint selection over every epoch, stop and create/review a
new training producer/source; do not relabel macro best.pt or invent missing epochs.

Add typed V3 rung evidence: immutable last bytes+full validation_metrics count
provenance, source/data/plan/config/profile pins, recomputed four-task scalar and
reference/formula hashes. Bind separately retained macro-selected best bytes/epoch/
CSV history. No GT decode or forward is needed by the coordinator. Macro CSV/epoch/
step/schedule checks remain mandatory but no longer assert the Optuna scalar equals
macro. At60 require all5700 exact eligible TRAIN annotation IDs positive, all2529
photos/minvisits60,491520delivered,finite float32 stored weights/Adam,60-epoch cosine,
and21source inventory. Counts alone cannot prove exact eligible IDs. Disclose
macro-best coverage separately; only full-annotation terminal60 is a release candidate.
No TEST inside HPO; final selection/fresh float32 VAL/reviewed TEST remain separate.

## Minimal files and validation before any new DB

- New `tools/optuna_coordinator/src/deepretina_optuna/v3.py`: explicit mode/schema,
  source/corpus/config/search/objective/reference validators; no ML imports.
- `core.py`: isolated V3 dispatch, generic parameter→exactconfig mapping (geometry
  fixed, no mandatory V1geometry parameter), seed-control issuance, startup settings,
  min27 pruning, typed rung-objective verification and terminal annotation evidence.
  Keep V1/V2 branches and their receipts unchanged. Task carries study_kind,
  is_control,candidate_role(seed_control/adaptive_candidate),suggested_params;
  reconstruct exact config from template+params+profile overrides at the worker.
- V3-only issuance/report transition journal, reusing the finite pattern. The old
  adaptive path marks RUNNING before complete params/meta/pending; it can strand
  a lease, and promotion can replace pending before receipt/decision commit.
  Journal before ask/parameterization; recover the same trial/binding and stored
  parameters, explicitly use existing RNG migration if an interrupted partial draw
  cannot recover the original streams. Never silently seed42/replay-claim or alter
  already-issued tasks. Control's fixed params are known before issuance.
- New V3 tests/protocol draft/docs. Public worker/QA adapter and private envelope/
  Meta PBS adapter need corresponding new-kind support, new source/runtime/SIF/
  canonical/cache pins, native-BF16 admission and independent deadline; coordinated
  with private readiness agent. Existing finite-seven checks must not be bypassed.

Tests: default V1/V2 suites unchanged; TEMP DB only for V3 exact seed/control,
real suggestions beyond seven and post-startup adaptation, all six parameter bounds,
exact config/profile/cache reconstruction, cap5 distinct GPU leases, duplicate/
changed receipts, rung3/9 nonpruning and27 native decisions, control60, full-ID
terminal rejection, per-task count/reference/formula tampering, and restart/fault
injection at every ask/parameter/meta/pending/receipt/decision/tell seam. Show that
old records/tasks are byte-identical. No training/remote execution is required for
these contracts. This is a bounded extension with meaningful crash/metric work,
not a JSON-only launch patch; stage implementation/review before allocation.
