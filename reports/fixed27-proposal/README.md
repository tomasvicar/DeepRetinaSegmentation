# Separate fixed-27 TPE proposal — plan before outcomes

The main V3 study, its startup 8, protocol, trial states, database and sampler stay
unchanged. A separate **fixed-fidelity 27 proposal study** can use an actual
completed epoch 27 evaluation even while its original parent continues toward 60.
It must not call that parent a completed 60 trial. Every imported native Optuna
trial is COMPLETE only for the newly defined fixed-27 measurement problem, with
the original parent ID/status/source/node/checkpoint provenance in user attributes.

**No actual observation or proposal has been consumed/generated here.** Tests
use explicitly synthetic temporary metadata/SQLite only, never model/GT arrays.

## Revised pre-outcome decision rule

Root changed the original wait-for-eight / 12:30 plan before any E27 outcome was inspected:
operator runtime forecasts make eight timely observations unlikely. Generate
exactly one proposal **not before 10:35 UTC, only after own Ada is genuinely free,
and with at least four unique qualified configurations**. Prefer eight when
already available; do not wait or select observations by their score. Import all
available qualified observations at decision time, including promoted control and
parents still RUNNING 60. The complete inventory and Ada-free observation must be
fresh (at most five minutes old) and externally SHA-pinned. The helper does not
query a GPU or discover omitted parent results; completeness/free-device status
is the operator's explicit attestation.

The new native `TPESampler` has seed 42, startup 4, constant_liar=True,
multivariate=True and 24 EI candidates. All six source 8777 ranges are preserved,
including both widths 12 and 16. Four or five observations across six dimensions
are statistically thin; this is not a broad optimum claim. The later fresh
full 60-epoch candidate can run on own hardware toward 18:30 UTC; the parent's Meta 14:30
curfew does not shrink the proposal space or shorten its scheduler.

## Strict original-input interface

Provide a local JSON manifest and its **external byte SHA256**:

```json
{
  "schema_version": 1,
  "kind": "fixed27_proposal_inputs",
  "inventory_complete": true,
  "observed_at_utc": "2026-11-11T10:35:00Z",
  "own_ada_free": true,
  "available_rung27_count": 4,
  "observations": [
    {
      "task": {"path": "task0-e027.json", "sha256": "original task SHA"},
      "receipt": {"path": "result0-e027.json", "sha256": "original COMPLETE rung receipt SHA"},
      "parent_status": "RUNNING"
    }
  ]
}
```

The illustrated list is abbreviated: the actual count must equal the list length
and contain at least four distinct qualified configurations. Files remain their
original JSON bytes; do not project/retag E3/E9, macro-BEST or full 60-epoch evidence.
Parent statuses are current original study states, not inferred from rung status.

Each original typed task/COMPLETE receipt must bind source 72/training 3725,
protocol 053f05, exact native V3 config/plan, epoch 27, 55,296 steps, 221,184 samples and
Tmax 60. Native 8777 `validate_rung` verifies the actual LAST 770-photo six-ratio
statistics and fixed 0.5 GT/source support; native CSV validation checks contiguous
27 rows and macro provenance without using macro-BEST as the observation value.
The original host-verified independent milestone flag, destination, readonly-copy
receipt and LAST/BEST/CSV SHA roles are required. No checkpoint tensor, image,
GT mask or active database is opened by this helper.

Exact duplicate parameter configurations retain the earliest parent trial number,
without comparing their outcome. All origins remain in the proposal artifact.
Mismatched source/fidelity, missing metrics/proof, stale/incomplete inventory,
fewer than four unique configs, busy Ada or early time fails before any new
SQLite study is created. No production CLI bypass for the clock is provided.

## Native proposal only

After root review and the decision gate, use the existing Optuna 5 environment
and explicit frozen 8777 source; for example, with uv in an isolated checkout:

```bash
uv run --frozen --project /path/to/frozen8777/tools/optuna_coordinator \
  python /path/to/hpo/scripts/propose_fixed27_tpe.py \
  --core-source /path/to/frozen8777/tools/optuna_coordinator/src \
  --input-manifest /path/to/reviewed-inputs.json \
  --input-sha256 EXTERNAL_MANIFEST_SHA \
  --output /path/to/new-local-proposal
```

The new output must not exist and its parent must be on a local filesystem,
not NFS/ownCloud. The helper checks genuine clean core 8777/contract 332a, imports
all native `create_trial(COMPLETE,value27,params,distributions,user_attrs)` records,
then calls native `Study.ask(fixed_distributions=...)` once with all six parameters.
There are no manual edits/resampling of the returned parameters. Startup 4 is
already satisfied by completed fixed-27 observations; a test rejects any fallback
to RandomSampler startup. This is real TPE in a separate study, not a claim that
the original running V3 has passed its own startup 8.

The output contains the input manifest, new SQLite, native sampler-state hash and
trusted local pickle, exact proposed params and native full 60-epoch config/plan. The
proposal itself stays RUNNING and explicitly unevaluated in this study. A later
full 60-epoch result must not be silently imported as a fixed-27 measurement. Actual
training/submission is a separate root action: use a new output/lineage and the
full 60-epoch source 72 plan, not a 27-epoch replacement schedule. All 5,700 annotation
contribution, fresh VAL, final freeze and heldout TEST remain mandatory before a
final-model claim. Do not publish private operator paths/nodes from raw input
provenance without an appropriate public summary.
