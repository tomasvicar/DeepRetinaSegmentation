# Local single-writer Optuna5 control

This isolated uv project contains Optuna5 only, with no Torch, model inference,
SSH endpoints, job submission or worker-side Optuna. ML source3157/8c is unchanged.
SQLite must live on local Pascal storage, never NFS or ownCloud. A lifetime flock
allows one coordinator writer; methods also serialize threads. Root handles
transport, GPU readiness and stopping owned workers at the deadline.

```bash
uv sync --project tools/optuna_coordinator --frozen
uv run --project tools/optuna_coordinator deepretina-optuna --db /local/state/night.sqlite3 \
  --protocol tools/optuna_coordinator/protocol.json --profiles /private/approved-bindings.json \
  --baseline-csv /verified/metrics.csv --baseline-qa /verified/e060-qa.json
```

The process accepts JSON lines: `{"op":"ask","worker_id":"meta1","gpu_uuid":"GPU-..."}`,
`{"op":"report","result":{...}}`, or `{"op":"status"}`. Responses use `ok` and
`answer`. Alternatively import `Coordinator(db_path, protocol, profiles)` directly.
Profiles bind worker IDs to `{profile,profile_sha256,output_root,gpu_uuid?}`;
optional `config_fixed` must contain the matching worker count. All paths are
caller-provided approved output roots, not connection endpoints.

Tasks use the shared `trial`/`lease`/`trialNNNN-eNNN` schema and inline immutable
plan. Hashes are SHA256 of sorted compact JSON **including a final newline**.
Promotions return `{action:"continue",next_task:...}` and preserve plan/config,
worker/physicalGPU/cache/profile/output. Each newly generated stage receives a
deterministic per-rung lease token (`trial lease` + `-eNNN`) so exclusive stage
logs do not collide. Stored pending envelopes, including old unsuffixed E3
tokens, remain verbatim and their exact receipts remain valid. Decision tokens
identify the completed stage; retry/restart preserves each stage token. Workers return audit's
`checkpoint_qa` schema plus actual CSV text; core checks complete contiguous
learning curves, best-so-far metric, finite flags, coverage, budgets and pins.
Outer worker `COMPLETE` means stage completion, not an Optuna final model.
Only actual epoch60 plus completed-budget QA can receive Optuna COMPLETE.
Stopped partial stages are PRUNED; invalid/infrastructure worker failures are
FAIL, never a final model. Duplicate exact receipts are idempotent.

The predeclared search/runtimes are in `protocol.json`. Native TPE uses seed42,
constant_liar=True and multivariate=True. Native successive halving reports at
3/9/27/60 with min_resource3/reduction3; the final60 budget is enforced here,
because ASHA's next native pruning boundary after27 would be81. Up to48 new
configurations,5 Meta and3 own GPUs, deadline07:00 UTC;2–3 full survivors is a
soft target. No historical teacher/pretraining or TEST is involved.

Import only the independently verified actual3157 full60 baseline CSV/QA:
manual trial0 is excluded from48 new configurations. Its objective is the original
fixed0.5 training-VAL best score, not the later threshold-tuned score. The weight2
seed genotype is enqueued for the first Meta ask. Older87 pilots are not imported.
Checkpoint/tensor QA remains the reviewed worker's responsibility; this lightweight
coordinator checks receipts/artifact hashes, not hardware attestation. Profiles
and asynchronous completion order confound causal comparisons. Seed42 does not
promise identical sampling after coordinator restart or changed completion order.
Leases/tasks persist in Optuna user attributes and recover through documented RDB
IDs; sampler process RNG is restarted and that limitation is recorded. Native
ask/tell has no automatic heartbeat: transport must report losses explicitly.

Official APIs: [ask/tell](https://optuna.readthedocs.io/en/stable/tutorial/20_recipes/009_ask_and_tell.html),
[successive halving](https://optuna.readthedocs.io/en/stable/reference/generated/optuna.pruners.SuccessiveHalvingPruner.html),
[RDB storage](https://optuna.readthedocs.io/en/stable/reference/generated/optuna.storages.RDBStorage.html).
The documented Trial constructor is used only to reconnect stored trial IDs on
restart; ordinary creation/completion uses Study.ask/Trial.report/should_prune/tell.

Seven CPU/SQLite fixture tests pass in2.07s, including native ASHA peer pruning,
all four rungs, rejection of facade completion, bounds, affinity, config cap,
writer locking and restart/idempotence. No production study or GPU work was
started by these tests. Actual baseline fixtures are metadata only.
