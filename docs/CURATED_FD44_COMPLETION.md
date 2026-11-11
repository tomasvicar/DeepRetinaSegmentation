# Independent curated fd44 completion check

This checker checks the original five-arm patch/photometry study's completion
records and checkpoint bytes. The completion check itself performs no inference,
selection or TEST action. The separate [explicit final-eligibility adapter](CURATED_FD44_SELECTION_PROPOSAL.md)
requires the externally pinned original index and source-compatible fresh evidence
for the integrated curated route. Existing `standalone_0b`, `standalone_72`, native
V3 pool and historical completion routes are unchanged. This integration proposal
does not claim an actual final freeze or TEST result.

The accepted training source is `fd44cdfa83d1df9ccc42960df6d1d2018db29900`,
with its original execution/training digests. The three reviewed study contracts
cover exactly raw256, raw384, raw512, clahe384 and lc384. Original parent selection,
helper manifest, full configuration, corpus-v2 manifest/content/geometry, raw
cache and optional immutable photometry cache identities are bound explicitly.
No configuration fields are discarded to fit the current `UnifiedConfig`.

## CPU-only API and CLI

`deepretina.curated_completion.check_curated_completion` takes keyword arguments
`family`, `checkpoint_role`, `original_index`, `expected_index_sha256`, and
`manifest`. It returns immutable snapshots of the selected checkpoint and a
`checked_curated_fd44_completion` proof. Roles are `training_best` and
`fixed_terminal60`; the selected role's own epoch, score, SHA and annotation
coverage are used.

From a checkout with its uv environment:

```sh
CUDA_VISIBLE_DEVICES='' uv run python scripts/check_curated_completion.py \
  --family curated_fd44 --checkpoint-role training_best \
  --original-index /reviewed/inputs.json \
  --expected-index-sha256 REVIEWED_INDEX_SHA256 \
  --manifest /readonly/corpus/manifest.json --output /new/completion-proof.json
```

The output must not already exist. The CLI uses one Torch CPU thread and does
not initialize CUDA, run a model forward, read image/label assets, create a
selection artifact or invoke TEST.

## Original input index

The externally reviewed index has exactly this outer shape:

```json
{
  "schema_version": 1,
  "kind": "original_curated_completion_inputs",
  "files": {
    "best": {"path": "/readonly/best.pt", "sha256": "..."}
  }
}
```

The example entry is abbreviated. All entries below are mandatory, each with
an explicit local materialization path and the SHA256 of its original bytes:

- `best`, `last`, `csv`, `coverage`, `validation`, `run`, `summary`, `marker`:
  the independent HOST milestone's eight artifacts. `marker` is the original
  `search_trial.json` copy.
- `receipt27`, `receipt60`, `task27`, `task60`: genuine curated stage receipts
  and tasks. Both observed training child exits must be zero; E60 resumes the
  same original worker/GPU/configuration after E27.
- `qa60`, `copy`: the original CPU QA and independent HOST copy verification.
  The receipt's HOST overlay must match the original raw QA without retagging;
  all eight copied-byte hashes are recomputed.
- `run_result60`, `runtime60`, `identity60`: original actual training process,
  GPU, source and allocation records.
- `study`, `profile`, `parent`, `helpers`: original authorized study/profile
  and frozen dependency contracts, including all five complete arm configs.
- `terminal`, `cleanup`, `source_attestation`: original curated controller
  records. Cleanup is retained and hashed as evidence; completion does not
  imply that unrelated retained scratch directories were removed.
- `pbs`: an independently captured successful `qstat -x -f -F json JOB_ID`
  observation, with `command`, integer `returncode: 0`, and its selected `PBS`
  record containing `job_state: "F"`, integer `Exit_status: 0`. The allocation
  ID must match `identity60`. Scheduler success and logical worker success
  are checked separately.

No native V3 envelope or standalone supervisor terminal is fabricated. Input
paths may relocate the preserved bytes; historical paths inside original JSON
remain provenance. Each input is snapshotted, hashed and rechecked at the end.
The original task's `input_artifact_shas` binds raw original JSON bytes, using
the original key `parent_tpe`. Its `profile_sha256` and `study_contract_sha256`
instead bind canonical compact sorted JSON plus a newline. Both bindings are
verified separately; pretty original JSON must not be silently replaced by its
canonical serialization.
The manifest and checker source/runtime identity are also checked for changes
during the operation. This is a bounded consistency check, not a guarantee
against future storage changes or proof of previous TEST blindness. The trusted
index hash and original scheduler observation still require external review.

## Recomputed completion requirements

Both checkpoint files are loaded with `weights_only=True` on CPU. After exact
full-config admission, only the known compact UNet architecture is constructed;
strict state loading and finite FP32 state checks require no forward. LAST must
actually be E60/122880 steps, with finite matching Adam moments/steps, scheduler
horizon 60, resume RNG bytes, and all 60 CSV rows. BEST's epoch/score must match
the first maximum in that history. Full corpus metadata supplies the eligible
2529 TRAIN photos and 5700 annotation IDs independently of candidate claims.

LAST and the selected role must each have their own complete positive coverage.
A partial BEST cannot borrow LAST coverage; an explicitly selected complete
LAST can remain valid even if BEST is partial. Original QA role summaries must
agree with the independently loaded role bytes.

## Verification scope of this implementation

At the original independent-checker implementation, 28 focused CPU tests passed in 8.53 seconds. They use tiny constructed weights
and synthetic metadata, with NN forward and CUDA initialization prohibited.
Synthetic trust-anchor substitutions are explicit fixtures, not claims of a
real fd44 completion. Cases cover both roles, all five arms, full-config and
nonraw cache/backend binding, changed task/study/source/receipt, worker-success
versus nonterminal PBS state, distinct raw/canonical dependency hashes, stale
optimizer, wrong horizon, malformed/nonfinite
state, and zero/missing/foreign BEST annotation visits. Its original regression confirmed
that the then-existing final-selection gate rejected `curated_fd44`. The separate
integrated adapter now requires the distinct original index and fresh evidence;
the checker alone still creates no selection or inference permission.

The original raw256 small-record schemas informed this adapter. A subsequent
root-reviewed real CPU re-audit under source `3efbd8a` passed both roles with
externally observed CLI exit 0: BEST52 and LAST60 each independently cover all
2529 photos and 5700 annotations. BEST took 3.275 seconds, LAST 3.414 seconds;
peak RSS was approximately 581032 KiB. See the sanitized
[genuine audit summary](../reports/curated-fd44-completion/genuine-raw256-two-role-audit.json).
Original inputs/source remained unchanged. This is a completion audit, not a
fresh quality measurement, model promotion, selection admission or TEST result.

The same unchanged `3efbd8a` checker subsequently passed genuine **LC384** BEST60
and LAST60, each with actual observed CLI exit 0 and independently complete own
2529-photo/5700-annotation coverage. The full original LC descriptor, backend and
immutable nonraw cache were retained. Runs took 3.203s / 4.224s, with peak RSS
581108 / 581080 KiB. The
[LC audit summary](../reports/curated-fd44-completion/genuine-lc384-two-role-audit.json)
exports only scientific/proof identities and derived counts. This was neither
photometric processing nor a new VAL/quality measurement; the existing gates
remain closed to the new family.
