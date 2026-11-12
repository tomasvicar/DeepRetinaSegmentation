# Controlled vessel loss2: actual LAST60 fresh VAL

**The prespecified dense-vessel endpoint improved against its matched control,
with adverse tradeoffs. This is not an all-head improvement or model promotion.**
Source72 scratch full60/8192/b4/384/BF16/seed42: only scientific change from first
native TPE5 was `vessel_loss_weight:1→2`; `vessel_pos_weight` stayed1. Fresh LAST60
was evaluated with the exact same source72/b269, raw25, NONE/FP32/512/64/.5, data
and runtime context. Full configuration/source/inference equality and the sole
loss delta were checked directly. BEST58 was not substituted for LAST60.

Original full60 terminal/generic4bQA/outer all returned0. LAST60 owns all2529TRAIN
photos/all5700annotations (minimum62visits); BEST58 also owns5700 but remains
unselected here. Actual fresh VAL child0 took92.716s; supervisor93.082s and outer
observed0 completed05:21:54.754UTC. Supervisor's own planned_returncode is **not**
an observed exit; the independent outer receipt is the witness. No TEST.

| Pooled metric | TPE5 LAST60 control | Loss2 LAST60 | Difference / paired95%CI |
|---|---:|---:|---|
| **Dense vessel (primary)** | .7385115361 | **.7440593871** | **+.0055478510 [+.0034889816,+.0079735800]** |
| All vessel | .7608027722 | .7673374971 | +.0065347250 [+.0044225165,+.0089833214] |
| Artery | .7326536676 | .7355705327 | +.0029168651 [−.0027957811,+.0087829442] |
| Vein | .7907396440 | .7924120796 | +.0016724356 [−.0028330723,+.0066152340] |
| Disc | .9353376016 | .9337082537 | −.0016293479 [−.0119577304,+.0051177626] |
| Cup | .8748724200 | .8603939413 | −.0144784787 [−.0473650298,+.0063496856] |
| Conditional AV accuracy | .8217549968 | .8223463266 | +.0005913298 [−.0046775857,+.0061533994] |
| Source-balanced five-head macro | .8088325417 | .8081398927 | −.0006926490 [−.0074891296,+.0068655592] |

Dense domain is exactly87VALphotos/55known split groups/8sources, excluding only
AVRDB. It is **not** a bootstrap over all612groups or a claim of proven independent
patients. Source-stratified known-group paired bootstrap:2000draws/seed42, unchanged
reviewed method. All-head/source comparisons cover770photos/612known groups.
AVRDB20positive-only photos are reported separately: vessel Dice.9814074510→
.9938068572; structural FP0/precision1 does not establish dense specificity.

Loss2 remains below original182 on dense vessel: .7440593871 vs .7613266071,
Δ−.0172672200 CI[−.0256150258,−.0101800088]. Its all-vessel point.7673374971 is
also below original.7701233998, Δ−.0027859027 CI[−.0110659944,+.0049685350]; that
interval crossing0 does not demonstrate equivalence. Original pipeline exposure/
preprocessing differs, so its contrast is a pipeline comparison, not causal loss
attribution. Original reports/gates/objectives remain unchanged.

Adverse findings must accompany the primary gain. Cup **source-macro** Dice fell
−.0198613411 CI[−.0464035386,−.0001809912], while pooled cup interval crosses0.
Source intervals below0 versus control include AVRDB artery, HRF vein and HRF
conditionalAV; all point-negative source/head/AV entries are retained in
`actual-results/adverse-source-results.json`. No multiplicity adjustment, clinical
margin/noninferiority claim, sparse-TRAIN causal explanation, universal quality
gain, default/weight change or TEST result follows from this single-seed ablation.

## Evidence and reproducible count-only command

- Original fresh report SHA `03928460e391ef87df5ad4faf6e3005be1e5190bf87d766c6260c2803ee3fb79`.
- LAST SHA `1cd10ac6faa9f4096e5a9e0cd045deceefa40015a63db883a9add1e25aba5286`.
- Matched firstTPE5 native LAST report SHA `7570255b5eac5855d6bd3a1068ff12effbc66c7f05c6699ba26f82fc25c022be`.
- Original182 VAL SHA `182ad346b7bda554ae58e76acbc3f17d99228a30bd19cfa335c89534e9f66836`.

`actual-results/` preserves full-precision paired/all-head/source/AV intervals,
dense/all/AVRDB pooled counts, per-source dense counts and full configuration/
source/inference provenance. The aggregate export removes per-photo arrays and
only physical-UUID runtime fields, recording original whole-byte SHA; it is a
**derived scientific count view, not a replacement evaluation/eligibility gate**.
No confusion counts/source attribution/config were changed. Original770photo
reports and original operational receipts remain immutable outside this export.
`completion-and-role-provenance.json` distinguishes actual outer/child0, own role
coverage and the planned supervisor return. No private paths/UUID are exported.

`compare_when_complete.py` never imports Torch or runs a model; only JSON/CSV-free
manifest metadata and saved confusion counts. It reuses unchanged comparison
SHA8fcd8711…fe4e and dense helper13bc973a…b304. The helper's inherited generic
cross-pipeline `scope` string is kept verbatim; the primary contrast's stricter
matched LAST60/source/config checks above apply here. This does not imply a
role/TTA change in the matched control contrast.

From repository root, with original SHA-pinned reports/receipts and metadata
manifest available, use a NEW output directory:

```sh
python reports/loss2-LAST60-matched-val-20261112/compare_when_complete.py \
  --fresh-report "$LOSS2_REPORT" --fresh-report-sha256 03928460e391ef87df5ad4faf6e3005be1e5190bf87d766c6260c2803ee3fb79 \
  --control-report "$TPE5_LAST60_REPORT" \
  --terminal "$LOSS2_TERMINAL" --terminal-sha256 9d8e48e22e5b9c18d485c825a96c69f522897e50b54ac728476ef6ab93d9dc79 \
  --outer-exit "$LOSS2_OUTER" --outer-exit-sha256 d62b590ab901b0e9b0a13fb596703e0019d017c1d9a01400d6a5c21b0765e03b \
  --postcheck "$LOSS2_POSTCHECK" --postcheck-sha256 7810d915ea9ec88c9aedc1f7c43ef74c9463374a026c150d0c80484b4413f549 \
  --runtime "$LOSS2_RUNTIME" --runtime-sha256 9cab57e7aff61da12f3a37b5f25284d14071965c6c8c95314fc1e67dac7caacc \
  --manifest "$CANONICAL_MANIFEST" --output "$NEW_ANALYSIS_DIRECTORY"
```

Prior9/17-row snapshots and their citations were not rewritten; exact byte pins
are in `unchanged-prior-report-pins.json`. See the [previous overview](../all-head-candidate-comparison-20261112/README.md).
No new model/GT decode, GPU/NN/TEST, DB, job or input mutation occurred in this
count-analysis task.
