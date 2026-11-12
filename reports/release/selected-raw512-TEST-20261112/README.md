# Final frozen TEST: raw512 BEST50 and original four nets

Both one-time TEST cases completed over **all 856 photos**, with actual child exits 0 and external observed exit 0 at **14:17:29.224791 UTC on 12 November 2026**. Source and inputs remained unchanged, and photo/record/source/GT support matched exactly. There was no retry, model fallback, threshold retuning or further inference. The 1,687.201-second supervising duration includes gates, I/O and scoring; it is not a latency benchmark.

**The vessel objective remains unmet.** Artery, vein, disc, cup, macro and AV accuracy improve. Pooled vessel is slightly lower with an interval crossing zero; this does not establish equivalence. The dense known-label vessel deficit has a wholly negative interval. All-head superiority must not be claimed.

| Metric | Original | Unified | Difference | Paired 95% interval |
|---|---:|---:|---:|---:|
| vessel | 0.787840026 | 0.786564780 | -0.001275247 | [-0.003412136, +0.000889569] |
| artery | 0.695384358 | 0.784664776 | +0.089280418 | [+0.074746681, +0.102069006] |
| vein | 0.706571583 | 0.832444103 | +0.125872521 | [+0.112839597, +0.140195285] |
| disc | 0.892787774 | 0.946856475 | +0.054068701 | [+0.037375008, +0.079894503] |
| cup | 0.454490768 | 0.885679944 | +0.431189176 | [+0.415276774, +0.449559914] |
| Source/head macro | 0.681857637 | 0.839614248 | +0.157756611 | [+0.148868329, +0.167305340] |
| Conditional AV | 0.751861050 | 0.860869209 | +0.109008158 | [+0.095499033, +0.122330485] |
| End-to-end AV | 0.641227548 | 0.752939771 | +0.111712223 | [+0.098469290, +0.123645020] |
| Dense vessel | 0.780767926 | 0.776547691 | -0.004220235 | [-0.006314644, -0.002161060] |

The first five rows are pooled Dice. Macro is the declared source/head macro. Conditional and end-to-end AV share 4,268,020 eligible/scored pixels in both reports. The paired method uses 2,000 source-stratified draws, seed 42, across 696 known groups. Known groups are not proven independent patients; source intervals are exploratory and uncorrected.

The dense supplement contains **102 TEST photos / 71 known groups**, excluding only 20 positive-only AVRDB vessel photos. Unified dense precision/recall is **0.742584207 / 0.813766845**, versus original **0.743084250 / 0.822477852**. This TEST domain is distinct from the earlier VAL 87/55 supplement. It is a labeled-support supplement, not a new objective or gate.

Source-specific adverse vessel results remain visible: ARIA −0.005993817, CHASE −0.022241496, DRHAGIS −0.029477518 and HRF −0.008058729 have negative paired intervals; RITE −0.000771691 has an interval crossing zero. The complete [paired results](paired-test.json) retain every head, source and AV interval, including AVRDB and null unsupported domains. [Adverse counts and intervals](source-adverse-results.json) and the [dense supplement](dense-vessel.json) are also included.

## Frozen scope and provenance

Unified is genuine **curated_fd44 raw512 training BEST50**, original checkpoint `521b8881d1b2c69a7395be6687a8b86581838e3f5dbb004a150333d454132cea`, own 2,529 TRAIN photos / 5,700 contributing annotations, with original full 60 completion. Its frozen protocol is rot4, scalar 0.35, tile/overlap 512/64, FP32 without autocast. Original is exactly four nets: universal, vessels, dedicated disc0 and AVv9, using raw vessel inputs, calibrated provided-FOV AV and universal cup intersected with the same-forward dedicated disc0. Full original weight hashes, saved configuration, role coverage, source and protocol are in [results.json](results.json).

The selection is `caef7716a5656a151cf6c0ce1e9b8cd694aa9f13620cef36f14abcd0a6368e80`; coupled original policy is `f9f54af9610bc9dbf3094ec2b1cb1d9ce972b7b3ca66504de23c45871bd509e2`. Both were frozen and checked before this TEST run. Inference used immutable commit `6d3593f5a8dbf177ebebb231b5120d4fd601ea4f`, evaluation identity `f13a23d1af43e2c23e098ce15f1d69f78251817b1a1be86415a48ffe6330b29a`, on the same GTX1060 backend with CPU 2/inter 1 and affinity 2,3. The distinct source 4f2 comparison only checks saved metadata/counts; it changes no inference mathematics.

Original complete TEST report SHA: `620f037b1b07c64bd4b92a5b309fb12cec006e29341bca9960c71b26a0c09852`.
Unified complete TEST report SHA: `a2e63c77c8e04473d8d281f81e77428dba3abb8413fc2b2884d09aee850e20ae`.

Comparison commit is `4f2bfb4e9cf75577167de459ffd127e82efad7fd`. Exact [before/after source and whole-input pins](source-and-input-unchanged.json) are included. The [per-photo ordinal counts](per-photo-ordinal-counts.json) retain every integer count and stable grouping relationship while withholding photo/record identifiers. This sanitized projection is **not an original selection-gate input**. Exact original bytes remain private; [export provenance](export-provenance.json) binds them and the actual evidence commit `a69845f`.

These scores identify the original training checkpoint 521b, not a separately exported CPU/product artifact with a different byte identity. Historical original-model training exposure and preprocessing provenance remain limitations; prior blindness, clinical generalization and equivalence are not established. TEST was not used to select another checkpoint or protocol, and the vessel deficit remains in the conclusion.
