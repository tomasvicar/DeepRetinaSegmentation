# Measured vascular4 BEST55 inference tiling on GTX1060

All four cases completed with observed child exits0 and external observer exit0.
Operational identities and paths remain private; original artifact SHA provenance
and exact sanitized per-photo count views are provided here.
Total supervisor wall890.21s (14m50s), within the reviewed60min budget; original
model/source/canonical/provenance inputs remained unchanged. No retraining, TEST
or model selection was performed. The reports use frozen source0b and actual991…BEST55
with all5700 selected annotations, on the same GTX1060/FP32/seed42/runtime2/1.

| Tile / overlap | Macro | Vessel | Artery | Vein | Disc | Cup | Conditional AV | Invocation s | VAL-loop s |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 512 / 64 | 0.80597259 | 0.76038525 | 0.73122814 | 0.79260736 | 0.93051121 | 0.86003912 | 0.82287940 | 204.96 | 147.67 |
| 512 / 192 | 0.80617481 | 0.76060451 | 0.73131691 | 0.79252555 | 0.93076198 | 0.86076932 | 0.82292204 | 217.13 | 159.91 |
| 768 / 192 | 0.80628008 | 0.76051280 | 0.73194063 | 0.79302509 | 0.93069206 | 0.86064048 | 0.82353432 | 214.13 | 156.95 |
| 1024 / 256 | 0.80643060 | 0.75962890 | 0.73228342 | 0.79310077 | 0.93092850 | 0.86098986 | 0.82390214 | 253.70 | 196.58 |

The512/64 report is a new matched GTX1060 control, not a bitwise retag of earlier
Ada evidence. Actual Python3.12.14 and GPU backend are recorded. All770 photos,
source/record/observer/positive/valid/conditional supports match exactly. The
15 manifest dataset labels reduce to14 scored annotation-source labels because
DRIVE/RITE photograph aliases use the deterministic RITE A/V annotation priority.

## Paired changes relative to512/64

Source-stratified bootstrap of612 known split groups,2000 draws/seed42. Groups
preserve known aliases; they are not proven complete independent patients.
Intervals below are exploratory percentile95% intervals, without multiplicity
correction across protocols or heads. The original complete machine reports
include every source/head count, regression and supported-group size.

| Protocol | Macro delta [CI] | Vessel delta [CI] | Conditional AV delta [CI] |
|---|---|---|---|
| tile512-overlap192 | +0.00020221 [-0.00002268, +0.00045699] | +0.00021926 [+0.00008479, +0.00035344] | +0.00004265 [-0.00057592, +0.00065935] |
| tile768-overlap192 | +0.00030748 [-0.00017152, +0.00074053] | +0.00012755 [-0.00003189, +0.00029319] | +0.00065492 [-0.00028239, +0.00153805] |
| tile1024-overlap256 | +0.00045800 [-0.00008181, +0.00092324] | -0.00075635 [-0.00099917, -0.00052785] | +0.00102274 [+0.00006205, +0.00200324] |

No variant establishes an overall macro improvement with a95% interval excluding
zero.512/192 shows a small pooled vessel improvement (+0.00021926) and cup gain,
with a slightly lower vein point estimate.1024/256 has the largest macro point
estimate and conditional A/V gain, but **worsens pooled vessel Dice by0.00075635**,
with an entirely negative paired interval. It is not a uniform improvement and
no new default/protocol/model is promoted.

Invocation and VAL-loop times are measured single passes in fixed order; cold
startup, filesystem state, shape-specific kernels and pre/post identity scans
differ. These are not steady resident latency or peak-memory benchmarks. No
VRAM reduction or general cross-hardware speed claim is made.

## Vessel source detail

All complete per-source five-head counts and intervals are in the paired JSON.
This descriptive table keeps every source with vessel support visible.

| Source | Supported photos / groups | Baseline vessel Dice | Delta512/192 | Delta768/192 | Delta1024/256 |
|---|---:|---:|---:|---:|---:|
| aria | 28 / 16 | 0.70868474 | +0.00048614 | +0.00019763 | -0.00201589 |
| avrdb | 20 / 20 | 0.97140484 | +0.00000000 | -0.00020651 | -0.00020651 |
| chasedb1 | 6 / 3 | 0.75685988 | +0.00000000 | +0.00002552 | +0.00002552 |
| drhagis | 8 / 8 | 0.75602038 | -0.00002544 | +0.00047759 | +0.00065686 |
| erlangen | 22 / 11 | 0.63285177 | +0.00000000 | +0.00000000 | +0.00000000 |
| hrf | 9 / 3 | 0.78561151 | +0.00007947 | +0.00023846 | +0.00024050 |
| iostar | 6 / 6 | 0.80138846 | +0.00001084 | -0.00004515 | -0.00016130 |
| rite | 4 / 4 | 0.79807759 | +0.00000037 | +0.00010829 | +0.00119847 |
| stare | 4 / 4 | 0.80651845 | +0.00000000 | +0.00025215 | +0.00012744 |

Original JSONs are retained byte-identically. The analysis used explicitly
derived temporary count views, binding both whole enriched-report SHA and
derived-view SHA; no original standalone512/64 qualification was bypassed or
relabelled. Comparison math is unchanged `deepretina.comparison.compare_reports`.
