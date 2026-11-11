# Actual LC BEST60 CPU compatibility parity on one TRAIN photo

The original LC BEST60 checkpoint
`dfef53e453c51e87980c5132b8d829f7720b0f32b172982f4e1e1594b7449386`
produced bitwise-identical network inputs, FP32 probabilities and all five mask
planes under frozen [fd44](https://github.com/tomasvicar/DeepRetinaSegmentation/tree/fd44cdfa83d1df9ccc42960df6d1d2018db29900)
and reviewed product [1fb](https://github.com/tomasvicar/DeepRetinaSegmentation/tree/1fb4a0c304578c092f4e0c03c4684e4bb104fd70).
This used **one actual DRIVE TRAIN photograph**, its canonical prepared RGB and
standalone official/image FOV. No GT masks, VAL/TEST data or GPU were accessed.

The full saved config, preprocessing descriptor, geometry policy, original
training-source and cache identity were retained and checked without stripping
fields. Each source applied `legacy_lc_clahe_v1` **once to the full1031×1033 prepared
image before NN tiling**, preserving raw RGB outsideFOV. This was on-the-fly input
processing, not a training-RGB-cache lookup. Original BF16 training metadata stayed
unchanged; actual inference was CPU FP32, none, tile512/overlap64/threshold.5.

| Comparison | Shape | Result |
|---|---|---|
| Processed uint8 RGB and FP32 network input | 1031×1033×3 /3×1031×1033 | Exact matching SHA |
| Prepared probabilities | 5×1031×1033 | Finite FP32, bitwise exact, maxerror0 |
| Native-restored probabilities | 5×584×565 | Finite FP32, bitwise exact, maxerror0 |
| All five native mask planes | 5×584×565 | Every bit exact |

Prepared probability SHA is
`eb8ed5876191ee2498cd1056778c0699285814ddc8d920cb19b1911cb3e351b5`;
native probability SHA
`51dd0ef07804653aed17531b23ff8ef7d662c88720d3875c8424446b74cd6040`;
mask SHA
`7b91decb6624b244b45b15b8d55d59f83760c917f83a2ac266d90cf5b164fdb3`.
Complete saved recipe, configuration, input/output hashes, error metrics and
portable cache provenance are retained in [results.json](results.json).

The actual child and external observer both returned0. Duration29.402s includes
imports, loading, LC and both source predictions; sampled ownedgroup peakRSS was
862,760,960bytes (~0.804GiB), within120s/RSS2.5GiB/minavailable2.5GiB limits. One CPU
core, intra1/inter1/OpenCV1, nice17/idleIO, CUDA hidden; no limits or retries.
Original sources/model/inputs/helper hashes remained unchanged. Arrays stayed
transient outside Git, and no image identifiers, host paths, process handles,
commands or source arrays are published. [Digests](provenance-digests.json) bind
original private numerical/exit/integrity evidence.

This checks a specific nonraw **functionality compatibility path**, not segmentation
quality, clinical generalization, corpus-wide or cross-device bitwise parity.
The recorded duration is not resident model latency or a speedup benchmark. It
does not select/promote a model, freeze a TEST method or create completion
eligibility. The [earlier raw compatibility checks](../product-photometry-real-train-parity)
remain separate observations. No additional model operation follows from this report.
