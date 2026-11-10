# Strong original-model validation reference

Completed on Ada on 2026-11-10: **770 validation photographs**, using the
original dedicated vessel model, original universal disc/cup model and original
VesClass v9 A/V weights. A/V inputs were regenerated through the explicit Python
raw-image proxy. This run evaluated validation only; it opened no test scores.

| Output | Dice over valid pixels |
|---|---:|
| Vessel | 0.770123 |
| Artery | 0.650863 |
| Vein | 0.664795 |
| Disc | 0.807155 |
| Cup | 0.401057 |

The mean over dataset-balanced output-channel Dice values is **0.610351**.
Conditional A/V accuracy is 0.710883; end-to-end A/V accuracy is 0.598154 over
2,626,284 labelled pixels. No undefined empty/full-vessel input was encountered.
The valid-pixel and ground-truth-positive totals match the corresponding unified
validation evaluation exactly for all five channels.

These are **exposure-qualified reference scores**. Historical image counters
cannot reliably establish which current photographs were seen during original
training or selection. The checkpoints do not fully establish whether their
training images used the raw or contrast-normalized profile. This run explicitly
uses the raw single-network profile. The regenerated A/V preprocessing is an
uncertified Python proxy for the MATLAB pipeline; native old DICOM inputs were
unavailable for a MATLAB pixel comparison. No independent quality-improvement
claim follows from this reference run.

The source was immutable commit `87cf045ab89425c4994df577fd510d210a1a0eab`.
The manifest SHA256 was
`529edba7743eb5b9175da429c2124d53f48c419b090a284b5650fd68f1fdcf69`;
the verified schema-2 prepared-content SHA256 was
`61032be218c623a910c0f3f5e2df3bfdc58e978fe494433a4e6f18d55e41693e`.

The process ran on CPU with eight threads, niceness 5 and CUDA hidden. Wall time
from launch to the final observed process sample was **1,937.41 seconds**.
Cumulative actual CPU time in that sample was **14,938.10 seconds**, sampled from
procfs at 0.1-second intervals; process exit followed within one sampling
interval. Timing covers loading, all original models, proxy preprocessing and
validation scoring on this CPU setup. Prediction-speed comparisons require
matched hardware and workloads.

Evidence:

* [Complete scores and per-photograph results](original-strong-raw-val.json)
* [Command, environment, model hashes and launch provenance](ada-original-strong-raw-val.launch.json)
* [CPU and wall-time observations](original-strong-raw-val.timing.json)

The full command and artifacts also remain on Ada under
`~/code/DeepRetinaSegmentation_reference_87cf045/`.
