# TPE72 BEST58: measured resident CPU example

The genuine completed native TPE BEST58 checkpoint (`20d4caab…f474d4`) was
measured against the original selected four-network pipeline on the SAME raw
DRIVE TRAIN21 image and official FOV. This is a performance example, not a
final model choice or a quality test.

| Measure | Original four networks | TPE BEST58, no TTA |
| --- | ---: | ---: |
| Resident median, one CPU thread | 59.635057 s | 8.416287 s |
| Three timed calls | 60.116718 / 58.461116 / 59.635057 s | 8.388617 / 8.471065 / 8.416287 s |
| Parameters | 5,943,312 | 1,965,637 |
| Actual checkpoint bytes | 25,068,700 combined | 8,894,002 |

Within this run the median ratio is **7.085673×**. One warmup preceded each
three-call measurement. Timing includes preprocessing, inference and restoration
of five masks, excluding file I/O and model loading. The final mask reference is
captured uniformly inside each call; shape/boolean dtype/finite checks and mask
hashing occur outside the clocks. Exactly four calls per pipeline, no extra NN.

The TPE BEST contains no optimizer/training state. Floating model state is
float32 (7,874,468 tensor bytes including nonfloating buffers); BF16 was a TRAIN
setting, not prediction quantization. Serialized size includes metadata and
coverage and is not a compression measurement. This width16 model is larger
than the earlier width12 vascular2 preview; their quality evidence is separate.

## Actual conditions and evidence

Runtime package: exact independent scientific source
`1f723ad92898fad8b04a6ef8b39065f021f52a9c` (`experiments/fd44-joint-tta`),
with default `none` inference; source files/pyproject/lock were hashed unchanged
before/after. Original checkpoint training source remains72/3725, separate from
this evaluation source. The unchanged public benchmark script is pinned by
SHA256`3a0324e09c128b36381b1ab219c88c8f10e06090918c10023901588da7ac1aae`.

AMD Ryzen5 1600, affinity4, one Torch intra/inter-op/OpenCV thread,
OMP/MKL/OpenBLAS1, nice17/idle I/O, CUDA hidden and uninitialized.
Python3.12.13/Torch2.6.0+cu124; complete versions/source/output checks are in
[runtime evidence](runtime-output-verification.json). No installation.
The actual bounded child returned0 after283.31s under a600s owned-process cap.
Whole-process peak RSS was2,955,284KiB (~2.82GiB), not per-pipeline RAM.
All eight image/FOV/checkpoint/reference byte hashes stayed unchanged.

[Every timing sample](timing.json), [actual terminal](producer-terminal.json),
and [safe checkpoint metadata](checkpoint-role-metadata.json) are preserved.
The model's [full770 fresh VAL](../../tpe72-completed-fresh-val/README.md) is
separate: vessel regression versus the original remains. These timings measure
NO TTA; do not attach the ongoing TTA sweep's quality scores to this latency.

## Qualifications

Only one TRAIN image (584×565), no segmentation GT/TEST arrays or extra images.
Representative geometries and hardware need their own measurements. These are
resident clocks, not CLI startup throughput. Host load and fixed pipeline order
can affect times; no universal speed or GPU claim follows.

The original reference is composed528233fe…: vessels +dedicated disc0/raw
+universal/raw cup clipped by the same-forward dedicated disc +AV-v9 with
calibrated provided FOV and predicted vessels. The timing recipe matches that
reference. It is distinct from corrected flatd14 VAL; this timing does not
relabel its quality or establish MATLAB/raw bitwise equivalence. Historical
original exposure and photometric residual qualifications remain.
