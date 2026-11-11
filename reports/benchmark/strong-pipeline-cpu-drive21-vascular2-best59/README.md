# Current bundled vascular2 BEST59: one-photo resident CPU timing

The **actual current bundled artifact** (`unified-v2-vascular2.pt`, SHA256
`81113e7e74d86a3d51994754895d18e90b2ee8d54d5a8660a8d5535096451b7b`)
was measured against the selected original four-network pipeline. This replaces
neither the prior control timing nor any quality reference. It is a matched
single-photograph CPU example, not a general speed/quality or GPU claim.

| Measurement | Original four networks | Current unified BEST59 |
|---|---:|---:|
| Resident median | 63.324382 s | 6.026454 s |
| Three measured calls | 63.324382 / 63.938297 / 62.887477 s | 6.020245 / 6.026454 / 6.042740 s |
| Model parameters | 5,943,312 | 1,106,597 |
| Actual serialized checkpoint bytes | 25,068,700 total | 5,453,234 |

The within-run median ratio is **10.507735×**. The unified architecture has about
81.4% fewer parameters than this four-network reference. File bytes are recorded
separately from parameter counts; they include serialization and metadata.
The current BEST59 has **no optimizer/training state**. The older control LAST60
file (14,385,264 bytes) includes AdamW/RNG state and has identical model-state shapes
and model tensor bytes to this BEST. Their different file sizes do not establish
model compression or quantization; both contain float32 floating state.

## Exact scope and conditions

One known **DRIVE TRAIN21** raw RGB photograph,584×565, and its official FOV only;
no vessel/disc/cup GT arrays, held-out TEST, extra images or accuracy scoring.
Each pipeline received one warmup and three measured calls. Timing includes CPU
preprocessing, inference and restoration of all five masks to the original frame;
file/image I/O and model loading occur outside the resident clocks.

Runtime: stable production commit `2139555a4680d0d85cb151971c19aaed8a903d27` from a
separate unchanged package snapshot; source/pyproject/uv.lock hash before/after
`8b3dd3d1674bcdcaeacc14f8b3905df0e632a1540e4c302341288476ac6dbfc0`.
CPU AMD Ryzen5 1600, one logical CPU (affinity4), Torch intra1/interop1/OpenCV1,
OMP/MKL/OpenBLAS1, nice17, idle I/O, CUDA hidden and not initialized.
Python3.12.13, Torch2.6.0+cu124; all actual package versions are retained in
`runtime-output-verification.json`. No package was installed for this run.

The unchanged public `scripts/benchmark_strong_pipeline.py` from `6bf8916` is bound
by SHA256 `3a0324e09c128b36381b1ab219c88c8f10e06090918c10023901588da7ac1aae`.
The wrapper captures the last returned mask reference uniformly inside each call,
then verifies shape/dtype/finite values/hash **outside the measured clocks**.
Both returned finite boolean5×584×565 arrays; exactly four calls per pipeline,
zero additional NN calls. Checkpoint, input and reference byte hashes all matched
before/after. The original supervised BEST59 completion/quality evidence is
separate from this timing; the benchmark does not certify convergence or select
an HPO winner. A later final tuned model needs its own benchmark.

The original numeric terminal observed exit0, no timeout, total287.98 seconds
under its600-second owned-process budget. **Actual whole-process peak RSS was
2,890,024 KiB (~2.76 GiB)**, above the initial estimate. This includes imports,
loading and both pipelines; it is not a per-pipeline RAM comparison. Exact
operator commands, original process records and supervision remain private.

## Original policy and quality qualifications

The timed strong original pipeline uses the same vessel, dedicated disc0/raw,
universal/raw cup and AV-v9 weights as completed **composed reference528233fe…**.
Its declared optional cup policy intersects raw universal cup with the same-forward
dedicated disc0 before nearest restoration; calibrated AV uses predicted vessels
and provided FOV. The validation-selected component/weight/cup recipe is identical
in the declared timing reference. This does not modify original weights/defaults.

The earlier composed reference is **not** the separately corrected flat d14
VAL report (`182ad346…`) and is not relabeled as it. Timing uses current stable
nearest-corrected Python. Official raw FOV in this TRAIN example differs from the
restored canonical FOV used for full VAL. No quality or numeric equivalence to that
older composed report follows from timing these weights. Historical exposure and
photometric residual qualifications remain. The current model still has documented
vessel regression versus the strong original; faster inference does not remove it.

`timing.json` retains every sample, parameter/artifact count, model/reference/input
SHA, selected cup policy and original training configuration. `verification.json`,
`runtime-output-verification.json`, `checkpoint-role-metadata.json` and the unchanged
`producer-terminal.json` retain source/runtime/output checks and numeric completion.
No checkpoint, raw image or FOV binary is newly published with these reports.
