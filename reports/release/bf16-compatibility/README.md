# Stable BF16 compatibility port: bounded CPU and CUDA evidence

The production diff ports only the original source 72 BF16 precision feature.
No coordinator, private launcher, model, dataset or alias changes are included.
The optional preset is the existing vascular2 recipe plus the precision flag.
Public CLI configuration discovery already reads dataclass fields; no new CLI
code path or separate precision override was needed.

`verification.json` records 58 CPU tests, exact default tiny training model,
optimizer and RNG equivalence against stable 2139555, CPU/non-native rejection,
AMP context and float32 state checks, resume protection and dry CLI parsing.
The source 72 precision helpers and inference wrapper are AST-identical; all other
module-level functions except the deliberately wrapped training/prediction paths
are unchanged. Existing configuration positional fields retain their order.

`actual-checkpoint-train-parity.json` records an actual immutable E3 checkpoint
from source 72, SHA 2275749ac482f603cde7b59ab67d8298cb8f0693227a5f89f6f0b1d02df612e0.
This partial snapshot has 6144 steps and is **not published as a model**. CPU
weights-only loading succeeded; parameters were finite float32. Its raw TRAIN
prediction matched frozen 72 probabilities/masks bitwise. The old bundled V2
control's probabilities also matched its saved CPU reference bitwise. All eleven
old model files, registry entries and aliases remained byte-identical.

No GT arrays or TEST were decoded/scored for this real-photo check.
`root-cpu-verification.json` records the independent61-test CPU check.
`native-cuda/verification.json` and the two runtime reports record the actual
RTX4070TiSUPER CUDA smoke: BF16 forward/backward/AdamW step, finite FP32 master
parameters/gradients/optimizer and FP32 prediction inside outer BF16 autocast.
Port and frozen72 model/optimizer/loss/prediction/CUDA-RNG outputs are bitwise
equal in separate fresh processes. Observed outer returncode0; synthetic check
elapsed4.096s. No dataset, GT, TEST or saved checkpoint was opened. Synthetic
tensor snapshots remain outside Git. This is operator compatibility, not a full
BF16 training completion, accuracy result, throughput benchmark or general GPU
architecture guarantee. Native hardware detection alone does not prove binary
compatibility. See the user-facing precision documentation.
No model images, probability arrays, optimizer snapshots or private operational
endpoints are included in these reports.
