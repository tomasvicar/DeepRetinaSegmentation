# Stable BF16 compatibility port: bounded CPU evidence

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

No GT arrays or TEST were decoded/scored for this real-photo check. Native CUDA
BF16 training of this port remains **PENDING**; real source 72 A100 training is not
misreported as a new port GPU test. Native hardware detection alone does not prove
binary architecture compatibility. See the user-facing precision documentation.
No model images, probability arrays, optimizer snapshots or private operational
endpoints are included in these reports.
