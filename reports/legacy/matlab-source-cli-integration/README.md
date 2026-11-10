# Optional source CLI integration review

Based on stable `2d07be5`, in the separate branch `experiments/matlab-source-cli-integration`.
The required primitive commits cb57257 and85f4d8c precede the requested44983fb→4b36d2c→08190d8→67fc303 sequence. All six cherry-picked without conflicts. No package dependencies were added: NumPy/SciPy/scikit-image/Pillow and the unchanged Torch inference route are already present under uv.

`--av-raw-matlab-source` explicitly selects automatic source first-cutoff FOV or the measured provided-mask source branch. Existing raw MATLAB/proxy modes, default oracle backend, aliases, weights and thresholds remain unchanged. Camera angle, binary provided vessel, optional original vessel checkpoint and diameter controls are retained. Invalid/ambiguous settings fail before optional model inference; provided/automatic masks retain their distinct source policies. The RGB preprocessing is computed once and exact nearest-center vessel resampling reuses the resulting geometry.

[2171 focused CPU regressions](focused-tests.log) passed, including previous1320 first-cutoff fixtures, native FOV/photometry goldens, CLI routing/mode rejection, automatic-only API compatibility, explicit provided-special calibration and exact input tensor equivalence with direct preparation. CLI source tests stub model forwards; no new real-image/model-forward study, GPU job, training, or TEST scoring was performed. Ruff and diff whitespace checks passed. [Current component hashes](verification.json) supplement the original immutable study receipts, whose recorded source hashes refer to their original commits.

**Qualification:** measured resize/photometric residuals and the documented DiaRet Sobel counterexample remain. This is not blanket MATLAB numerical parity, recovered training-file provenance, or an accuracy improvement. See [MATLAB_SOURCE_AUTO](../../../docs/MATLAB_SOURCE_AUTO.md) for original measured real-chain evidence, TP-independent class sensitivity and limits. The `.m` files are reference generators only; Python is the application runtime.

Frozen3157/8c/87 and stable2d07be5 were not modified. This branch requires root review before any stable promotion.
