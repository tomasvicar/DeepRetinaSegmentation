# Final completed-TEST metadata wheel check

The wheel was built from product commit
`e2e5261b59d3cdc45610dee7a194256430a52af2`, after the completed held-out results
were documented. Compared with the previously verified runtime84b3ab9, the only
`src/` change is registry JSON: exactly the selected model's `description` and
`limitations` strings changed. All runtime Python bytes,13 model bytes, inference
defaults and historical12 entries/aliases remain unchanged. The earlier
[exact single-TRAIN parity](../unified-raw512-selected-20261112/TRAIN-parity.json)
therefore remains the functional runtime evidence; no new forward was performed.

An exclusive final-attempt directory on Old's spacious HOME reused the existing
Python3.12.14/uv runtime, with short HOME `TMPDIR`, CPU1/core4/nice17/idleIO and
CUDA hidden. No dependency upgrade, source checkout modification or GPU use.
Actual commands were:

```
uv build --wheel --out-dir ../wheel
uv pip install --no-deps --python EXISTING_NATIVE87_PYTHON --target ../installed ../wheel/deepretina-0.1.0-py3-none-any.whl
PYTHONPATH=FINAL_INSTALLED python final-wheel-check.py --source FINAL_BUILD --installed FINAL_INSTALLED --original-inventory preserved-models.json --source-commit e2e5261b59d3cdc45610dee7a194256430a52af2 --expected-registry-sha256 6cadf25368cb4324816dba53e2d120ea68a66024143d61f405d8b60a5ae4cfb8 --output NEW_VERIFICATION_JSON
```

Build, no-dependency installation and checker each had observed exit0. The first
checker attempt exited1 because its old12 inventory metadata had not been copied;
that occurred before checking model files or any inference. The exact preserved
inventory was staged, then the same metadata-only checker completed successfully.
The wheel itself was not rebuilt or changed for this preparation correction.

From HOME **outside the source/build checkouts**, verification checked installed
module origin, all48 Python module bytes against both staged and controller source,
all13 actual bundled SHA256 values, unchanged old12/aliases/defaults, exact final
registry SHA, selected rot4/.35/512/64 defaults and CLI `predict --help` exit0.
Torch stayed absent from the checker. It read model bytes only for hashes, not
checkpoint deserialization. No model forward, dataset/GT/TEST read or repeated
53-test/full-suite run occurred. See [exact proof](verification.json).

The new wheel is69,830,640B, SHA256
`a35e8d2b0270ba2c704046c117d9775318f309edd040ed2accaac9c10fd989cf`.
It remains outside Git. The packaged deployment model is still292ea599…9b29,
whereas the selected/evaluated original is521b8881…2cea; metadata updates do not
retag the deployment artifact as separately TEST-evaluated.
