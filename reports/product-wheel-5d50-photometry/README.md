# Current photometry product wheel: packaging/import check PASS

Source `5d50fc523858909cf3ea482f4c018fd529ff48a8` produced **one** wheel using
`uv build --wheel`. The wheel was installed offline with `uv pip install
--no-deps --target` into an isolated HOME directory, then exercised **outside any
checkout** using the existing dependency runtime. Build/install/check actual
return codes were all0 (5.02s /0.67s /2.97s). No source archive/sdist was built.

- All **12 bundled weight SHAs**, source registry and aliases match in source,
  wheel and installed target. No weight, alias or default was changed.
- All45 runtime Python files in the actual source package are byte-exact in the
  wheel, including `photometry.py`, `photometry_cache.py`, unified CLI and native
  training modules. Product TTA stays inline in `unified.py`.
- Runtime imports and distribution metadata resolve to the isolated installed
  target, with no editable-checkout fallback. CLI main/train-unified/predict
  help and the generated console entry point succeed; both new photometry-cache
  options parse correctly.
- Default `photometric_profile=raw` and its immediate unchanged-input path are
  preserved. Both opt-in profile descriptors import correctly. Existing runtime
  dependencies were not updated: all42 distribution METADATA hashes stayed equal.
- Checkpoint loading, NN forward and CUDA query/initialization functions were
  trapped during import/help checks. No data/GT decode or inference occurred.

The locally built wheel was62,181,241 bytes; SHA, all12 individual model SHAs,
module coverage, command outcomes, runtime descriptors and scope are in
[`verification.json`](verification.json). This is a packaging check, **not** a
new numerical-parity, quality, speed, from-scratch dependency installation or
GPU-support test. Private operator commands/logs retain the actual evidence.

The handoff remains the Git repository. The throwaway verification wheel/target
are not published or committed; source, dependencies and model defaults remain
unchanged.
