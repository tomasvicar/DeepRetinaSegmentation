# Focused existing product regressions after optional bundled defaults

Actual source `19f8792d715219fcea1bcb7a5977f93bb7adafbb`, 12 November2026,
CPU1/core4/nice17/idleIO, all numerical threads1/CUDAhidden. Eight selected modules
ran once under an owned180s bound; **96passed,1failed,4warnings in16.50s**. The
externally observed pytest child returned1 after17.86s including startup. This
is not a fresh entire-suite result. The full original log and numeric receipt
are preserved, not replaced by the later correction.

Selected scope: bundled checksum/safe CPU metadata for all12 distributed models,
small synthetic pilot prediction, original vessel/A/V/raw-profile/calibration
CLI routes, toy composed-original reference and guards, prediction output
preservation, mocked source CLI/training forwarding, preview recipe, and all new
defaults routing tests. No real photo/GT, training, validation quality scoring,
GPU or held-out TEST ran. Original model and registry files were never modified.
Known skimage morphology deprecation warnings remain visible.

## Failure and narrow correction

The sole failure was existing
`test_tpe_preview_release::test_preview_recipe_exact_actual_checkpoint_except_portable_device`.
Original BEST58 config predates the opt-in `photometric_profile` field, whereas
the current dataclass expands it to the intended default `raw`. Strict whole-dict
comparison therefore saw exactly one extra item (`photometric_profile: raw`).
This was a pre-existing test expectation after the photometry port, not changed
weight bytes or a defaults-routing/prediction failure.

The root-reviewed test-only correction explicitly requires actual recipe/profile
raw and adds **only absent `photometric_profile=raw`** to the expected historical
configuration, alongside the existing portable-device exception. Every other
saved field still participates in exact dictionary comparison; no generic unknown
field filtering, checkpoint edits or runtime changes were introduced.

Only that metadata case was rerun afterward: **1passed2.29s**, actual child0 after
3.02s including startup, owned60s bound. The96 originally passing cases, including
small NN checks, were **not repeated**. See `corrected-case.stdout.txt` and
`corrected-case-observed-exit.json`; the latter pins the corrected test file SHA.
Do not describe this as one fresh97-pass run or latest-full-suite green.

`pytest.stdout.txt` and `observed-exit.json` retain the first original failure.
No runtime/registry/weight changes were needed. Future inventory regeneration
must reflect the one test-source hash update; module/definition counts are
unchanged. The overall product registry and all original aliases/defaults remain
as in the preceding optional-defaults verification.
