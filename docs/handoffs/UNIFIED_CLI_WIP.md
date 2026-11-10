# Unified public CLI: ready for review

Worktree: `/var/home/tom/code/DeepRetinaSegmentation_unified_cli`.
Branch: `experiments/unified-public-cli`, based on HPO `effabaf`.
No running training, immutable source87 checkout, canonical corpus, stable shipping
branch, model weights, or GPU was changed. Cache integration remains out of scope.

Implemented:

- `cli.py`: public `train-unified` and family=unified dispatch. Unified prediction
  uses checkpoint tiling defaults; pilot defaults remain512/64.
- `unified_cli.py`: delegates the verified trainer; flat JSON/TOML settings,
  saved config/content pin restored on resume; training logs stderr/result JSON
  stdout. Native raw single/folder prediction rejects incompatible legacy modes
  and non25PPD. Validation requires checkpoint manifest+audited content pins;
  test remains blocked until a reviewed frozen-selection schema exists.
- `unified.py`: optional inference tiling overrides are applied/recorded, while
  whole-view checkpoints reject meaningless tile overrides.
- `test_unified_cli.py`:21 CPU cases, including actual tiny generated all5head
  training/prediction, JSON/TOML resume bitwise equal to uninterrupted training,
  native outputs, override forwarding, pins, failures, batch and pilot defaults.
- README and UNIFIED.md document experimental/current nonfinal quality and usage.

Validation: **505 full-suite tests passed**, 58 existing morphology/multiprocessing
warnings; ruff check and format checks passed (115files). No new checkpoint alias,
weights, or quality claim. Test data/checkpoints were generated only in temporary
test directories. No test metrics were opened for actual experimental models.

The optional inference API changes unified.py's executable-source fingerprint;
old immutable87 runs therefore remain on their original source for continuation.
No source-guard relaxation or old-run migration was introduced. Old unified
checkpoints remain prediction-compatible; new CLI runs resume exactly as tested.

Next: root review and integration into HPO only. Stable shipping integration and
final unified model selection remain separate tasks.
