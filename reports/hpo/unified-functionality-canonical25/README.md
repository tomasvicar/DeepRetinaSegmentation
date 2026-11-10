# Functionality check only

One shared RGB five-output model was run for two optimizer updates on the canonical legacy25 core preview (DRIVE/RITE/Drishti), not full data. Training had4photos/6annotation records; validation2photos. All five output heads and both AV classes had real reference support. Finite training loss1.4746486; training4.6seconds and validation1.75seconds on GTX1070. Validation Dice0 after two updates is not a quality estimate. No test evaluation occurred. Model parameters1,106,597. Checkpoints remain in ignored runs/unified-functionality-canonical25.

This check predates the final coverage-first sampler and joint-AV interpolation correction; current contracts are verified by tests/test_unified.py. No accuracy improvement is claimed.
