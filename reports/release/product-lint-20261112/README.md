# Full product lint and test inventory

The first whole `src tests scripts` Ruff check at source `d2ada5b` found four
unsorted test import blocks and two formatting differences. The original failing
outputs are preserved in `verification.json` and the adjacent logs.

Commit `275c769` fixes only those six files. All non-import AST statements remain
identical; the production module's entire AST is identical. Ruff 0.17.4 then
passed both checks: lint exit 0 and format exit 0 (136 files). These were low
priority checks on CPU6, with CUDA hidden, each under a 30-second limit. No
model, dataset, forward, training, package installation or GPU operation occurred.
See `verification-after.json` for exact commands, times and source-file hashes.

The old inventory producer hardcoded an old source revision when regenerated.
Commit `a30f43c` makes it record the actual Git HEAD and whether tracked files are
dirty. Its direct lint and format checks passed. `inventory-current.json` records
the clean `a30f43c` snapshot: 63 modules and 456 static test definitions, with exact
file hashes. The historical inventory is unchanged. These counts are not collected
pytest cases or PASS counts; the generator imports no tests or ML libraries.

A separate full CPU pytest run is being prepared on Old against exact source
`275c769`; its actual results belong in their own report. The later generator
change does not alter production code or test definitions.
