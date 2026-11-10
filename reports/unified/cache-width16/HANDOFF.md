# Cache-width16 analysis handoff

Analysis worktree: /tmp/DeepRetinaSegmentation_cache_analysis.
Registered branch: experiments/cache-width16-analysis, based on d83f17a.
Created with --no-checkout. Sparse initialization failed on the known shared Git
metadata ENOSPC writing .git/config. Publication uses a temporary GIT_INDEX_FILE,
read-tree from the complete d83 parent, explicit new-path add, write-tree and
commit-tree, followed by raw commit push. Shared index/config and local HEAD are
not changed; the remote published branch may therefore lead the local branch.
After admin repair, initialize sparse checkout for /reports/unified/cache-width16/
and /tests/test_cache_checkpoint_comparison.py before normal checkout operations.
Never add -A from this unpopulated no-checkout worktree.
No source/runtime/data/weights files from any training checkout were modified.

Ready source: compare_checkpoint_states.py plus README and tiny CPU tests.
Six synthetic CPU cases pass; ruff check/format pass with HPO pyproject settings.
The actual original/cached E1 comparison has NOT been run. Wait for fresh cached
width16 E1 completion; do not migrate old checkpoint state into the new run.

Tool snapshots safe weights_only checkpoint loads, recursively reports exact
model/buffer/optimizer/scheduler/RNG/coverage/config state and scalar differences,
nonfinite/missing keys, retains model/source/canonical/cache identities, compares
embedded validation headers and per-photo confusion counts, and snapshots epoch1
CSV loss/timing. Extra new per-photo AV counts are explicitly excluded. Existing
output reports are preserved. No images/model tensors are written into reports.
CSV times exclude startup and preflight; optional external paired wall-time
arguments provide an honest end-to-end timing instead. No universal CUDA
bitwise/determinism claim is made from a single observed pair.

The exact helper SHA256 is
9032f00e2244b401b488bdaf3d877933ce8b918e4f89c2145b6449f61aff1367.
A verified read-only copy is staged on Ada outside all training checkouts at
/var/home/tom/analysis/deepretina/cache-width16/9032f00e2244b401/.
Fresh cached width16E1 PID1067139 started2026-11-10T14:48:35UTC. This handoff
does not claim that it has finished or passed QA; actual comparison must wait for
root confirmation that E1 is terminal and valid.
