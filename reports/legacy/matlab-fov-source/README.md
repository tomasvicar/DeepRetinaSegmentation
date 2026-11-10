# Rebuilding the optional provided-source FOV oracle

Only small synthetic masks are committed. Real source image pixels and MAT arrays
stay in external scratch. The oracle is a test/reference operation; Python model
prediction does not require MATLAB.

Export the unchanged USER-OWNED helper from original history:

```bash
git show 7bb0539b90dfce2ed6eaeef1de812072be1155d0:matlab_preprocessing/souradnice.m > /ABS/souradnice.m
uv run python reports/legacy/matlab-fov-source/prepare_oracle_inputs.py \
  --data-root /ABS/fundus_databases --helper /ABS/souradnice.m --output /ABS/NEW_SCRATCH
```

The helper checksum is checked. Original Erlangen/RIM-ONE ZIPs are read, one raw
image from each is copied only to scratch and hashed, and source metadata is
recorded. Original archives are not changed. A shallow shipping checkout may need
the original history or a user-owned helper exported from that reference.

In MATLAB, set current directory to that scratch and add this report directory to
the path, then run generate_provided. Use CUDA_VISIBLE_DEVICES='', nice10,
-singleCompThread and an outer90s wall limit. It calls public toolbox APIs and
souradnice only; no proprietary toolbox source is inspected. Then run:

```bash
uv run python reports/legacy/matlab-fov-source/package_provided.py \
  --oracle-dir /ABS/SCRATCH --fixture-dir /ABS/NEW_FIXTURE \
  --report /ABS/NEW_COMPARISON.json
```

Review any differences before changing committed fixtures. Real-photo cases are
represented by source hashes, generated loader FOV shapes and calibration facts;
real arrays are excluded from the fixture archive. This gate covers provided
source branches only and does not certify automatic FOV or full photometry.

Automatic-stage study (optional, not public prediction): after the same verified
`souradnice.m` has been placed in the scratch directory, copy and run
`generate_automatic_primitives.m`, then `generate_automatic_zero.m` in MATLAB
with CUDA hidden, `nice -n10`, `-singleCompThread`. Both use synthetic inputs only.
Run Python `package_automatic.py --oracle-dir SCRATCH --fixture-dir FIXTURES
--report NEW_REPORT.json` with this branch on `PYTHONPATH`. The packaged small
synthetic fixtures require no MATLAB at test time. `generate_histogram_counterexample.m`
uses `automatic_zero_goldens.mat` to reproduce the rejected floating edge guess;
it prints true source edge locations without inspecting toolbox source.

`automatic-comparison.json` explicitly distinguishes supplied oracle histogram
edges from native Python behavior. N3/4/5 multithresh and automatic histogram
edge placement remain guarded. Neither synthetic stage agreement nor the
provided-special measurements certify the complete image_adjustment chain.

Real caller gate reproduction:

1. Use `prepare_oracle_inputs.py` above to extract the two owned source frames to
   a new external scratch directory. Never use the database directory as output.
2. Export the exact user-owned helpers from revision
   `7bb0539b90dfce2ed6eaeef1de812072be1155d0` into that scratch directory:
   `git show REF:matlab_preprocessing/image_adjustment.m > SCRATCH/image_adjustment.m`
   and the same for `local_contrast_and_clahe.m`. The already verified
   `souradnice.m` is also required. Check their SHA256 against
   `adjustment-comparison.json`; toolbox source is not needed or inspected.
3. Copy `generate_adjustment_real.m` to scratch and run it with MATLAB
   `-singleCompThread -batch generate_adjustment_real`, CUDA hidden and nice10.
   This creates external `adjustment_real_goldens.mat`. It invokes original
   image_adjustment, then measures LC/CLAHE separately. The nearest-label pattern
   is generated synthetically and is not claimed to be source ground truth.
4. With this branch on PYTHONPATH, run `compare_adjustment_real.py --oracle-dir
   SCRATCH --report NEW_REPORT.json`. Reports are opened exclusively. The script
   checks source calibration/mask equality and reports all float, quantization
   and nonfinite errors; no real arrays need to be checked into Git.

The recorded runtime used the existing Python/uv environment and MATLAB R2026b,
CPU1 only. MATLAB remains an optional reference generator, never a runtime
package dependency.
