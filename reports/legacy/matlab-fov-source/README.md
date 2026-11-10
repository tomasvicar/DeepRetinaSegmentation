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
