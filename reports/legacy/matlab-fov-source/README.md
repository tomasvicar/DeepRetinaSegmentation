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

Bounded threshold follow-up reproduction:

- In the same external oracle directory, run the public-API synthetic generators
  `generate_hist_multithresh_probe.m`, `generate_hist_holdout.m`,
  `generate_hist_constant.m`, `generate_multithresh_fminsearch.m`, and
  `generate_threshold_holdout.m` in order. Each is a script; invoke its actual
  filename stem with MATLAB `-batch`. Use CUDA hidden, nice10 and
  `-singleCompThread`; these are small histogram/search calls, not model inference.
  The last script uses the previously extracted two real frames at stride8 and
  never loads annotations or performs model-quality evaluation.
- Run `package_threshold_study.py --oracle-dir SCRATCH --fixture NEW_METADATA.json
  --report NEW_REPORT.json`. It imports the adjacent, clearly uncertified
  `nelder_mead_candidate.py`. Reports/metadata open exclusively.
- `package_multithresh_fixture.py --oracle-dir SCRATCH --fixture-dir NEW_DIRECTORY`
  packages only synthetic images; names beginning`real_` are excluded. The small
  synthetic NPZ is85,992bytes. CI retains the known local-search counterexample.
- `threshold-study-report.json` records407 native fixed250 histogram matches,
  the54 nondegenerate threshold cases (53 full vectors/54 first cutoffs match),
  negative-infinity statuses and unported degenerate policies. Native histograms
  are available only in the optional module; no public profile was activated.

The explicit non250 development counterexample and source local/global objective
contrast prevent extending the measured claim to arbitrary histogram bin counts
or a global Otsu substitute. No proprietary toolbox source was inspected.

`global_otsu_counterexample.py --oracle-dir SCRATCH` independently recomputes the
reported three-mode global-DP objective contrast; it is not a runtime replacement.

First-cutoff follow-up (current optional contract):

1. Run `generate_firstcut_holdout.m` and `generate_firstcut_holdout2.m` in MATLAB
   with CPU1/nice10/CUDA hidden. They preserve independent seeds11827/90173 and
   checkpoint generated MAT files every25 synthetic inputs.
2. `prepare_firstcut_real.py --data-root OWNED_DATABASES --output SCRATCH` extracts
   only three raw calibration images. Output must be outside owned databases;
   source metadata names train STARE/Drishti frames and an image-only DiaRetDB0
   input. It never reads annotations. Use the verified7bb souradnice helper in
   scratch, then run `generate_firstcut_real.m` and
   `generate_firstcut_synthetic.m`. These are FOV stages only, never model inference.
3. `compare_firstcut_real.py --oracle-dir SCRATCH --report
   SCRATCH/firstcut_real_native_comparison.json` compares the explicit native
   backend with actual source stages. Reports refuse overwrite.
4. `package_firstcut_followup.py --oracle-dir SCRATCH --fixture-dir NEW_DIRECTORY
   --report NEW_REPORT.json` creates the numerical contract report plus compact
   histogram-count/synthetic-spatial fixtures. Real photos never enter fixtures.
   The old threshold-study report used rounded coordinates; the follow-up
   explicitly corrects its fractional failed-search interpretation.

See [MATLAB_FIRST_CUTOFF.md](../../../docs/MATLAB_FIRST_CUTOFF.md) for the actual
source caller lines and opt-in API. Full threshold-vector parity remains
unclaimed, and the default oracle guard/public profiles remain unchanged.

`firstcut-verification.json` and`firstcut-tests.log` preserve the focused1869-pass
component run, observed Python/NumPy/SciPy versions and actual module/log hashes.
They do not attest a complete model-evaluation runtime or claim model quality.
