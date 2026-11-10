# Rebuilding external MATLAB goldens

Real RGB/FOV/MAT files are deliberately outside Git. A clean restart can recreate
them from the user's original data and repository; `/tmp` is not required.
MATLAB R2026b Update1 + Image Processing Toolbox was used for the recorded oracle.
The Python package itself needs neither MATLAB nor these generators.

1. Extract the owned `DRIVE.zip` and `HRF.zip` into an external scratch directory.
   Use `DRIVE/training/images/21_training.tif` if present **only after checking it
   decodes to the same pixels**; this audit actually uses `21_training.bmp` from
   the owned archive. The official FOV is `training/mask/21_training_mask.gif`.
   HRF uses `images/01_g.jpg` and `mask/01_g_mask.tif`. Archive layouts may include
   a redundant shared-drive prefix; select the canonical matching resource.
2. Recover the three **user-owned** MATLAB helpers from the original repository:

   ```bash
   mkdir -p /tmp/retina-reference/helpers /tmp/retina-reference/goldens
   git show 7bb0539b90dfce2ed6eaeef1de812072be1155d0:matlab_preprocessing/souradnice.m > /tmp/retina-reference/helpers/souradnice.m
   git show 7bb0539b90dfce2ed6eaeef1de812072be1155d0:matlab_preprocessing/image_adjustment.m > /tmp/retina-reference/helpers/image_adjustment.m
   git show 7bb0539b90dfce2ed6eaeef1de812072be1155d0:matlab_preprocessing/local_contrast_and_clahe.m > /tmp/retina-reference/helpers/local_contrast_and_clahe.m
   ```

   A shallow shipping clone may need the original history fetched, or the helpers
   from the user's original checkout at that reference. No toolbox source is used.
3. Start MATLAB CPU-only with `CUDA_VISIBLE_DEVICES=''`, `nice -n 5`,
   `-singleCompThread -batch`, and call the committed
   `generate_real_goldens(output_dir, helper_dir, drive_image, drive_fov,
   hrf_image, hrf_fov)` function after adding this report directory to MATLAB's
   path. All six parameters are explicit absolute paths; output directory and
   source helper directory are separate. The generator writes DRIVE25, HRF25,
   DRIVE35 and the intermediate coords/photometric stages used by the Python
   comparisons. Original data/helpers are only read.
4. In that external output directory run the committed `generate_resize.m`,
   `generate_photometry.m`, and `probe_gaussian_domains.m` scripts (add their
   report directory plus helper directory to MATLAB's path). These create the
   synthetic arrays used by `compare_profile.py`.
5. Run `uv run python reports/legacy/matlab-av-profile/compare_profile.py
   --golden-dir /ABS/OUTPUT --output /ABS/comparison.json`, followed optionally by
   `compare_model_forward.py --golden-dir /ABS/OUTPUT --database-dir /ABS/ARCHIVES
   --checkpoint /ABS/legacy-av-v9.pt --output /ABS/model-forward.json`, with
   `CUDA_VISIBLE_DEVICES=''`. Neither script updates weights or source data.

MAT-file file hashes can differ across reruns because MATLAB stores a creation
header. Compare decoded arrays and stage errors as well as provenance hashes;
bitwise file equality is not the acceptance criterion. The radius/edge generator
and synthetic-mask fixture generator are in the sibling
`matlab-fov-calibration` report directory.

A shell example using explicit environment paths (replace every `/ABS/...`):

```bash
export RETINA_ORACLE_REPORTS=/ABS/REPO/reports/legacy/matlab-av-profile
export RETINA_ORACLE_HELPERS=/ABS/HELPERS
export RETINA_ORACLE_OUTPUT=/ABS/OUTPUT
export RETINA_DRIVE_IMAGE=/ABS/DRIVE/training/images/21_training.bmp
export RETINA_DRIVE_FOV=/ABS/DRIVE/training/mask/21_training_mask.gif
export RETINA_HRF_IMAGE=/ABS/HRF/images/01_g.jpg
export RETINA_HRF_FOV=/ABS/HRF/mask/01_g_mask.tif
CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 nice -n 5 matlab -singleCompThread -batch "addpath(getenv('RETINA_ORACLE_REPORTS')); generate_real_goldens(getenv('RETINA_ORACLE_OUTPUT'), getenv('RETINA_ORACLE_HELPERS'), getenv('RETINA_DRIVE_IMAGE'), getenv('RETINA_DRIVE_FOV'), getenv('RETINA_HRF_IMAGE'), getenv('RETINA_HRF_FOV'))"
```
