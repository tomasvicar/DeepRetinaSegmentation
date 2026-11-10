# Automatic-source composition evidence

All MATLAB calls use public APIs plus unchanged user-owned7bb helpers, CPU only,
nice10/CUDA hidden. No proprietary toolbox source, GT annotations or actual
held-out-quality evaluation was accessed. Real arrays remain external scratch;
only synthetic fixtures, hashes and numerical evidence are committed.

Prepare the existing first-cutoff scratch inputs as described in
[the FOV guide](../matlab-fov-source/README.md), including verified souradnice.m,
image_adjustment.m and local_contrast_and_clahe.m from7bb. Then copy/run:

- `generate_half_factor.m`: four E-Ophtha/RIDB cases.
- `generate_source_auto_chain.m`: three raw train/image-only frames and two
  synthetic inputs; source image_adjustment, separateLC and stored12 contract.
- `generate_diaret_photometry_stages.m`, then
  `generate_diaret_gaussian_domain.m`: first-divergence/public Gaussian domain
  probes on the actual DiaRet shape/input.
- `generate_sobel_operators.m`: set `DEEPRETINA_REFERENCE_GOLDENS` to the existing
  Drive/HRF golden directory (rebuild via
  `reports/legacy/matlab-av-profile/generate_real_goldens.m` and its documented
  builder inputs). No image/model is inferred by this operator probe.

Use `-singleCompThread`, nice10 and CUDA hidden. Generators write external MAT
files. Python comparison tools accept `--oracle-dir SCRATCH`; they are reference
analysis utilities, not public prediction commands. `package_synthetic_fixtures.py`
retains only the four half-factor/two synthetic-chain inputs and masks/stored RGB;
real photographs and full intermediate arrays are never packaged.

For the one numerical-sensitivity reproduction, choose a fresh scratch/report
location and the exact safe bundled checkpoint directory. First prepare compact
RGB inputs with `prepare_diaret_sensitivity.py --oracle-dir SCRATCH`, then:

```bash
CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 \
  nice -n10 uv run python reports/legacy/matlab-source-auto/diaret_av_sensitivity.py \
  vessels --oracle-dir SCRATCH --model-root SAFE_MODEL_DIRECTORY

CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 \
  nice -n10 uv run python reports/legacy/matlab-source-auto/run_sensitivity_bounded.py \
  --oracle-dir SCRATCH --model-root SAFE_MODEL_DIRECTORY
```

The first command predicts the dedicated original vessel mask from the same RAW
image at rawPython profile/50°/25px-per-degree, strictlogit>0, nearest restore;
then source-nearest resize supplies the identical fourth plane for both AV
inputs. The second runs the unchanged full AV forward twice with a6GiB sampled-RSS
stop and300s limit; resource receipt also records kernel child peakRSS. No dummy
mask or downsampling substitutes for a resource failure. The reports contain
actual safe checkpoint SHA256 and original provenance, input/support hashes,
standardization/logit/probability deltas and gated class counts.

`source-auto-chain-comparison.json` preserves every raw chain's numerical errors.
`diaret-stage-comparison.json` and`diaret_gaussian_domain.log` isolate the first
operator discrepancy. `sobel-operator-comparison.json` preserves all negative
candidate results; none was installed as an unverified fix.
`diaret-av-sensitivity.json` and`sensitivity-resource-v3.json` describe a single
paired numerical study, not model quality or universal parity.
