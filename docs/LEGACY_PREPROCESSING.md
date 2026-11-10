# Inputs for the original trained models

The original `single_net` models and the four-channel VesClass artery/vein model
have different input contracts. This module preserves those contracts separately;
it does not replace them with the newer fixed-size training recipe.

## Original single_net models

`prepare_single_net(rgb, dataset=None, fov=None, field_of_view_degrees=None,
pixels_per_degree=25, normalized=False)` returns a `PreparedImage` containing
uint8 HWC `image`, boolean HW `fov`, reversible `geometry`, preprocessing metadata,
and `.tensor()` producing the exact original CHW float32 input:

```python
input_tensor = prepared.image.astype(np.float32) / 255 - 0.5
```

The tensor range is `[-0.5,0.5]`. It is not scaled by 0.5 again, standardized per
channel, or normalized with ImageNet statistics.

The implementation reproduces `LoaderGeneric.preprocess` from source commit
`7a152c31c2947ddf760716dfa8a6dfc4d2b881c9`:

1. Read RGB uint8 in the original image frame. For UoA-DR, reproduce the old
   `zero_region` timestamp removal before FOV detection.
2. Use the supplied original-frame FOV mask, or the original automatic detector.
3. Find the **inclusive** FOV bounding box `(row_min,row_max,col_min,col_max)`.
4. Compute `scale = pixels_per_degree * field_of_view_degrees / max(original_rgb.shape)`.
   The denominator is the original image size, **before** cropping. It is not the
   cropped FOV diameter and is not a 1024-pixel cap.
5. Crop RGB/FOV inclusively, then use `skimage.transform.rescale`. RGB uses the
   original default linear interpolation/antialiasing with `preserve_range=True`,
   followed by uint8 truncation. FOV and categorical masks use nearest neighbour.
6. Optionally apply the original Python local-contrast/CLAHE routine.
7. Convert the resulting uint8 RGB to CHW float32 `/255 - 0.5` for the network.

| Original dataset | Field of view |
|---|---:|
| AVRDB, CHASEDB1, Drishti-GS |30°|
| STARE |35°|
| DRIVE/RITE, DRHAGIS, G1020, IOSTAR, ONHSD, REFUGE, RIGA, UoA-DR |45°|
| ARIA |50°|
| HRF |60°|

For another camera, supply its calibrated field angle explicitly. The generic
fallback is 45°, which is recorded in the returned object.

The automatic detector uses mean RGB hysteresis thresholds 7/15, binary closing
with disk 10, erosion with disk 6, the largest connected component, then hole filling.
ARIA uses 1/15 and retries 58/80 when less than 5% of the frame is black; STARE uses
35/70. DRIVE/RITE, HRF, DRHAGIS and IOSTAR originally load official FOV masks.
Supply those masks for dataset reproduction. If absent, the helper exposes an
explicit `fov_source` indicating that it estimated a fallback mask.

### Raw versus normalized source data

`normalized=False` is the source-inferred default: the saved Config source points
to `../data_25.hdf5`. The optional normalization-building script instead writes
`data_25_normalized.hdf5`. The historical checkpoints do not serialize the inherited
Config class's `dataset_fname`; its value therefore cannot be automatically
recovered from the checkpoint instance. Choosing normalization is an explicit
option, not a claimed detection from the weights.

With `normalized=True`, the exact original Python sequence is:

- Erode the FOV by a SciPy disk 1 operation; compute outside Euclidean distance.
- Extend boundary colors with the original Sobel-distance coordinate remapping and
  OpenCV bilinear remap.
- Estimate illumination with Gaussian `sigma=80*pixels_per_degree/55` and apply
  `(image-background)/background + 0.5`, clipping to `[0,1]`.
- Convert to HSV and apply scikit-image CLAHE to the value channel with
  `kernel_size=250*pixels_per_degree/55`, `clip_limit=0.005`.
- Convert back to RGB, zero outside FOV, multiply by 255, clip, and truncate to uint8.

A zero illumination denominator is reported clearly instead of emitting nonfinite
model inputs. The raw path leaves the original camera pixels unchanged outside
FOV; it does not add the normalized path's explicit black mask.

### Restoring geometry

`ImageGeometry` records `original_shape`, `crop_position`, `rescale_factor` and
`prepared_shape`. `geometry.prepare_mask(original_hw_mask)` applies the same crop
and nearest-neighbour scale to source categorical labels. This is useful for
comparing model outputs with original annotations without changing the image grid.

`geometry.restore(prediction, order=0)` supports HW, HWC and CHW categorical maps;
`order=1` supports continuous probability/logit maps. Restoration resizes to the
original crop dimensions and pastes into the original full frame. Outside the crop
is filled with zero, or an explicit `fill` value. It never stretches a cropped
prediction across the whole original photograph. FOV masking is a separate caller
policy; the geometry helper does not silently modify model logits.

## Original four-channel VesClass model

The original `Network.Predict` reads four already prepared DICOM planes in
**R,G,B,vessel** order, casts `pixel_array` directly to float32, and passes the
full image to the network. It does not apply DICOM rescale slope/intercept,
windowing, `/255`, offsets or per-tile intensity normalization.

`load_vesclass_dicoms([r_path,g_path,b_path,vessel_path])` reproduces this loading
operation and returns contiguous CHW float32. All four planes must be finite,
single-frame 2D arrays with identical geometry. Stored vessel levels are preserved
exactly in this DICOM path.

`stack_vesclass_inputs(preprocessed_rgb, vessel_mask)` supports existing HWC RGB
arrays without changing their stored RGB units. Integer PNG vessel masks are
binarized at `>0`; floating masks must already contain only 0/1 values. Probability
maps must be thresholded explicitly before classification.

The AV network performs its own global full-image per-channel standardization.
That operation belongs to the original architecture, not to this preprocessing
helper. Full-image normalization also means that independently normalized tiles
are not equivalent to the original prediction path.

The historical MATLAB DICOM export uses `uint16(round(I*4096))`, including 4096 at
white, while the newer uint8/NIfTI export uses `uint8(round(I*255))`.
`quantize_vesclass_rgb(normalized_rgb, storage="dicom12"|"uint8")` reproduces the
positive half-up rounding and these scale factors for RGB already normalized to
`[0,1]`. It does not claim to reproduce the preceding MATLAB image-processing steps.

### MATLAB and Python normalization are different references

The original MATLAB AV `image_adjustment` retains the **full frame**, with scale
`pixels_per_degree * field_angle / (2 * estimated_radius)`. Its later nearest
resize back to the original frame is appropriate for those full-frame DICOM inputs.
It is not an inverse for `single_net`'s cropped inputs.

The MATLAB normalization uses `imgaussfilt` and `adapthisteq` with computed tile
counts; the supplied Python version uses scikit-image Gaussian and kernel-size
CLAHE. They are not numerically interchangeable by assertion. No original retinal
HDF5/prepared DICOM input archive or MATLAB runtime was found in the available
workspace, so MATLAB pixel parity is not certified. The stored-DICOM tensor loading
is tested exactly; any standalone raw RGB AV reproduction must identify its Python
preprocessing provenance and retain the full-frame geometry or invert its own crop.

### Explicit raw RGB AV proxy

`prepare_vesclass_raw_proxy(rgb, vessel_mask, dataset=None, fov=None,
field_of_view_degrees=None, pixels_per_degree=25, fov_diameter_pixels=None,
storage="dicom12", normalized=True)` is an explicitly selected **Python proxy**
for situations where old prepared DICOM planes are unavailable. Its metadata sets
`matlab_parity_certified=False`; it is never silently presented as native MATLAB
preprocessing.

The proxy retains the **entire original frame**. It scales by
`pixels_per_degree * field_angle / fov_diameter_pixels`, rounding output frame
dimensions upward as in the original MATLAB scale-based image resize. Supply the
original-frame calibrated FOV diameter when known. Otherwise the maximum inclusive
FOV bounding-box extent is used and marked as a heuristic, distinct from MATLAB's
edge/radius estimator. An official FOV mask is preferred; automatic Python FOV
estimation is also marked as a heuristic.

RGB uses explicitly identified Python linear/antialiasing resampling; vessel/FOV
masks use nearest neighbour. Python local contrast and HSV CLAHE operate on floating
RGB **before** positive half-up 12-bit storage quantization, retaining 4096 at white.
The result exposes `image`/`stored_rgb`, `vessel`, `fov`, full-frame `geometry`,
`profile`/`metadata`, `.inputs()` returning float32 CHW4 stored units, and `.tensor()`.
Integer vessel masks are binarized; floating probability maps require an explicit
threshold before this helper.

The original AV prediction wrapper can restore proxy class maps with its normal
full-frame OpenCV nearest resize using the original image size. Do not replace
this profile with a cropped single_net image and then stretch that crop over the
whole photograph. Neither successful shape/finite checks nor the shared Python
normalization establish MATLAB pixel parity or a new accuracy result.

## Executed numeric checks

The original `LoaderGeneric.preprocess` and original Python
`local_contrast_and_clahe` functions were executed directly from Git source. Only
the image/FOV reader and HDF5 output sink were replaced by in-memory fixtures;
the original geometry and normalization method bodies ran unchanged. The saved
uncommitted patch's three relevant preprocessing files differ only in line endings.

**Twelve comparisons passed bitwise:** raw and normalized versions of actual
DRIVE, HRF, STARE, Drishti-GS and UoA-DR images, plus two deterministic synthetic
fixtures. Prepared RGB bytes, FOV bytes, inclusive crop/scale metadata and final
CHW network-input tensors matched exactly (maximum RGB difference zero).
Source hashes, input/output hashes, dimensions and runtime versions are recorded
in `reports/legacy/preprocessing-reference.json`.

Twelve portable regression cases cover source-derived numeric goldens, original-size
scale calculation, forward/inverse crop geometry, automatic FOV behavior, UoA
legacy timestamp handling, AV stored pixel units, MATLAB rounding, actual DICOM
loading without rescale/windowing, raw AV full-frame/binary-mask/calibration behavior,
and invalid-input handling. The DICOM test
includes nontrivial slope/intercept metadata to verify that those values remain
unapplied, exactly as in the original code.

These checks establish parity with the original Python source under the locked
runtime (NumPy 2.5.4, SciPy 1.18.1, scikit-image 0.26.0, OpenCV 5.0.0, h5py 3.16.0,
Torch 2.6.0). They do not certify an unavailable 2023 preprocessing environment or
MATLAB output bytes. Keep the supplied `uv.lock` when reproducing these checks;
changing morphology/interpolation/CLAHE libraries requires repeating the goldens.


After sharing the floating normalization core with the AV proxy, all twelve
source-executed comparisons were repeated and remained bitwise identical. A real
DRIVE21 training photograph and its original vessel/FOV masks also passed the
raw-proxy finite/range/binary/full-frame checks; its proxy metadata and inspection
overlay are `reports/legacy/raw-av-proxy-sanity.json` and
`reports/legacy/raw-av-proxy-drive21.png`. The proxy check is a functionality check,
not a certified MATLAB comparison.

## Exact automatic-FOV CPU optimization

Automatic FOV closing and erosion use cross decompositions of the historical
radius-10 and radius-6 disks. Their reconstructed footprints are bitwise equal
to the original disks. The illumination-correction radius-1 footprint is
unchanged. This avoids scanning the full large disk at every original pixel.

On five actual G1020 photographs, automatic FOV computation took 14.13 seconds
with full disks and 7.47 seconds with decomposed disks (1.89× aggregate speedup).
All five resulting FOVs were bitwise identical. The measurement covers this CPU
stage on the recorded local runtime; it is not a model-quality or whole-pipeline
speed claim. Evidence is in
`reports/legacy/fov-exact-footprint-timing.json`.

Both kernels are checked explicitly. Equivalence tests cover tiny, empty, full,
random and boundary-touching masks, including the combined closing/erosion
sequence. The twelve original-source preprocessing comparisons also passed with
the optimization: RGB, FOV, tensor and geometry remained bitwise equal for raw
and normalized input profiles. An already-running full-data preparation retains
the code it loaded at startup; its arrays are unaffected by this later change.
