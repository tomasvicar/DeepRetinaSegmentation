"""Input preparation matching the historical models, with reversible geometry.

The single_net path reproduces LoaderGeneric + optional Python local contrast.
VesClass consumes stored preprocessed RGB values and a binary vessel channel;
that input scale is deliberately separate from single_net's /255 - 0.5.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass

import cv2
import numpy as np
from scipy.ndimage import binary_erosion, binary_fill_holes, distance_transform_edt, sobel
from skimage.color import hsv2rgb, rgb2hsv
from skimage.exposure import equalize_adapthist
from skimage.filters import apply_hysteresis_threshold, gaussian
from skimage.measure import label
from skimage.morphology import binary_closing, disk
from skimage.morphology import binary_erosion as sk_binary_erosion
from skimage.transform import rescale, resize

DATASET_FOV_DEGREES = {
    "aria": 50.0,
    "avrdb": 30.0,
    "chasedb1": 30.0,
    "drhagis": 45.0,
    "drishti": 30.0,
    "drive": 45.0,
    "rite": 45.0,
    "g1020": 45.0,
    "hrf": 60.0,
    "iostar": 45.0,
    "onhsd": 45.0,
    "refuge": 45.0,
    "riga": 45.0,
    "stare": 35.0,
    "uoadr": 45.0,
}
OFFICIAL_FOV_DATASETS = frozenset({"drive", "rite", "hrf", "drhagis", "iostar"})


def _rgb(image):
    image = np.asarray(image)
    if image.ndim != 3 or image.shape[2] != 3 or image.dtype != np.uint8:
        raise ValueError("Expected an RGB uint8 image in its original coordinate frame")
    return image


def _mask(fov, shape):
    fov = np.asarray(fov)
    if fov.shape != shape:
        raise ValueError(f"FOV shape {fov.shape} does not match image shape {shape}")
    fov = fov > 0
    if not fov.any():
        raise ValueError("Cannot prepare an image with an empty FOV")
    return fov


def get_fov_auto(image, t1=7, t2=15):
    """Original LoaderGeneric hysteresis/closing/erosion/largest-component FOV."""
    image = _rgb(image)
    average_rgb = np.mean(image, axis=2)
    mask = apply_hysteresis_threshold(average_rgb, t1, t2)
    mask = binary_closing(mask, disk(10))
    mask = sk_binary_erosion(mask, disk(6))
    labelled = label(mask)
    areas = np.bincount(labelled.ravel())[1:]
    if not len(areas):
        raise ValueError("Automatic FOV detection found no foreground component")
    mask = labelled == (areas.argmax() + 1)
    return binary_fill_holes(mask)


def dataset_fov(image, dataset=None):
    """Use original automatic thresholds; official dataset masks remain preferable."""
    name = dataset.lower() if dataset else None
    if name == "aria":
        fov = get_fov_auto(image, t1=1, t2=15)
        if np.sum(fov == 0) < 0.05 * fov.size:
            fov = get_fov_auto(image, t1=58, t2=80)
        return fov
    if name == "stare":
        return get_fov_auto(image, t1=35, t2=70)
    return get_fov_auto(image, t1=7, t2=15)


def get_bbox(fov):
    """Return historical inclusive (row_min,row_max,col_min,col_max) bounds."""
    rows, cols = np.where(np.asarray(fov, dtype=bool))
    if not len(rows):
        raise ValueError("Cannot find a bounding box for an empty FOV")
    return int(rows.min()), int(rows.max()), int(cols.min()), int(cols.max())


def crop_to_bbox(image, bbox):
    row_min, row_max, col_min, col_max = bbox
    return image[row_min : row_max + 1, col_min : col_max + 1, ...]


def _local_contrast_float(image, fov, pixels_per_degree):
    """Shared original Python operations, before the source's uint8 export."""
    fov = _mask(fov, image.shape[:2])
    sigma = 80 / (55 / pixels_per_degree)
    kernel = 250 / (55 / pixels_per_degree)
    image = np.asarray(image, dtype=np.float32)
    eroded = binary_erosion(fov, disk(1))
    distance = distance_transform_edt(eroded == 0)
    sx, sy = sobel(distance, axis=0) / 4, sobel(distance, axis=1) / 4
    xx, yy = np.meshgrid(np.arange(distance.shape[1]), np.arange(distance.shape[0]))
    mapped_x, mapped_y = xx - sy * distance, yy - sx * distance
    interpolated = cv2.remap(
        image, mapped_x.astype(np.float32), mapped_y.astype(np.float32), cv2.INTER_LINEAR
    )
    background = gaussian(interpolated, sigma, channel_axis=2)
    with np.errstate(divide="ignore", invalid="ignore"):
        interpolated = (interpolated - background) / background + 0.5
    interpolated[interpolated < 0] = 0
    interpolated[interpolated > 1] = 1
    if not np.isfinite(interpolated).all():
        raise ValueError("Legacy illumination correction produced a zero/invalid background")
    interpolated = rgb2hsv(interpolated)
    interpolated[:, :, 2] = equalize_adapthist(
        interpolated[:, :, 2], kernel_size=kernel, clip_limit=0.005
    )
    interpolated = hsv2rgb(interpolated)
    interpolated[fov == 0] = 0
    return interpolated


def local_contrast_and_clahe(image, fov, pixels_per_degree=25.0):
    """Original single_net Python normalization, including uint8 truncation.

    This is the Python implementation from the saved source, not a claim of
    numerical equivalence to MATLAB adapthisteq or its Gaussian defaults.
    """
    image = _rgb(image)
    if not math.isfinite(pixels_per_degree) or pixels_per_degree <= 0:
        raise ValueError("pixels_per_degree must be finite and positive")
    interpolated = _local_contrast_float(image.astype(np.float32) / 255, fov, pixels_per_degree)
    interpolated = interpolated * 255
    interpolated[interpolated < 0] = 0
    interpolated[interpolated > 255] = 255
    return interpolated.astype(np.uint8)


@dataclass(frozen=True)
class ImageGeometry:
    original_shape: tuple[int, int]
    crop_position: tuple[int, int, int, int]
    rescale_factor: float
    prepared_shape: tuple[int, int]

    @property
    def bbox(self):
        return self.crop_position

    def as_dict(self):
        return asdict(self)

    def prepare_mask(self, mask):
        """Apply the original crop/scale to an original-frame categorical HW mask."""
        mask = np.asarray(mask)
        if mask.shape != self.original_shape:
            raise ValueError(
                f"Mask shape {mask.shape} does not match original {self.original_shape}"
            )
        cropped = crop_to_bbox(mask, self.crop_position)
        prepared = rescale(
            cropped, self.rescale_factor, order=0, preserve_range=True, anti_aliasing=False
        )
        if prepared.shape != self.prepared_shape:
            # Full-frame AV proxies explicitly preserve MATLAB's ceil-sized
            # frame. Their nominal scale can differ from shape ratios by1pixel.
            prepared = resize(
                cropped, self.prepared_shape, order=0, preserve_range=True, anti_aliasing=False
            )
        return prepared.astype(mask.dtype)

    def restore(self, array, *, order=0, fill=0):
        """Resize HW, HWC or CHW predictions into the original inclusive crop.

        Use order0 for categorical maps, order1 for probabilities/logits. Values
        outside the original crop receive fill. No whole-frame stretch occurs.
        """
        if order not in (0, 1):
            raise ValueError("Restoration supports categorical order0 or continuous order1")
        array = np.asarray(array)
        channel_first = False
        if (
            array.ndim == 2
            and array.shape == self.prepared_shape
            or array.ndim == 3
            and array.shape[:2] == self.prepared_shape
        ):
            pass
        elif array.ndim == 3 and array.shape[1:] == self.prepared_shape:
            channel_first = True
            array = np.moveaxis(array, 0, -1)
        else:
            raise ValueError(f"Prediction shape {array.shape} does not match {self.prepared_shape}")
        row_min, row_max, col_min, col_max = self.crop_position
        crop_shape = (row_max - row_min + 1, col_max - col_min + 1)
        shape = crop_shape + (() if array.ndim == 2 else (array.shape[2],))
        if order and not np.issubdtype(array.dtype, np.floating):
            array = array.astype(np.float32)
        restored_crop = resize(array, shape, order=order, preserve_range=True, anti_aliasing=False)
        if order == 0:
            restored_crop = restored_crop.astype(array.dtype)
        shape = self.original_shape + (() if array.ndim == 2 else (array.shape[2],))
        result = np.full(shape, fill, dtype=restored_crop.dtype)
        result[row_min : row_max + 1, col_min : col_max + 1] = restored_crop
        return np.moveaxis(result, -1, 0) if channel_first else result


@dataclass(frozen=True)
class PreparedImage:
    image: np.ndarray
    fov: np.ndarray
    geometry: ImageGeometry
    normalized: bool
    fov_source: str
    pixels_per_degree: float
    field_of_view_degrees: float

    def tensor(self):
        """Original single_net CHW float32 input, in [-0.5,0.5]."""
        import torch

        image = self.image.astype(np.float32) / 255 - 0.5
        return torch.from_numpy(np.transpose(image, (2, 0, 1)).copy())


def prepare_single_net(
    image,
    dataset=None,
    fov=None,
    field_of_view_degrees=None,
    pixels_per_degree=25.0,
    normalized=False,
):
    """Reproduce single_net LoaderGeneric geometry and optional normalization.

    The scale denominator is the ORIGINAL maximum image dimension, before the
    inclusive FOV crop. It is neither a1024 cap nor the cropped FOV diameter.
    Official DRIVE/HRF/DRHAGIS/IOSTAR masks should be supplied when reproducing
    their saved training data; automatic fallback is recorded explicitly.
    """
    image = _rgb(image)
    dataset = dataset.lower() if dataset else None
    if dataset == "uoadr":
        # LoaderUoa_dr.read_img_fcn zero_region applied before FOV/crop/scale.
        # Reproducing old inputs differs deliberately from the clean-data adapter,
        # which preserves photographs and only masks timestamp supervision.
        image = image.copy()
        height = image.shape[0]
        image[round(0.97 * height) :, : round(0.25 * height)] = 0
    if field_of_view_degrees is None:
        if dataset is not None and dataset not in DATASET_FOV_DEGREES:
            raise ValueError(
                f"Unknown dataset {dataset!r}; provide field_of_view_degrees explicitly"
            )
        field_of_view_degrees = DATASET_FOV_DEGREES.get(dataset, 45.0)
    if not all(math.isfinite(x) and x > 0 for x in (pixels_per_degree, field_of_view_degrees)):
        raise ValueError("Pixels/degree and field of view must be finite and positive")
    if fov is None:
        fov = dataset_fov(image, dataset)
        fov_source = (
            "automatic fallback; original dataset uses an official mask"
            if dataset in OFFICIAL_FOV_DATASETS
            else "automatic original Python thresholds"
        )
    else:
        fov = _mask(fov, image.shape[:2])
        fov_source = "provided original-frame mask"
    bbox = get_bbox(fov)
    scale = pixels_per_degree * field_of_view_degrees / np.max(image.shape)
    original_shape = image.shape[:2]
    prepared = rescale(
        crop_to_bbox(image, bbox), scale, channel_axis=2, preserve_range=True
    ).astype(np.uint8)
    prepared_fov = rescale(crop_to_bbox(fov, bbox), scale, order=0)
    if normalized:
        prepared = local_contrast_and_clahe(prepared, prepared_fov, pixels_per_degree)
    geometry = ImageGeometry(original_shape, bbox, float(scale), prepared.shape[:2])
    return PreparedImage(
        prepared,
        prepared_fov,
        geometry,
        bool(normalized),
        fov_source,
        float(pixels_per_degree),
        float(field_of_view_degrees),
    )


def stack_vesclass_inputs(preprocessed_rgb, vessel_mask):
    """Stack legacy stored RGB pixel values and a0/1 vessel channel as CHW.

    No /255, offset, ImageNet normalization or per-tile standardization occurs.
    The historical AV network applies its own full-image channel normalization.
    Float masks must already be binary; integer0/255 masks are binarized.
    """
    rgb, mask = np.asarray(preprocessed_rgb), np.asarray(vessel_mask)
    if rgb.ndim != 3 or rgb.shape[2] != 3 or mask.shape != rgb.shape[:2]:
        raise ValueError("Expected preprocessed RGB HWC and matching binary vessel HW")
    if not np.isfinite(rgb).all() or not np.isfinite(mask).all():
        raise ValueError("VesClass inputs must be finite")
    if np.issubdtype(mask.dtype, np.floating) and not np.isin(mask, (0.0, 1.0)).all():
        raise ValueError("Threshold vessel probabilities explicitly before VesClass prediction")
    return np.ascontiguousarray(
        np.concatenate(
            [np.transpose(rgb.astype(np.float32), (2, 0, 1)), (mask > 0)[None].astype(np.float32)],
            axis=0,
        )
    )


def quantize_vesclass_rgb(normalized_rgb, storage="dicom12"):
    """Historical positive MATLAB rounding for already normalized RGB in [0,1].

    dicom12 uses round(I*4096), including4096 at white; the newer uint8 export
    uses round(I*255). This function does not substitute a raw preprocessing path.
    """
    image = np.asarray(normalized_rgb)
    if image.ndim != 3 or image.shape[2] != 3 or not np.isfinite(image).all():
        raise ValueError("Expected finite normalized RGB HWC")
    if (image < 0).any() or (image > 1).any():
        raise ValueError("Normalized RGB values must be in [0,1]")
    if storage == "dicom12":
        return np.floor(image * 4096 + 0.5).astype(np.uint16)
    if storage == "uint8":
        return np.floor(image * 255 + 0.5).astype(np.uint8)
    raise ValueError("Choose historical storage dicom12 or uint8")


def load_vesclass_dicoms(paths):
    """Read R,G,B,vessel DICOM planes exactly as original Network.Predict did.

    Stored pixel values are stacked as float32 without rescale slope/intercept,
    windowing, intensity conversion, offsets, thresholds or image resizing.
    The four paths must be supplied in R,G,B,vessel order.
    """
    import pydicom

    paths = tuple(paths)
    if len(paths) != 4:
        raise ValueError("Supply exactly four DICOM planes in R,G,B,vessel order")
    planes = []
    for path in paths:
        image = pydicom.dcmread(path).pixel_array.astype(np.float32)
        if image.ndim != 2 or not np.isfinite(image).all():
            raise ValueError(f"Expected a finite single-frame 2D DICOM plane: {path}")
        if planes and image.shape != planes[0].shape:
            raise ValueError("All four legacy DICOM planes must have identical dimensions")
        planes.append(image)
    return np.ascontiguousarray(np.stack(planes))


@dataclass(frozen=True)
class PreparedVesClassProxy:
    image: np.ndarray
    vessel: np.ndarray
    fov: np.ndarray
    geometry: ImageGeometry
    profile: dict

    @property
    def stored_rgb(self):
        return self.image

    @property
    def metadata(self):
        return self.profile

    def inputs(self):
        return stack_vesclass_inputs(self.image, self.vessel)

    def tensor(self):
        import torch

        return torch.from_numpy(self.inputs())


def prepare_vesclass_raw_proxy(
    image,
    vessel_mask,
    dataset=None,
    fov=None,
    field_of_view_degrees=None,
    pixels_per_degree=25.0,
    fov_diameter_pixels=None,
    storage="dicom12",
    normalized=True,
):
    """Explicit full-frame Python proxy for raw-image VesClass prediction.

    This is NOT certified MATLAB preprocessing. Native prepared DICOM input is
    the exact reference. Unlike single_net, this path never crops the FOV bbox.
    """
    image = _rgb(image)
    dataset = dataset.lower() if dataset else None
    if field_of_view_degrees is None:
        if dataset is not None and dataset not in DATASET_FOV_DEGREES:
            raise ValueError(
                f"Unknown dataset {dataset!r}; provide field_of_view_degrees explicitly"
            )
        field_of_view_degrees = DATASET_FOV_DEGREES.get(dataset, 45.0)
    if not all(math.isfinite(x) and x > 0 for x in (pixels_per_degree, field_of_view_degrees)):
        raise ValueError("Pixels/degree and field of view must be finite and positive")
    vessel_mask = np.asarray(vessel_mask)
    if vessel_mask.shape != image.shape[:2] or not np.isfinite(vessel_mask).all():
        raise ValueError("Provide a finite original-frame binary vessel mask matching RGB")
    if np.issubdtype(vessel_mask.dtype, np.floating) and not np.isin(vessel_mask, (0.0, 1.0)).all():
        raise ValueError("Threshold vessel probabilities explicitly before preparing the AV proxy")
    vessel_mask = vessel_mask > 0
    if fov is None:
        fov = dataset_fov(image, dataset)
        fov_source = "Python automatic FOV heuristic; not original MATLAB souradnice"
    else:
        fov = _mask(fov, image.shape[:2])
        fov_source = "provided original-frame FOV"
    if fov_diameter_pixels is None:
        r0, r1, c0, c1 = get_bbox(fov)
        fov_diameter_pixels = float(max(r1 - r0 + 1, c1 - c0 + 1))
        diameter_source = "maximum FOV bbox extent heuristic; not MATLAB edge/radius estimator"
    else:
        diameter_source = "explicit original-frame diameter calibration"
    if not math.isfinite(fov_diameter_pixels) or fov_diameter_pixels <= 0:
        raise ValueError("fov_diameter_pixels must be finite and positive")
    scale = pixels_per_degree * field_of_view_degrees / fov_diameter_pixels
    height, width = image.shape[:2]
    prepared_shape = (max(1, math.ceil(height * scale)), max(1, math.ceil(width * scale)))
    float_rgb = resize(
        image.astype(np.float32) / 255,
        (*prepared_shape, 3),
        order=1,
        preserve_range=True,
        anti_aliasing=scale < 1,
    )
    prepared_fov = resize(
        fov, prepared_shape, order=0, preserve_range=True, anti_aliasing=False
    ).astype(bool)
    prepared_vessel = resize(
        vessel_mask, prepared_shape, order=0, preserve_range=True, anti_aliasing=False
    ).astype(bool)
    if normalized:
        float_rgb = _local_contrast_float(float_rgb, prepared_fov, pixels_per_degree)
    stored_rgb = quantize_vesclass_rgb(np.clip(float_rgb, 0, 1), storage)
    geometry = ImageGeometry(
        (height, width), (0, height - 1, 0, width - 1), float(scale), prepared_shape
    )
    profile = {
        "name": "vesclass_raw_python_proxy",
        "matlab_parity_certified": False,
        "geometry": "full frame; no FOV crop; ceil-sized output",
        "fov_source": fov_source,
        "fov_diameter_pixels": float(fov_diameter_pixels),
        "diameter_source": diameter_source,
        "pixels_per_degree": float(pixels_per_degree),
        "field_of_view_degrees": float(field_of_view_degrees),
        "photometric": "original Python local-contrast/HSV-CLAHE in float before storage quantization"
        if normalized
        else "raw RGB; explicit normalization bypass",
        "rgb_resampling": "Python linear/antialiasing proxy, not certified MATLAB bicubic",
        "mask_resampling": "nearest neighbour",
        "storage": storage,
        "output_restore": "original AV full-frame INTER_NEAREST resize, not single_net crop restoration",
    }
    return PreparedVesClassProxy(stored_rgb, prepared_vessel, prepared_fov, geometry, profile)
