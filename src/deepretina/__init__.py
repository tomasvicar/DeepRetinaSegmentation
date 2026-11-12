"""Retinal segmentation, from public datasets to predictions."""

__version__ = "0.1.0"


def predict_original(image, **kwargs):
    """Lazily call the fixed four-component original-model prediction API."""
    from .original import predict_original as predict

    return predict(image, **kwargs)


def predict_bundled_unified(model, image_path, output, **kwargs):
    """Lazy bundled unified prediction with optional per-model inference defaults."""
    from .bundled_inference import predict_bundled_unified as predict

    return predict(model, image_path, output, **kwargs)
