"""Retinal segmentation, from public datasets to predictions."""

__version__ = "0.1.0"


def predict_original(image, **kwargs):
    """Lazily call the fixed four-component original-model prediction API."""
    from .original import predict_original as predict

    return predict(image, **kwargs)
