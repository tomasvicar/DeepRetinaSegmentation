"""Keep each exported prediction in its own directory."""

from pathlib import Path


def check_prediction_output(output: Path) -> Path:
    """Accept a new or empty directory without creating or modifying it.

    Reusing a populated directory can mix masks from different model families.
    In particular, a vessel-only prediction must not leave an older cup mask.
    """
    output = Path(output)
    if output.is_symlink() or (
        output.exists() and (not output.is_dir() or any(output.iterdir()))
    ):
        raise FileExistsError(
            f"Prediction output must be a new or empty directory: {output}. "
            "Choose a different --output for each prediction."
        )
    return output
