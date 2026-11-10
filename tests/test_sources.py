"""Resource identity and read-only archive behavior."""

import gzip
import io
import tarfile
import zipfile

import numpy as np
import pytest
from PIL import Image

from deepretina.sources import Source


def _png():
    stream = io.BytesIO()
    Image.fromarray(np.full((9, 13, 3), 100, np.uint8)).save(stream, format="PNG")
    return stream.getvalue()


def test_zip_transport_prefixes_deduplicate_without_changing_source(tmp_path):
    path = tmp_path / "data.zip"
    with zipfile.ZipFile(path, "w") as archive:
        archive.writestr("content/drive/Data/databases/DB/image.png", _png())
        archive.writestr("DB/image.png", _png())
        archive.writestr("DB/labels.txt", "x,y\n1,2\n")
    previous = path.read_bytes()
    with Source(path) as source:
        assert source.names == ["DB/image.png", "DB/labels.txt"]
        assert source.read("image.png", "RGB").shape == (9, 13, 3)
        assert source.text("labels.txt") == "x,y\n1,2\n"
        assert source.identity()["members"][0]["crc32"] > 0
    assert path.read_bytes() == previous


def test_conflicting_zip_copies_raise_instead_of_selecting_labels(tmp_path):
    path = tmp_path / "data.zip"
    with zipfile.ZipFile(path, "w") as archive:
        archive.writestr("content/drive/Data/databases/DB/labels.txt", "artery")
        archive.writestr("DB/labels.txt", "vein")
    with pytest.raises(ValueError, match="Conflicting duplicate"):
        Source(path)


def test_ambiguous_basename_requires_qualified_pairing(tmp_path):
    (tmp_path / "a").mkdir()
    (tmp_path / "b").mkdir()
    (tmp_path / "a/labels.txt").write_text("first")
    (tmp_path / "b/labels.txt").write_text("second")
    with Source(tmp_path) as source:
        with pytest.raises(ValueError, match="Ambiguous"):
            source.find("labels.txt")
        assert source.text("a/labels.txt") == "first"
        assert source.find("missing", optional=True) is None


def test_tar_and_gzipped_images_read_without_extraction(tmp_path):
    path = tmp_path / "source.tar.gz"
    payload = gzip.compress(_png())
    with tarfile.open(path, "w:gz") as archive:
        info = tarfile.TarInfo("DB/image.png.gz")
        info.size = len(payload)
        archive.addfile(info, io.BytesIO(payload))
    before = set(tmp_path.iterdir())
    with Source(path) as source:
        assert source.read("DB/image.png.gz", "RGB").shape == (9, 13, 3)
    assert set(tmp_path.iterdir()) == before


def test_identical_nested_release_aliases_are_checked_before_resolution(tmp_path):
    path = tmp_path / "source.zip"
    with zipfile.ZipFile(path, "w") as archive:
        archive.writestr("DRHAGIS/Fundus_Images/1.jpg", _png())
        archive.writestr("DRHAGIS/DRHAGIS/Fundus_Images/1.jpg", _png())
    with Source(path) as source:
        assert source.find("Fundus_Images/1.jpg") == "DRHAGIS/Fundus_Images/1.jpg"
        assert source.read("Fundus_Images/1.jpg", "RGB").shape == (9, 13, 3)
