"""Synthetic IO/identity guards only; no models, Torch or real annotation inputs."""

import hashlib
import importlib.util
import io
import struct
import unittest
import zipfile
from pathlib import Path

from PIL import Image

BASE = Path(__file__).parent
spec = importlib.util.spec_from_file_location("cohort_builder", BASE / "build_cohort.py")
builder = importlib.util.module_from_spec(spec)
spec.loader.exec_module(builder)


class CohortTests(unittest.TestCase):
    def test_decoded_identity_binds_shape_prefix_and_RGB_pixels(self):
        rgb = Image.new("RGB", (3, 2), color=(1, 2, 3))
        expected = hashlib.sha256(struct.pack("<qqq", 2, 3, 3) + rgb.tobytes()).hexdigest()
        self.assertEqual(builder.photo_identity(rgb), "sha256:" + expected)
        self.assertNotEqual(expected, hashlib.sha256(rgb.tobytes()).hexdigest())

    def archive(self, entries):
        stream = io.BytesIO()
        with zipfile.ZipFile(stream, "w") as z:
            for name, data in entries:
                z.writestr(name, data)
        stream.seek(0)
        return zipfile.ZipFile(stream)

    def test_exact_official_FOV_alias_and_no_GT_entry_read(self):
        with self.archive(
            [
                ("mask/04_dr_mask.tif", b"official-FOV"),
                ("content/x/Data/databases/HRF/mask/04_dr_mask.tif", b"official-FOV"),
                ("manual1/04_dr.tif", b"never-read-annotation"),
            ]
        ) as z:
            read, original = [], z.read

            def observed(info):
                read.append(info.filename)
                return original(info)

            z.read = observed
            data, pin = builder.selected_bytes(z, "mask/04_dr_mask.tif")
            self.assertEqual(data, b"official-FOV")
            self.assertEqual(read, ["mask/04_dr_mask.tif"])
            self.assertEqual(pin["transport_alias_count"], 2)

    def test_missing_official_FOV_never_uses_annotation_and_conflict_rejects(self):
        with (
            self.archive([("manual1/04_dr.tif", b"not-FOV")]) as z,
            self.assertRaisesRegex(AssertionError, "Missing"),
        ):
            builder.selected_bytes(z, "mask/04_dr_mask.tif")
        with (
            self.archive(
                [("mask/04_dr_mask.tif", b"official"), ("HRF/mask/04_dr_mask.tif", b"conflicting")]
            ) as z,
            self.assertRaisesRegex(AssertionError, "Conflicting"),
        ):
            builder.selected_bytes(z, "mask/04_dr_mask.tif")


if __name__ == "__main__":
    unittest.main()
