"""Read-only indexed access to ZIP, TAR and directory dataset resources.

Archive exports sometimes include the same resource under both a shared-drive
prefix and a short path. Canonical paths remove that transport prefix; conflicting
copies are rejected using their archive CRC/size identity.
"""

from __future__ import annotations

import gzip
import hashlib
import io
import tarfile
import unicodedata
import zipfile
from contextlib import ExitStack
from pathlib import Path

import numpy as np
from PIL import Image


def canonical_path(name: str) -> str:
    name = unicodedata.normalize("NFC", name.replace("\\", "/")).lstrip("/")
    marker = "/data/databases/"
    pos = name.lower().find(marker)
    if pos >= 0:
        name = name[pos + len(marker) :]
    return name


class Source:
    """An indexed source; paths passed to ``bytes``/``read`` never write to it."""

    def __init__(self, path: Path):
        self.path = Path(path)
        self._stack = ExitStack()
        self.zip = None
        self.tar = None
        self._entries = {}
        self._identities = {}
        if self.path.is_dir():
            for file in sorted(self.path.rglob("*")):
                if file.is_file():
                    self._add(file.relative_to(self.path).as_posix(), file)
        elif self.path.is_file() and zipfile.is_zipfile(self.path):
            self.zip = self._stack.enter_context(zipfile.ZipFile(self.path))
            for info in self.zip.infolist():
                if not info.is_dir():
                    self._add(info.filename, info, (info.file_size, info.CRC))
        elif self.path.is_file() and tarfile.is_tarfile(self.path):
            self.tar = self._stack.enter_context(tarfile.open(self.path))  # noqa: SIM115 -- owned by ExitStack
            for info in self.tar.getmembers():
                if info.isfile():
                    self._add(info.name, info, (info.size,))
        else:
            raise FileNotFoundError(f"Expected dataset directory, ZIP or TAR archive: {self.path}")
        self.names = sorted(self._entries)

    def _add(self, name, entry, identity=None):
        name = canonical_path(name)
        if "__MACOSX" in name or any(part.startswith(".") for part in Path(name).parts):
            return
        if name in self._entries:
            conflict = identity is not None and self._identities[name] != identity
            if self.tar is not None and not conflict:

                def digest(member):
                    with self.tar.extractfile(member) as stream:
                        return hashlib.file_digest(stream, "sha256").digest()

                conflict = digest(self._entries[name]) != digest(entry)
            if conflict:
                self.close()
                raise ValueError(f"Conflicting duplicate resource {name!r} in {self.path}")
            return
        self._entries[name] = entry
        self._identities[name] = identity

    def find(self, suffix: str, *, optional: bool = False) -> str | None:
        suffix = canonical_path(suffix).lower()
        exact = [name for name in self.names if name.lower() == suffix]
        matches = exact or [name for name in self.names if name.lower().endswith("/" + suffix)]
        if not matches:
            if optional:
                return None
            raise FileNotFoundError(f"Missing {suffix!r} in {self.path}")
        if len(matches) > 1:
            # Some releases repeat a complete dataset directory inside another
            # copy of itself. Accept aliases only after checking resource identity.
            if self.zip is not None:
                identities = {self._identities[name] for name in matches}
                identical = len(identities) == 1
            else:
                identical = (
                    len(
                        {
                            hashlib.sha256(self._read_entry(self._entries[name])).digest()
                            for name in matches
                        }
                    )
                    == 1
                )
            if identical:
                return min(matches, key=lambda name: (len(name), name))
            raise ValueError(f"Ambiguous resource {suffix!r} in {self.path}: {matches[:5]}")
        return matches[0]

    def images(self, folder: str, extension: str) -> list[str]:
        folder = folder.lower().strip("/")
        return sorted(
            {
                Path(name).name
                for name in self.names
                if ("/" + folder + "/") in ("/" + name.lower())
                and name.lower().endswith(extension.lower())
            }
        )

    def bytes(self, path: str) -> bytes:
        name = self.find(path)
        return self._read_entry(self._entries[name])

    def _read_entry(self, entry) -> bytes:
        if self.zip is not None:
            return self.zip.read(entry)
        if self.tar is not None:
            with self.tar.extractfile(entry) as stream:
                return stream.read()
        return entry.read_bytes()

    def text(self, path: str) -> str:
        data = self.bytes(path)
        if path.lower().endswith(".gz"):
            data = gzip.decompress(data)
        try:
            return data.decode("utf-8-sig")
        except UnicodeDecodeError:
            return data.decode("cp1252")

    def read(self, path: str, mode: str = "L") -> np.ndarray:
        data = self.bytes(path)
        if path.lower().endswith(".gz"):
            data = gzip.decompress(data)
        with Image.open(io.BytesIO(data)) as image:
            return np.asarray(image.convert(mode)).copy()

    def close(self):
        self._stack.close()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()

    def identity(self) -> dict:
        """Archive member CRCs or filesystem file stats; no whole-dataset rehash."""
        stat = self.path.stat()
        members = []
        for name, entry in sorted(self._entries.items()):
            if self.zip is not None:
                members.append({"name": name, "size": entry.file_size, "crc32": entry.CRC})
            elif self.tar is not None:
                members.append({"name": name, "size": entry.size})
            else:
                file_stat = entry.stat()
                members.append(
                    {"name": name, "size": file_stat.st_size, "mtime_ns": file_stat.st_mtime_ns}
                )
        return {
            "path": str(self.path.resolve()),
            "size": stat.st_size,
            "mtime_ns": stat.st_mtime_ns,
            "kind": "directory" if self.path.is_dir() else "archive",
            "members": members,
        }
