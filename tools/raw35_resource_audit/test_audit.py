"""Only tiny synthetic encoded resources, directory metadata and mocked unit facts."""

import ast
import hashlib
import importlib.util
import io
import json
import stat
import sys
import zipfile
from pathlib import Path

import pytest

HERE = Path(__file__).parent
spec = importlib.util.spec_from_file_location("resource_audit", HERE / "audit.py")
m = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = m
spec.loader.exec_module(m)


class Limits:
    def __init__(self):
        self.calls = 0

    def check(self):
        self.calls += 1


def budget(**kwargs):
    return m.Budget(Limits(), **kwargs)


def fixture_zip(root, entries):
    p = root / "source.zip"
    with zipfile.ZipFile(p, "w", compression=zipfile.ZIP_DEFLATED) as z:
        for name, data in entries:
            z.writestr(name, data)
    return p


def test_exact_frozen_function_text_pins():
    pins = json.loads((HERE / "resolver-source-pins.json").read_bytes())
    text = (HERE / "resolver.py").read_text()
    tree = ast.parse(text)
    for node in tree.body:
        if isinstance(node, ast.FunctionDef):
            assert (
                hashlib.sha256(ast.get_source_segment(text, node).encode()).hexdigest()
                == pins[node.name]["function_text_sha256"]
            )
    assert m.digest(HERE / "resolver.py") == m.RESOLVER_SHA


def test_exact_suffix_crc_alias_rules():
    r = m.resolver()
    names = ["long/path/RGB.jpg", "path/RGB.jpg"]
    ids = {n: (7, 123) for n in names}
    assert r.resolve_requests(names, ["RGB.jpg"], ids) == {"RGB.jpg": "path/RGB.jpg"}
    assert r.resolve_requests(
        names, ["long/path/RGB.jpg"], {n: (7, i) for i, n in enumerate(names)}
    ) == {"long/path/RGB.jpg": "long/path/RGB.jpg"}
    with pytest.raises(ValueError, match="ambiguous"):
        r.resolve_requests(names, ["RGB.jpg"], {names[0]: (7, 1), names[1]: (7, 2)})
    with pytest.raises(ValueError, match="ambiguous"):
        r.resolve_requests(names, ["RGB.jpg"])


def test_zip_only_retained_payload_hashed_and_producer_schema(tmp_path, monkeypatch):
    blob = b"encoded-not-an-image\0" * 29
    fixture_zip(
        tmp_path,
        [
            ("long/train/rgb.jpg", blob),
            ("train/rgb.jpg", blob),
            ("TEST/do-not-open.jpg", b"notread"),
        ],
    )
    opened = []
    original = zipfile.ZipFile.open

    def recording(self, name, *a, **kw):
        n = name.filename if isinstance(name, zipfile.ZipInfo) else name
        assert "TEST" not in n
        opened.append(n)
        return original(self, name, *a, **kw)

    monkeypatch.setattr(zipfile.ZipFile, "open", recording)
    result = m.audit_requests({"source.zip": {"rgb.jpg": ["RGB"]}}, tmp_path, budget())
    e = result["resources"]["source.zip"]
    assert e == {
        "request_to_member": {"rgb.jpg": "train/rgb.jpg"},
        "members": {
            "train/rgb.jpg": {"size": len(blob), "sha256": hashlib.sha256(blob).hexdigest()}
        },
    }
    assert opened == ["train/rgb.jpg"] and result["member_payload_bytes"] == len(blob)
    assert (
        result["source_stats"]["source.zip"]["before"]
        == result["source_stats"]["source.zip"]["after"]
    )


def test_conflicting_canonical_zip_duplicate_fails_before_payload(tmp_path, monkeypatch):
    fixture_zip(tmp_path, [("prefix/data/databases/a.jpg", b"A"), ("a.jpg", b"B")])

    def forbidden(*a, **kw):
        raise AssertionError("No payload before duplicate rejection")

    monkeypatch.setattr(zipfile.ZipFile, "open", forbidden)
    with pytest.raises(ValueError, match="duplicate CRC/size"):
        m.audit_requests({"source.zip": {"a.jpg": ["RGB"]}}, tmp_path, budget())


def test_same_crc_canonical_duplicate_retains_original_first_member(tmp_path):
    blob = b"same bytes"
    fixture_zip(tmp_path, [("prefix/data/databases/a.jpg", blob), ("a.jpg", blob)])
    result = m.audit_requests({"source.zip": {"a.jpg": ["RGB"]}}, tmp_path, budget())
    assert result["resources"]["source.zip"]["members"] == {
        "a.jpg": {"size": len(blob), "sha256": hashlib.sha256(blob).hexdigest()}
    }
    assert result["member_payload_bytes"] == len(blob)


def test_zip_symlink_and_escape_rejected_before_payload(tmp_path, monkeypatch):
    p = tmp_path / "source.zip"
    info = zipfile.ZipInfo("link.jpg")
    info.create_system = 3
    info.external_attr = (stat.S_IFLNK | 0o777) << 16
    with zipfile.ZipFile(p, "w") as z:
        z.writestr(info, b"outside")
    with pytest.raises(ValueError, match="symlink"):
        m.audit_requests({"source.zip": {"link.jpg": ["RGB"]}}, tmp_path, budget())
    p.unlink()
    fixture_zip(tmp_path, [("../outside.jpg", b"no")])
    with pytest.raises(ValueError, match="escape"):
        m.audit_requests({"source.zip": {"outside.jpg": ["RGB"]}}, tmp_path, budget())


def test_central_directory_entry_ceiling_before_zip_index(tmp_path, monkeypatch):
    fixture_zip(tmp_path, [("a", b"A"), ("b", b"B")])
    monkeypatch.setattr(m, "MAX_ENTRIES", 1)

    def forbidden(*a, **kw):
        raise AssertionError("ZipFile index must not allocate")

    monkeypatch.setattr(zipfile, "ZipFile", forbidden)
    with pytest.raises(ValueError, match="metadata ceiling"):
        m.audit_requests({"source.zip": {"a": ["RGB"]}}, tmp_path, budget())


def test_directory_only_selected_file_hashed(tmp_path, monkeypatch):
    d = tmp_path / "dataset"
    (d / "train").mkdir(parents=True)
    (d / "TEST").mkdir()
    (d / "train/a.jpg").write_bytes(b"abc")
    (d / "TEST/notselected").write_bytes(b"no")
    opened = []
    original = m.open_member

    def record(root, name):
        opened.append(name)
        return original(root, name)

    monkeypatch.setattr(m, "open_member", record)
    result = m.audit_requests({"dataset": {"a.jpg": ["RGB"]}}, tmp_path, budget())
    assert opened == ["train/a.jpg"]
    assert (
        result["resources"]["dataset"]["members"]["train/a.jpg"]["sha256"]
        == hashlib.sha256(b"abc").hexdigest()
    )


def test_directory_symlink_and_relative_escape_fail(tmp_path):
    d = tmp_path / "dataset"
    d.mkdir()
    outside = tmp_path / "outside"
    outside.write_bytes(b"no")
    (d / "link").symlink_to(outside)
    with pytest.raises(ValueError, match="symlink"):
        m.audit_requests({"dataset": {"link": ["RGB"]}}, tmp_path, budget())
    with pytest.raises(ValueError, match="escape"):
        m.audit_requests({"../outside": {"x": ["RGB"]}}, tmp_path, budget())


def test_directory_ambiguous_canonical_collision_has_no_crc_proof(tmp_path):
    d = tmp_path / "dataset"
    (d / "prefix/data/databases").mkdir(parents=True)
    (d / "a").write_bytes(b"A")
    (d / "prefix/data/databases/a").write_bytes(b"A")
    with pytest.raises(ValueError, match="canonical directory duplicate"):
        m.audit_requests({"dataset": {"a": ["RGB"]}}, tmp_path, budget())


def test_payload_limit_checked_before_read_and_chunk_size_at_most1MiB():
    class Reader(io.BytesIO):
        def __init__(self, b):
            super().__init__(b)
            self.sizes = []

        def read(self, n=-1):
            self.sizes.append(n)
            return super().read(n)

    f = Reader(b"A" * 10)
    with pytest.raises(ValueError, match="overbudget"):
        m.hash_stream(f, 10, budget(maximum=9))
    assert not f.sizes
    blob = b"A" * (m.CHUNK + 17)
    f = Reader(blob)
    result = m.hash_stream(f, len(blob), budget())
    assert f.sizes == [m.CHUNK, 17] and result["sha256"] == hashlib.sha256(blob).hexdigest()


def test_time_and_truncated_payload_fail():
    now = [0]
    b = budget(clock=lambda: now[0])
    now[0] = 1800
    with pytest.raises(ValueError, match="time"):
        m.hash_stream(io.BytesIO(b"A"), 1, b)
    with pytest.raises(ValueError, match="Truncated"):
        m.hash_stream(io.BytesIO(b"A"), 2, budget())


def test_exclusive_output_and32MiB_cap(tmp_path, monkeypatch):
    out = tmp_path / "audit.json"
    out.write_text("original")
    with pytest.raises(ValueError, match="overwrite"):
        m.save_exclusive(out, {"x": 1})
    assert out.read_text() == "original"
    out = tmp_path / "new.json"
    monkeypatch.setattr(m, "MAX_OUTPUT", 32)
    with pytest.raises(ValueError, match="ceiling"):
        m.save_exclusive(out, {"x": "A" * 80})
    assert not out.exists() and out.with_name("new.json.partial").exists()


def test_false_cli_rejects_before_runtime_plan_or_source(tmp_path, monkeypatch):
    p = tmp_path / "False.json"
    p.write_text(json.dumps({"execution_authorized": False, "root_reviewed": False}))

    def forbidden(*a, **k):
        raise AssertionError("No runtime/source under False")

    monkeypatch.setattr(m, "Runtime", forbidden)
    with pytest.raises(ValueError, match="Disabled"):
        m.main(
            [
                "--profile",
                str(p),
                "--trusted-profile-sha256",
                m.digest(p),
                "--plan",
                "missing",
                "--source-root",
                "missing",
                "--output",
                "missing",
            ]
        )


def test_full_metadata_scope_and_no_added_canonical_test_partition():
    requests = {f"{i}.zip": {} for i in range(22)}
    for i in range(10999):
        requests[f"{i % 22}.zip"][f"retained/{i}.jpg"] = ["RGB"]
    plan = {
        "producer_commit": m.PRODUCER,
        "membership_sha256": m.MEMBERSHIP,
        "manifest_sha256": "17a2277c39cb37b7eee1b2063cfd13f2cf4adf4ebdf86cf449a1f51367c5a767",
        "photos_sha256": "765709a1ddc08035565ebf3d54d414dd1f53a9882a5cfa9c0fd03811b8e48a4d",
        "records": {"train": 5700, "val": 993},
        "photos": {"train": 2529, "val": 770},
        "VAL_groups": 612,
        "row_sources": {"train": sorted(m.TRAIN), "val": sorted(m.VAL)},
        "requests": requests,
    }
    m.check_plan(plan)
    plan["records"]["test"] = 1
    with pytest.raises(ValueError, match="scope"):
        m.check_plan(plan)


def test_small_real_zip64_eocd_is_supported_without_gigabyte_fixture(tmp_path):
    import struct

    p = fixture_zip(tmp_path, [("train/a.jpg", b"abc")])
    raw = p.read_bytes()
    pos = raw.rfind(b"PK\x05\x06")
    end = list(struct.unpack("<4s4H2LH", raw[pos : pos + 22]))
    entries, size, offset = end[4], end[5], end[6]
    record = struct.pack(
        "<4sQ2H2L4Q", b"PK\x06\x06", 44, 45, 45, 0, 0, entries, entries, size, offset
    )
    locator = struct.pack("<4sLQL", b"PK\x06\x07", 0, pos, 1)
    end[3] = end[4] = 0xFFFF
    end[5] = end[6] = 0xFFFFFFFF
    p.write_bytes(raw[:pos] + record + locator + struct.pack("<4s4H2LH", *end))
    result = m.audit_requests({"source.zip": {"a.jpg": ["RGB"]}}, tmp_path, budget())
    assert (
        result["resources"]["source.zip"]["members"]["train/a.jpg"]["sha256"]
        == hashlib.sha256(b"abc").hexdigest()
    )


@pytest.mark.parametrize("bad", ["runtime", "swap", "CPU", "group"])
def test_wrong_actual_unit_limit_refused_before_source(monkeypatch, bad):
    unit = "deepretina-raw35-resource-audit-fixture.service"
    group = "/user.slice/" + unit
    props = {
        "ControlGroup": group,
        "MemoryMax": str(2 * 1024**3),
        "MemorySwapMax": "0",
        "CPUQuotaPerSecUSec": "1s",
        "RuntimeMaxUSec": "30min",
        "KillMode": "control-group",
    }
    cg = {"memory.max": str(2 * 1024**3), "memory.swap.max": "0", "cpu.max": "100000 100000"}
    if bad == "runtime":
        props["RuntimeMaxUSec"] = "31min"
    if bad == "swap":
        cg["memory.swap.max"] = "1"
    if bad == "CPU":
        cg["cpu.max"] = "200000 100000"
    if bad == "group":
        props["ControlGroup"] = "/other"
    monkeypatch.setattr(
        m.subprocess,
        "check_output",
        lambda cmd, **kw: (
            "idle" if cmd[0] == "ionice" else "".join(k + "=" + v + "\n" for k, v in props.items())
        ),
    )
    monkeypatch.setattr(
        Path,
        "read_text",
        lambda p, *a, **kw: "0::" + group + "\n" if str(p) == "/proc/self/cgroup" else cg[p.name],
    )
    monkeypatch.setattr(m.os, "sched_getaffinity", lambda _: {4})
    monkeypatch.setattr(m.os, "getpriority", lambda *a: 17)
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "")
    with pytest.raises(ValueError):
        m.Runtime({"unit": unit, "cpu_core": 4})


def test_correct_actual_unit_metadata_is_logged_without_other_environment(monkeypatch):
    unit = "deepretina-raw35-resource-audit-fixture.service"
    group = "/user.slice/" + unit
    props = {
        "ControlGroup": group,
        "MemoryMax": str(2 * 1024**3),
        "MemorySwapMax": "0",
        "CPUQuotaPerSecUSec": "1s",
        "RuntimeMaxUSec": "30min",
        "KillMode": "control-group",
    }
    cg = {"memory.max": str(2 * 1024**3), "memory.swap.max": "0", "cpu.max": "100000 100000"}
    monkeypatch.setattr(
        m.subprocess,
        "check_output",
        lambda cmd, **kw: (
            "idle" if cmd[0] == "ionice" else "".join(k + "=" + v + "\n" for k, v in props.items())
        ),
    )
    monkeypatch.setattr(
        Path,
        "read_text",
        lambda p, *a, **kw: "0::" + group + "\n" if str(p) == "/proc/self/cgroup" else cg[p.name],
    )
    monkeypatch.setattr(m.os, "sched_getaffinity", lambda _: {4})
    monkeypatch.setattr(m.os, "getpriority", lambda *a: 17)
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "")
    proof = m.Runtime({"unit": unit, "cpu_core": 4}).proof
    assert proof["actual_cgroup"] == cg and proof["core"] == 4 and "environment" not in proof
