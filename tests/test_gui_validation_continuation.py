"""Fake process metadata; no real /proc, GPU, checkpoint or model operations."""

import importlib.util
from pathlib import Path

import pytest

p = Path(__file__).parents[1] / "scripts/resume_six_e060_validation.py"
s = importlib.util.spec_from_file_location("gui_resume", p)
h = importlib.util.module_from_spec(s)
s.loader.exec_module(h)


def identity():
    return {
        "pid": 6311,
        "uid": 1000,
        "exe": "/usr/libexec/gnome-remote-desktop-daemon",
        "starttime_ticks": 123456,
    }


def test_identity_pinned_established_gui_accepted_and_pure_graphics_ignored():
    allowed = identity()
    rows = [{"pid": 6311, "type": "C+G"}, {"pid": 99, "type": "G"}]
    assert h.check_desktop_processes(rows, [allowed], {6311: dict(allowed)}) == [allowed]


@pytest.mark.parametrize("change", ["unknown", "UID", "exe", "start", "type"])
def test_unknown_or_changed_compute_identity_rejected(change):
    allowed = identity()
    row = {"pid": 6311, "type": "C+G"}
    observed = dict(allowed)
    if change == "unknown":
        row["pid"] = 4321
    elif change == "UID":
        observed["uid"] = 0
    elif change == "exe":
        observed["exe"] = "/usr/bin/python3"
    elif change == "start":
        observed["starttime_ticks"] += 1
    elif change == "type":
        row["type"] = "C"
    with pytest.raises(ValueError):
        h.check_desktop_processes([row], [allowed], {row["pid"]: observed})


def test_xml_filters_target_uuid_without_retaining_command_lines():
    xml = b"<nvidia_smi_log><gpu><uuid>GPU-target</uuid><processes><process_info><pid>3</pid><type>C+G</type><process_name>private command ignored</process_name></process_info></processes></gpu></nvidia_smi_log>"
    assert h.xml_processes(xml, "GPU-target") == [{"pid": 3, "type": "C+G"}]
    with pytest.raises(ValueError):
        h.xml_processes(xml, "GPU-wrong")


def test_reused_completed_report_requires_exact_bytes(tmp_path, monkeypatch):
    path = tmp_path / "threshold-050.json"
    payload = b'{"complete":true}\n'
    path.write_bytes(payload)
    monkeypatch.setattr(h, "REUSED_REPORTS", {0.5: h.hashlib.sha256(payload).hexdigest()})
    assert h.reuse_completed_report(tmp_path, 0.5) == payload
    path.write_bytes(b'{"complete":false}\n')
    with pytest.raises(ValueError, match="changed"):
        h.reuse_completed_report(tmp_path, 0.5)
