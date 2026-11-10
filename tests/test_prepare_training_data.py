"""RAW-only workflow ordering and identity guards, without an author-owned parent."""

import importlib.util
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

SPEC = importlib.util.spec_from_file_location(
    "raw_training_workflow", Path(__file__).parents[1] / "scripts/prepare_training_data.py"
)
workflow = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(workflow)
POLICY = Path(__file__).parents[1] / "reports/data/corpus-v2-geometry-policy/policy.json"
FIXED = "named_cup_is_disc_named_disc_is_cup"


def archives(path):
    path.mkdir()
    for name in ("DRIONS-DB.zip", "GAMMA.zip", "IDRiD.zip", "BIOMISA.zip"):
        (path / name).touch()


def test_pipeline_pins_actual_audited_parent_after_group_repair(tmp_path, monkeypatch):
    raw, parent, final = [tmp_path / n for n in ("raw", "parent", "final")]
    archives(raw)
    calls = []

    def prepare(data, out):
        calls.append("prepare")
        out.mkdir()

    def regroup(out, data, **kw):
        assert kw == {"seed": 42, "train_only_datasets": ["riga", "uoadr"]}
        calls.append("regroup")

    def audit(out):
        calls.append("audit-parent")
        return {"prepared_content_sha256": "newly-audited-parent"}

    def append(p, o, data, policy, **kw):
        assert p == parent and kw["expected_parent"] == "newly-audited-parent"
        assert kw["biomisa_policy"] == FIXED
        calls.append("append-and-audit")
        o.mkdir()
        return {
            "prepared_content_sha256": "new-final",
            "head_counts": [
                {"dataset": name, "split": "train", "positive_pixels": 1}
                for name in ("drive", "drions", "gamma", "idrid_disc", "biomisa")
            ],
        }

    fake = SimpleNamespace(
        FIXED_POLICY=FIXED,
        GEOMETRY_POLICY_SHA256=__import__("hashlib").sha256(POLICY.read_bytes()).hexdigest(),
        base=SimpleNamespace(
            prepare_unified_data=prepare,
            regroup_patients=regroup,
            audit_prepared=audit,
            DEFAULT_DATASETS=["drive"],
        ),
        append=append,
        write_json=lambda p, obj: p.write_text(json.dumps(obj)),
    )
    monkeypatch.setattr(workflow, "_load_builder", lambda: fake)
    result = workflow.build(raw, parent, final, POLICY, biomisa_policy=FIXED)
    assert calls == ["prepare", "regroup", "audit-parent", "append-and-audit"]
    assert result["historical_identity_reproduction_claim"] is False
    assert (
        json.loads((final / "raw-workflow.json").read_text())["final_content_sha256"] == "new-final"
    )
    with pytest.raises(ValueError, match="new directories"):
        workflow.build(raw, parent, final, POLICY, biomisa_policy=FIXED)

    # append's input assertions refer to the first pair; change only its report
    # to exercise the missing-source gate with fresh output paths.
    def missing_source_append(p, o, *args, **kwargs):
        o.mkdir()
        return {
            "prepared_content_sha256": "partial",
            "head_counts": [
                {"dataset": name, "split": "train", "positive_pixels": 1}
                for name in ("drive", "drions", "idrid_disc", "biomisa")
            ],
        }

    fake.append = missing_source_append
    with pytest.raises(ValueError, match="required sources.*gamma"):
        workflow.build(raw, tmp_path / "parent2", tmp_path / "final2", POLICY, biomisa_policy=FIXED)
    assert not (tmp_path / "final2/raw-workflow.json").exists()


@pytest.mark.parametrize("case", ["nested", "wrong-policy", "missing-archive"])
def test_fail_before_any_preparation(tmp_path, monkeypatch, case):
    raw, parent, final = [tmp_path / n for n in ("raw", "parent", "final")]
    archives(raw)
    if case == "nested":
        parent = raw / "parent"
    policy = POLICY
    if case == "wrong-policy":
        policy = tmp_path / "policy.json"
        policy.write_text("{}")
    if case == "missing-archive":
        (raw / "GAMMA.zip").unlink()
    with pytest.raises((ValueError, FileNotFoundError)):
        workflow.build(raw, parent, final, policy, biomisa_policy=FIXED)
    assert not parent.exists() and not final.exists()


def test_append_cli_exposes_explicit_parent_pin(monkeypatch, tmp_path, capsys):
    v2 = workflow._load_builder()
    args = [
        "prepare_corpus_v2.py",
        "--parent",
        "parent",
        "--output",
        "final",
        "--data-root",
        "raw",
        "--geometry-policy",
        str(POLICY),
        "--biomisa-label-policy",
        FIXED,
        "--expected-parent-fingerprint",
        "a" * 64,
    ]
    monkeypatch.setattr(sys, "argv", args)
    monkeypatch.setattr(v2, "append", lambda *a, **kw: kw)
    v2.main()
    assert json.loads(capsys.readouterr().out)["expected_parent"] == "a" * 64


def test_original_topology_preset_is_scratch_and_keeps_complete_budget():
    import tomllib

    from deepretina.unified import UnifiedConfig, build_unified

    path = Path(__file__).parents[1] / "configs/unified_legacy_topology.toml"
    cfg = UnifiedConfig(**tomllib.loads(path.read_text()))
    cfg.validate()
    model = build_unified(cfg)
    assert cfg.epochs == 60 and cfg.epoch_samples == 8192
    assert cfg.legacy_initialization is None and model.lineage["initialization"] == "scratch"
    assert model.architecture_spec["name"] == "legacy_universal"
    assert model.architecture_spec["outputs"] == 5


def test_four_raw_source_fixture_runs_real_prepare_copy_append_and_audit(tmp_path, monkeypatch):
    """Real DRIVE/RITE pairing, Drishti disc/cup, and DRIONS polygons; other additions empty in fixture.

    The production call remains fixed to all sources; narrowing is test-only.
    """
    import io
    import zipfile

    import numpy as np
    from PIL import Image

    v2 = workflow._load_builder()
    raw, parent, final = [tmp_path / n for n in ("raw", "parent", "final")]
    raw.mkdir()

    def encoded(array):
        stream = io.BytesIO()
        Image.fromarray(array).save(stream, format="PNG")
        return stream.getvalue()

    yy, xx = np.indices((40, 60))
    rgb = np.stack([80 + xx, 70 + yy, 90 + (xx + yy) // 2], -1).astype(np.uint8)
    mask = np.zeros((40, 60), np.uint8)
    mask[8:30, 20:24] = 255
    av = np.zeros_like(rgb)
    av[mask > 0] = [255, 0, 0]
    with zipfile.ZipFile(raw / "DRIVE.zip", "w") as z:
        z.writestr("DRIVE/training/images/21_training.tif", encoded(rgb))
        z.writestr("DRIVE/training/1st_manual/21_manual1.gif", encoded(mask))
        z.writestr("DRIVE/training/mask/21_training_mask.gif", encoded(np.full_like(mask, 255)))
    with zipfile.ZipFile(raw / "RITE.zip", "w") as z:
        prefix = "RITE/AV_groundTruth/training/"
        z.writestr(prefix + "images/21_training.tif", encoded(rgb))
        z.writestr(prefix + "vessel/21_training.png", encoded(mask))
        z.writestr(prefix + "av/21_training.png", encoded(av))
    disc = np.zeros_like(mask)
    disc[8:32, 20:44] = 255
    cup = np.zeros_like(mask)
    cup[15:25, 27:37] = 255
    with zipfile.ZipFile(raw / "Drishti-GS.zip", "w") as z:
        z.writestr("Training/Images/drishtiGS_001.png", encoded(rgb + 20))
        prefix = "Training/GT/drishtiGS_001/SoftMap/drishtiGS_001"
        z.writestr(prefix + "_ODsegSoftmap.png", encoded(disc))
        z.writestr(prefix + "_cupsegSoftmap.png", encoded(cup))
    with zipfile.ZipFile(raw / "DRIONS-DB.zip", "w") as z:
        z.writestr("images/image_001.jpg", encoded(rgb + 10))
        angles = np.linspace(0, 2 * np.pi, 36, endpoint=False)
        points = "\n".join(f"{30 + 8 * np.cos(t)},{20 + 8 * np.sin(t)}" for t in angles)
        for observer in (1, 2):
            z.writestr(f"experts_anotation/anotExpert{observer}_001.txt", points)
    for name in ("GAMMA.zip", "IDRiD.zip", "BIOMISA.zip"):
        with zipfile.ZipFile(raw / name, "w"):
            pass
    original_prepare = v2.base.prepare_unified_data
    monkeypatch.setattr(
        v2.base,
        "prepare_unified_data",
        lambda data, out: original_prepare(data, out, ["drive", "rite", "drishti"]),
    )
    monkeypatch.setattr(v2, "gamma", lambda source, emit: None)
    monkeypatch.setattr(v2, "idrid_disc", lambda source, emit: None)
    monkeypatch.setattr(v2, "biomisa", lambda source, emit, **kw: None)
    monkeypatch.setattr(workflow, "_load_builder", lambda: v2)
    monkeypatch.setattr(workflow, "ADDITIONAL_DATASETS", ("drions",))
    monkeypatch.setattr(v2.base, "DEFAULT_DATASETS", ["drive", "rite", "drishti"])
    result = workflow.build(raw, parent, final, POLICY, biomisa_policy=FIXED)
    rows = json.loads((final / "manifest.json").read_text())
    assert {r["dataset"] for r in rows} == {"drive", "rite", "drishti", "drions"}
    assert len(rows) == 5
    parent_rows = json.loads((parent / "manifest.json").read_text())
    assert rows[:3] == parent_rows
    by_dataset = {r["dataset"]: r for r in parent_rows}
    assert by_dataset["drive"]["photo_id"] == by_dataset["rite"]["photo_id"]
    assert result["parent_content_sha256"] != v2.PARENT_FINGERPRINT
    assert result["final_audit"]["annotations_checked"] == 5
    assert not result["final_audit"]["preview_only"]
    assert {
        h["channel"] for h in result["final_audit"]["head_counts"] if h["positive_pixels"] > 0
    } == {"vessel", "artery", "disc", "cup"}
    # Vein is genuinely supervised negative on this artery-only fixture, not invented positive.
    assert any(
        h["channel"] == "vein" and h["valid_pixels"] > 0
        for h in result["final_audit"]["head_counts"]
    )
    for r in rows:
        with np.load(final / r["mask"]) as labels:
            if r["dataset"] == "drions":
                assert labels["valid"][0].any() and not labels["valid"][1].any()
        if r["dataset"] in {"drive", "rite"}:
            assert (final / r["image"]).read_bytes() == (parent / r["image"]).read_bytes()
            assert (final / r["image"]).stat().st_ino != (parent / r["image"]).stat().st_ino
    from deepretina.integrity import verify_prepared_content

    assert verify_prepared_content(final / "manifest.json", result["final_content_sha256"])
