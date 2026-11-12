"""Ensure latency comparison uses the selected dedicated reference components."""

import copy
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch


@pytest.fixture
def frozen_reference(benchmark_module, monkeypatch, tmp_path):
    from evaluate_original_frozen import POLICY

    policy = copy.deepcopy(POLICY)
    keys = {
        "vessel": "vessel_checkpoint_sha256",
        "disc": "disc_checkpoint_sha256",
        "universal": "checkpoint_sha256",
        "av": "av_checkpoint_sha256",
    }
    paths = {name: tmp_path / (name + ".pt") for name in keys}
    monkeypatch.setattr(benchmark_module, "sha256", lambda p: policy["model_sha256"][Path(p).stem])
    report = {
        "schema_version": 1,
        "family": "frozen_original_reference",
        "split": "val",
        "preview_only": False,
        "policy_spec": policy,
        "manifest_sha256": "a" * 64,
        "prepared_content_sha256": "b" * 64,
        **{key: policy["model_sha256"][name] for name, key in keys.items()},
    }
    return paths, report


def test_frozen_original_reference_schema_without_retagging(benchmark_module, frozen_reference):
    paths, report = frozen_reference
    before = json.dumps(report, sort_keys=True)
    benchmark_module.validate_reference(
        report, paths, cup_policy="inside-disc", disc_variant="disc0/raw"
    )
    assert benchmark_module.reference_cup_policy(report) == "inside-disc"
    assert benchmark_module.reference_disc_variant(report) == "disc0/raw"
    assert benchmark_module.reference_gt_policy(report) == report["policy_spec"]["gt"]
    assert json.dumps(report, sort_keys=True) == before
    assert report["family"] == "frozen_original_reference"


@pytest.mark.parametrize(
    "change",
    ["cup", "av", "policy_model", "top_model", "schema", "test", "preview", "native", "disc1"],
)
def test_frozen_reference_protocol_and_identity_drift_rejected(
    benchmark_module, frozen_reference, change
):
    paths, original = frozen_reference
    report = copy.deepcopy(original)
    options = {"cup_policy": "inside-disc", "disc_variant": "disc0/raw"}
    if change in {"cup", "av"}:
        report["policy_spec"][change] = "different operator"
    elif change == "policy_model":
        report["policy_spec"]["model_sha256"]["av"] = "c" * 64
    elif change == "top_model":
        report["av_checkpoint_sha256"] = "c" * 64
    elif change == "schema":
        report["schema_version"] = 2
    elif change == "test":
        report["split"] = "test"
    elif change == "preview":
        report["preview_only"] = True
    elif change == "native":
        options["cup_policy"] = "native"
    else:
        options["disc_variant"] = "disc1/raw"
    with pytest.raises(ValueError):
        benchmark_module.validate_reference(report, paths, **options)


@pytest.fixture
def benchmark_module(monkeypatch):
    scripts = Path(__file__).parents[1] / "scripts"
    monkeypatch.syspath_prepend(str(scripts))
    spec = importlib.util.spec_from_file_location(
        "strong_pipeline_benchmark", scripts / "benchmark_strong_pipeline.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def reference_files(tmp_path, benchmark_module):
    paths = {}
    keys = {
        "vessel": "vessel_checkpoint_sha256",
        "disc": "disc_checkpoint_sha256",
        "universal": "checkpoint_sha256",
        "av": "av_checkpoint_sha256",
    }
    report = {
        "family": "original_reference",
        "split": "val",
        "preview_only": False,
        "single_net_normalized": False,
        "av_preprocessing_profile": "vesclass_matlab_provided_fov_calibrated",
        "disc_reference_variant": "disc0/raw",
        "manifest_sha256": "a" * 64,
        "prepared_content_sha256": "b" * 64,
        "policy": "Synthetic selected quality domain",
    }
    for name, key in keys.items():
        path = tmp_path / (name + ".pt")
        path.write_bytes(name.encode())
        paths[name] = path
        report[key] = benchmark_module.sha256(path)
    return paths, report


def test_exact_selected_components_required(benchmark_module, reference_files):
    paths, report = reference_files
    benchmark_module.validate_reference(report, paths)
    paths["disc"].write_bytes(b"a different disc checkpoint")
    with pytest.raises(ValueError, match="disc weights differ"):
        benchmark_module.validate_reference(report, paths)


@pytest.mark.parametrize(
    "change",
    [
        {"split": "test"},
        {"preview_only": True},
        {"av_preprocessing_profile": "proxy"},
        {"disc_reference_variant": "universal/raw"},
        {"single_net_normalized": True},
    ],
)
def test_weaker_or_different_reference_is_rejected(benchmark_module, reference_files, change):
    paths, report = reference_files
    with pytest.raises(ValueError, match="full calibrated"):
        benchmark_module.validate_reference({**report, **change}, paths)


def test_four_network_pipeline_uses_dedicated_heads_native_tiles_and_calibrated_av(
    benchmark_module, monkeypatch
):
    m = benchmark_module
    rgb = np.zeros((2, 3, 3), dtype=np.uint8)
    fov = np.ones((2, 3), dtype=bool)

    class Geometry:
        def restore(self, values, order):
            assert order == 0
            return values

    prepared = SimpleNamespace(tensor=lambda: torch.zeros(3, 2, 3), geometry=Geometry())
    monkeypatch.setattr(m, "prepare_single_net", lambda *a, **kw: prepared)
    vessel = torch.tensor([[[0.0, 1.0, -1.0], [1.0, -1.0, 0.0]]])
    disc = torch.ones(1, 2, 3)
    universal = -torch.ones(3, 2, 3)
    universal[2, 1, 2] = 1
    specs = {
        "vessel": (vessel, 320, 40),
        "disc": (disc, 192, 24),
        "universal": (universal, 288, 36),
    }
    calls = []

    def tiled(model, image, device, crop_size, border):
        name = model
        assert (crop_size, border) == specs[name][1:]
        calls.append(name)
        return specs[name][0]

    monkeypatch.setattr(m, "tiled_logits", tiled)

    def calibrated(raw, predicted_vessel, *, fov, **kwargs):
        assert raw is rgb
        np.testing.assert_array_equal(predicted_vessel, vessel[0].numpy() > 0)
        return SimpleNamespace(inputs=lambda: "calibrated inputs")

    monkeypatch.setattr(m, "prepare_vesclass_matlab_calibrated", calibrated)
    classes = np.array([[0, 1, 0], [2, 0, 0]], np.uint8)

    def av(model, inputs, *, device, original_size):
        assert inputs == "calibrated inputs" and original_size == (2, 3)
        return {"classes": classes}

    monkeypatch.setattr(m, "predict_av_input", av)
    models = {
        name: (name, {"tiling": {"crop_size": spec[1], "border": spec[2]}})
        for name, spec in specs.items()
    }
    result = m.original_operation(rgb, fov, models, "av", torch.device("cpu"), {})
    assert calls == ["vessel", "disc", "universal"]
    np.testing.assert_array_equal(result[0], vessel[0] > 0)
    np.testing.assert_array_equal(result[1], classes == 1)
    np.testing.assert_array_equal(result[2], classes == 2)
    np.testing.assert_array_equal(result[3], disc[0] > 0)
    np.testing.assert_array_equal(result[4], universal[2] > 0)


def test_nonfinite_original_predictions_fail(benchmark_module, monkeypatch):
    m = benchmark_module
    prepared = SimpleNamespace(tensor=lambda: torch.zeros(3, 2, 3))
    monkeypatch.setattr(m, "prepare_single_net", lambda *a, **kw: prepared)
    monkeypatch.setattr(m, "tiled_logits", lambda *a, **kw: torch.full((1, 2, 3), float("nan")))
    with pytest.raises(ValueError, match="Nonfinite"):
        m.original_operation(None, None, {"vessel": ("vessel", {"tiling": {}})}, None, "cpu", {})


def test_unified_uses_checkpoint_geometry_and_masks_original_frame(benchmark_module, monkeypatch):
    m = benchmark_module
    fov = np.array([[False, True, True], [True, True, False]])

    class Geometry:
        def restore(self, value, order):
            return value

    prepared = SimpleNamespace(
        image=np.zeros((2, 3, 3), dtype=np.uint8), fov=fov, geometry=Geometry()
    )
    monkeypatch.setattr(m, "prepare_single_net", lambda *a, **kw: prepared)
    cfg = SimpleNamespace(
        tile_size=96, overlap=16, photometric_profile="raw", pixels_per_degree=25.0
    )

    def predict(model, image, device, config, tile, overlap):
        assert config is cfg and (tile, overlap) == (96, 16)
        probability = np.full((5, 2, 3), 0.9, np.float32)
        probability[2] = 0.1
        probability[3, 0, 1] = 0.1
        return probability

    monkeypatch.setattr(m, "predict_probabilities", predict)
    monkeypatch.setattr(m, "restore_probabilities", lambda p, g: p)
    model = SimpleNamespace(supervised_channels=[True] * 5)
    result = m.unified_operation(None, fov, model, cfg, "cpu", {})
    assert not result[:, ~fov].any()
    assert result[1, 0, 1] and not result[2, 0, 1]
    assert not result[4, 0, 1]  # Cup containment retained.
    assert result[4, 1, 0]


def test_selected_threshold_controls_vessel_gate_and_disc_cup(benchmark_module, monkeypatch):
    m = benchmark_module
    fov = np.ones((2, 3), dtype=bool)
    geometry = SimpleNamespace(restore=lambda value, order: value)
    prepared = SimpleNamespace(image=np.zeros((2, 3, 3), np.uint8), fov=fov, geometry=geometry)
    monkeypatch.setattr(m, "prepare_single_net", lambda *a, **kw: prepared)
    monkeypatch.setattr(m, "restore_probabilities", lambda p, g: p)
    probability = np.full((5, 2, 3), 0.4, np.float32)
    probability[1] = 0.7
    probability[2] = 0.3
    probability[3, 0, 0] = 0.2
    monkeypatch.setattr(m, "predict_probabilities", lambda *a: probability.copy())
    model = SimpleNamespace(supervised_channels=[True] * 5)
    config = SimpleNamespace(
        tile_size=96, overlap=16, photometric_profile="raw", pixels_per_degree=25.0
    )
    baseline = m.unified_operation(None, fov, model, config, "cpu", {})
    selected = m.unified_operation(None, fov, model, config, "cpu", {}, threshold=0.35)
    assert not baseline.any()
    assert selected[0].all() and selected[1].all() and not selected[2].any()
    assert not selected[3, 0, 0] and not selected[4, 0, 0]
    assert selected[3, 1, 1] and selected[4, 1, 1]


@pytest.mark.parametrize("threshold", [float("nan"), float("inf"), 0, 1, -0.2])
def test_invalid_selected_threshold_rejected_before_device_initialization(
    benchmark_module, monkeypatch, threshold
):
    def forbidden(*a):
        pytest.fail("Device resolution must not precede threshold validation")

    monkeypatch.setattr(benchmark_module, "resolve_device", forbidden)
    with pytest.raises(ValueError, match="Unified threshold"):
        benchmark_module.benchmark(SimpleNamespace(unified_threshold=threshold))


def test_palette_fov_uses_displayed_intensity_not_indices(tmp_path, benchmark_module):
    from PIL import Image

    m = benchmark_module
    image = tmp_path / "image.png"
    fov_path = tmp_path / "fov.png"
    Image.fromarray(np.full((2, 3, 3), 80, np.uint8)).save(image)
    indices = np.array([[0, 1, 1], [1, 0, 1]], np.uint8)
    fov = Image.frombytes("P", (3, 2), indices.tobytes())
    fov.putpalette([255, 255, 255, 0, 0, 0] + [0] * 762)
    fov.save(fov_path)
    rgb, mask, identity = m.input_snapshots(image, fov_path)
    np.testing.assert_array_equal(mask, indices == 0)
    assert rgb.shape == (2, 3, 3) and identity["fov_sha256"] == m.sha256(fov_path)


def test_input_identity_matches_decoded_snapshot_after_path_replacement(
    tmp_path, benchmark_module, monkeypatch
):
    import hashlib
    import io

    from PIL import Image

    m = benchmark_module
    image = tmp_path / "image.png"
    fov = tmp_path / "fov.png"
    Image.fromarray(np.full((2, 3, 3), 80, np.uint8)).save(image)
    Image.fromarray(np.array([[0, 255, 255], [255, 0, 255]], np.uint8)).save(fov)
    image_bytes, fov_bytes = image.read_bytes(), fov.read_bytes()
    replacement = io.BytesIO()
    Image.fromarray(np.full((2, 3, 3), 190, np.uint8)).save(replacement, format="PNG")
    read = Path.read_bytes

    def swap(self):
        data = read(self)
        if self == image:
            self.write_bytes(replacement.getvalue())
        if self == fov:
            self.write_bytes(image_bytes)
        return data

    monkeypatch.setattr(Path, "read_bytes", swap)
    rgb, mask, identity = m.input_snapshots(image, fov)
    assert (rgb == 80).all()
    np.testing.assert_array_equal(mask, np.array([[False, True, True], [True, False, True]]))
    assert identity == {
        "input_sha256": hashlib.sha256(image_bytes).hexdigest(),
        "fov_sha256": hashlib.sha256(fov_bytes).hexdigest(),
    }
    assert hashlib.sha256(read(image)).hexdigest() != identity["input_sha256"]


@pytest.mark.parametrize("cup_policy", ["native", "inside-disc"])
def test_benchmark_report_retains_timed_input_snapshot_hashes(
    tmp_path, benchmark_module, reference_files, monkeypatch, cup_policy
):
    import hashlib
    import json

    from PIL import Image

    m = benchmark_module
    paths, reference = reference_files
    selected_policy = cup_policy
    if selected_policy == "inside-disc":
        reference["family"] = "optional_fresh_paired_cup_experiment"
        reference["fresh_paired_cup_component"] = {
            "cup_variant": "disc0",
            "reference_context": "disc0",
            "policy": m.CUP_COMPONENT_POLICY,
        }
    image = tmp_path / "input.png"
    fov = tmp_path / "fov.png"
    unified = tmp_path / "unified.pt"
    ref = tmp_path / "ref.json"
    Image.fromarray(np.full((2, 3, 3), 80, np.uint8)).save(image)
    Image.fromarray(np.ones((2, 3), np.uint8) * 255).save(fov)
    unified.write_bytes(b"unified")
    ref.write_text(json.dumps(reference))
    snapshots = {
        "input_sha256": hashlib.sha256(image.read_bytes()).hexdigest(),
        "fov_sha256": hashlib.sha256(fov.read_bytes()).hexdigest(),
    }
    channels = {"vessel": ["vessel"], "disc": ["disc"], "universal": ["vessel", "disc", "cup"]}
    monkeypatch.setattr(
        m,
        "load_single_net",
        lambda path, device: (
            torch.nn.Linear(1, 1),
            {"channels": channels[path.stem], "tiling": {"crop_size": 288, "border": 36}},
        ),
    )
    monkeypatch.setattr(m, "load_av", lambda *args: (torch.nn.Linear(1, 1), {}))
    model = torch.nn.Linear(1, 1)
    model.supervised_channels = [True] * 5
    monkeypatch.setattr(
        m,
        "load_unified",
        lambda *args: (model, {"config": {"purpose": "full_data"}}, torch.device("cpu")),
    )
    calls = []

    def inspect_original(rgb, mask, *args, cup_policy):
        assert (rgb == 80).all() and mask.all()
        assert cup_policy == selected_policy
        calls.append("original")
        return np.zeros((5, 2, 3), bool)

    def inspect_unified(rgb, mask, *args, threshold):
        assert (rgb == 80).all() and mask.all()
        assert threshold == 0.35
        calls.append("unified")
        return np.zeros((5, 2, 3), bool)

    monkeypatch.setattr(m, "original_operation", inspect_original)
    monkeypatch.setattr(m, "unified_operation", inspect_unified)

    def measured(operation, *args, **kwargs):
        operation()
        image.write_bytes(b"path replaced after decoding")
        fov.write_bytes(b"path replaced after decoding")
        return {"median_seconds": 1}

    monkeypatch.setattr(m, "measure", measured)
    args = SimpleNamespace(
        **paths,
        unified=unified,
        reference_report=ref,
        input=image,
        fov=fov,
        dataset="drive",
        field_of_view_degrees=None,
        device="cpu",
        unified_threshold=0.35,
        original_cup_policy=selected_policy,
        original_disc_variant="disc0/raw",
        warmup=0,
        repeats=1,
    )
    report = m.benchmark(args)
    assert report["unified"]["threshold"] == 0.35
    assert report["original"]["cup_policy"] == selected_policy
    assert report["original"]["quality_reference_cup_policy"] == selected_policy
    assert calls == ["original", "unified"]
    assert {key: report[key] for key in snapshots} == snapshots


@pytest.mark.parametrize("variant", ["disc0/raw", "disc1/raw"])
def test_explicit_cup_policy_binds_completed_quality_reference(
    benchmark_module, reference_files, variant
):
    m = benchmark_module
    paths, report = reference_files
    report["disc_reference_variant"] = variant
    report["family"] = "optional_fresh_paired_cup_experiment"
    disc = variant.split("/")[0]
    report["fresh_paired_cup_component"] = {
        "cup_variant": disc,
        "reference_context": disc,
        "policy": m.CUP_COMPONENT_POLICY,
    }
    m.validate_reference(report, paths, cup_policy="inside-disc", disc_variant=variant)
    with pytest.raises(ValueError, match="timed decoding"):
        m.validate_reference(report, paths, cup_policy="native", disc_variant=variant)
    with pytest.raises(ValueError, match="explicit"):
        m.validate_reference(report, paths, cup_policy="inside-disc")
    other = "disc1/raw" if variant == "disc0/raw" else "disc0/raw"
    with pytest.raises(ValueError, match="Selected disc"):
        m.validate_reference(report, paths, cup_policy="inside-disc", disc_variant=other)
    report["fresh_paired_cup_component"]["cup_variant"] = other.split("/")[0]
    with pytest.raises(ValueError, match="clipping disc"):
        m.validate_reference(report, paths, cup_policy="inside-disc", disc_variant=variant)


def test_cup_clip_exact_strict_source_semantics_before_restoration(benchmark_module, monkeypatch):
    m = benchmark_module
    native = {
        "vessel": torch.ones(1, 2, 3),
        "disc": torch.tensor([[[0.0, 1.0, -1.0], [1.0, 0.0, 1.0]]]),
        "universal": torch.ones(3, 2, 3),
    }
    native["universal"][2, 1, 2] = 0.0
    restored_inputs = []

    class Geometry:
        def restore(self, values, order):
            assert order == 0
            restored_inputs.append(values.copy())
            return values

    prepared = SimpleNamespace(tensor=lambda: torch.zeros(3, 2, 3), geometry=Geometry())
    monkeypatch.setattr(m, "prepare_single_net", lambda *a, **kw: prepared)
    monkeypatch.setattr(m, "tiled_logits", lambda model, *a, **kw: native[model])
    monkeypatch.setattr(
        m,
        "prepare_vesclass_matlab_calibrated",
        lambda *a, **kw: SimpleNamespace(inputs=lambda: None),
    )
    monkeypatch.setattr(
        m, "predict_av_input", lambda *a, **kw: {"classes": np.ones((2, 3), np.uint8)}
    )
    models = {key: (key, {"tiling": {}}) for key in native}
    rgb = np.zeros((2, 3, 3), np.uint8)
    kwargs = (rgb, np.ones((2, 3), bool), models, None, "cpu", {})
    baseline = m.original_operation(*kwargs)
    clipped = m.original_operation(*kwargs, cup_policy="inside-disc")
    np.testing.assert_array_equal(clipped[4], baseline[4] & baseline[3])
    np.testing.assert_array_equal(clipped[:4], baseline[:4])
    np.testing.assert_array_equal(restored_inputs[-1][2], clipped[4])
    assert not clipped[4, 0, 0] and not clipped[4, 1, 2]  # zero logits remain background.
    assert baseline[4].sum() == 5 and clipped[4].sum() == 2
    assert torch.count_nonzero(native["universal"][2]) == 5  # weights/logits unchanged.


@pytest.mark.parametrize("policy", ["native", "inside-disc"])
def test_structurally_exact_completed_reference_metadata(benchmark_module, monkeypatch, policy):
    import json

    m = benchmark_module
    path = Path(__file__).parent / "reference/completed_cup_policy" / (policy + ".json")
    report = json.loads(path.read_text())
    assert report["family"] == "optional_fresh_paired_cup_experiment"
    assert (
        report["prepared_content_sha256"]
        == "61032be218c623a910c0f3f5e2df3bfdc58e978fe494433a4e6f18d55e41693e"
    )
    keys = {
        "vessel": "vessel_checkpoint_sha256",
        "disc": "disc_checkpoint_sha256",
        "universal": "checkpoint_sha256",
        "av": "av_checkpoint_sha256",
    }
    paths = {key: Path(key + ".pt") for key in keys}
    # Actual model files are never opened: retain the real metadata hash contract.
    monkeypatch.setattr(m, "sha256", lambda path: report[keys[path.stem]])
    m.validate_reference(report, paths, cup_policy=policy, disc_variant="disc0/raw")
    report["prepared_content_sha256"] = "missing"
    with pytest.raises(ValueError, match="canonical"):
        m.validate_reference(report, paths, cup_policy=policy, disc_variant="disc0/raw")


@pytest.mark.parametrize("tta", ["none", "flips2", "rot4", "dihedral8"])
def test_explicit_tta_forwards_after_one_fullframe_photometry(benchmark_module, monkeypatch, tta):
    m = benchmark_module
    events = []
    fov = np.ones((3, 5), bool)
    prepared = SimpleNamespace(
        image=np.zeros((3, 5, 3), np.uint8),
        fov=fov,
        geometry=SimpleNamespace(restore=lambda x, order: x),
    )
    monkeypatch.setattr(m, "prepare_single_net", lambda *a, **k: prepared)

    def photo(image, profile, **kwargs):
        assert image.shape == (3, 5, 3) and profile == "clahe_hsv_v1"
        events.append("fullframe-photometry")
        return image

    def prediction(model, image, device, config, tile, overlap, **kwargs):
        assert image.shape == (3, 3, 5) and (tile, overlap) == (512, 64)
        assert kwargs == ({} if tta == "none" else {"tta": tta})
        events.append("prediction")
        return np.full((5, 3, 5), 0.6, np.float32)

    monkeypatch.setattr(m, "apply_photometry", photo)
    monkeypatch.setattr(m, "predict_probabilities", prediction)
    monkeypatch.setattr(m, "restore_probabilities", lambda p, g: p)
    cfg = SimpleNamespace(
        photometric_profile="clahe_hsv_v1", pixels_per_degree=25.0, tile_size=512, overlap=64
    )
    out = m.unified_operation(
        None,
        fov,
        SimpleNamespace(supervised_channels=[True] * 5),
        cfg,
        "cpu",
        {},
        threshold=0.35,
        tta=tta,
    )
    assert out.shape == (5, 3, 5) and events == ["fullframe-photometry", "prediction"]


def test_invalid_tta_fails_before_device_or_inputs(benchmark_module, monkeypatch):
    m = benchmark_module
    monkeypatch.setattr(m, "resolve_device", lambda *a: pytest.fail("No device"))
    with pytest.raises(ValueError, match="Unsupported"):
        m.benchmark(SimpleNamespace(unified_threshold=0.5, unified_tta="unknown"))


def test_phase_measure_synchronizes_every_recorded_call_without_extra_forward(
    benchmark_module, monkeypatch
):
    import benchmark_latency as timing

    events = []
    monkeypatch.setattr(timing.torch.cuda, "synchronize", lambda *a: events.append("sync"))
    monkeypatch.setattr(timing.torch.cuda, "memory_allocated", lambda *a: 10)
    monkeypatch.setattr(timing.torch.cuda, "reset_peak_memory_stats", lambda *a: None)
    monkeypatch.setattr(timing.torch.cuda, "max_memory_allocated", lambda *a: 14)
    ticks = iter(range(20))
    monkeypatch.setattr(timing.time, "perf_counter", lambda: next(ticks))
    out = timing.measure(
        lambda: events.append("call"),
        SimpleNamespace(type="cuda"),
        warmup=2,
        repeats=3,
        record_warmup=True,
    )
    assert out["warmup_seconds"] == [1, 1] and out["seconds"] == [1, 1, 1]
    assert events.count("call") == 5 and out["peak_working_bytes"] == 4
    assert all(
        events[i - 1] == events[i + 1] == "sync" for i, x in enumerate(events) if x == "call"
    )


def test_historical_measure_default_keeps_warmup_schema(benchmark_module):
    import benchmark_latency as timing

    calls = []
    out = timing.measure(lambda: calls.append(1), torch.device("cpu"), warmup=1, repeats=2)
    assert len(calls) == 3 and "warmup_seconds" not in out


def test_extended_benchmark_records_exact_protocol_load_IO_and_last_output_without_forward(
    tmp_path, benchmark_module, reference_files, monkeypatch
):
    import json

    from PIL import Image

    m = benchmark_module
    paths, reference = reference_files
    image, fov = tmp_path / "train.png", tmp_path / "fov.png"
    Image.fromarray(np.zeros((3, 5, 3), np.uint8)).save(image)
    mask = np.ones((3, 5), np.uint8) * 255
    mask[0, 0] = 0
    Image.fromarray(mask).save(fov)
    unified, ref = tmp_path / "unified.pt", tmp_path / "reference.json"
    unified.write_bytes(b"SYNTHETIC_PLACEHOLDER_NOT_MODEL")
    ref.write_text(json.dumps(reference))

    class Fake:
        supervised_channels = [True] * 5

        def parameters(self):
            return iter([torch.zeros(2)])

        def state_dict(self):
            return {"fixture": torch.zeros(2)}

    channels = {"vessel": ["vessel"], "disc": ["disc"], "universal": ["vessel", "disc", "cup"]}
    import weakref

    references = []

    def original_model():
        model = Fake()
        references.append(weakref.ref(model))
        return model

    monkeypatch.setattr(
        m,
        "load_single_net",
        lambda path, device: (
            original_model(),
            {"channels": channels[path.stem], "tiling": {"crop_size": 288, "border": 36}},
        ),
    )
    monkeypatch.setattr(m, "load_av", lambda *a: (original_model(), {}))

    def load_selected(*args):
        assert len(references) == 4 and all(ref() is None for ref in references)
        return Fake(), {"config": {"purpose": "full_data"}}, torch.device("cpu")

    monkeypatch.setattr(m, "load_unified", load_selected)
    calls, snapshots = [], []
    real_input = m.input_snapshots

    def read(*args):
        snapshots.append(1)
        return real_input(*args)

    monkeypatch.setattr(m, "input_snapshots", read)

    def original(*args, **kwargs):
        calls.append("original")
        return np.zeros((5, 3, 5), bool)

    def selected(*args, **kwargs):
        assert kwargs == {"threshold": 0.35, "tta": "rot4"}
        assert (args[3].tile_size, args[3].overlap) == (640, 80)
        calls.append("unified")
        return np.ones((5, 3, 5), bool)

    monkeypatch.setattr(m, "original_operation", original)
    monkeypatch.setattr(m, "unified_operation", selected)
    args = SimpleNamespace(
        **paths,
        unified=unified,
        reference_report=ref,
        input=image,
        fov=fov,
        dataset="drive",
        field_of_view_degrees=None,
        device="cpu",
        unified_threshold=0.35,
        unified_tta="rot4",
        unified_tile_size=640,
        unified_overlap=80,
        expected_unified_sha256=m.sha256(unified),
        phase_timings=True,
        include_input_io=True,
        original_cup_policy="native",
        original_disc_variant="disc0/raw",
        warmup=1,
        repeats=2,
    )
    report = m.benchmark(args)
    assert calls == ["original"] * 3 + ["unified"] * 3 and len(snapshots) == 7
    assert report["unified"]["tta"]["mode"] == "rot4" and report["unified"]["threshold"] == 0.35
    assert all(v >= 0 for v in report["model_load_seconds"].values())
    assert len(report["unified"]["timing"]["warmup_seconds"]) == 1
    assert report["last_mask_identity_no_extra_forward"]["unified"]["foreground_pixels"] == [15] * 5
    assert report["unified"]["model_tensor_bytes"] == 8 and report["unified"][
        "floating_model_dtypes"
    ] == ["torch.float32"]
    assert "includes input read/decode/SHA" in report["scope"]
    assert report["runtime"]["interop_threads"] == torch.get_num_interop_threads()
    args.expected_unified_sha256 = "f" * 64
    with pytest.raises(ValueError, match="explicit selected byte SHA"):
        m.benchmark(args)


def test_cli_forwards_selected_protocol_threads_and_scopes_without_model_work(
    tmp_path, benchmark_module, monkeypatch
):
    import sys

    m = benchmark_module
    seen = []

    def fake(args):
        assert (
            args.unified_tta,
            args.unified_threshold,
            args.unified_tile_size,
            args.unified_overlap,
        ) == ("rot4", 0.35, 512, 64)
        assert args.expected_unified_sha256 == "a" * 64
        assert args.phase_timings and args.include_input_io
        seen.append("benchmark")
        return {"original": {"timing": {}}, "unified": {"timing": {}}}

    monkeypatch.setattr(m, "benchmark", fake)
    monkeypatch.setattr(m.torch, "set_num_threads", lambda n: seen.append(("intra", n)))
    monkeypatch.setattr(m.torch, "set_num_interop_threads", lambda n: seen.append(("interop", n)))
    monkeypatch.setattr(m.cv2, "setNumThreads", lambda n: seen.append(("opencv", n)))
    command = ["benchmark_strong_pipeline.py"]
    for name in [
        "input",
        "fov",
        "vessel",
        "disc",
        "universal",
        "av",
        "unified",
        "reference-report",
    ]:
        command += ["--" + name, str(tmp_path / name)]
    command += [
        "--output",
        str(tmp_path / "report.json"),
        "--dataset",
        "drive",
        "--original-disc-variant",
        "disc0/raw",
        "--unified-tta",
        "rot4",
        "--unified-threshold",
        ".35",
        "--unified-tile-size",
        "512",
        "--unified-overlap",
        "64",
        "--expected-unified-sha256",
        "a" * 64,
        "--phase-timings",
        "--include-input-io",
        "--cpu-threads",
        "1",
        "--interop-threads",
        "1",
        "--opencv-threads",
        "1",
    ]
    monkeypatch.setattr(sys, "argv", command)
    m.main()
    assert seen == [("intra", 1), ("interop", 1), ("opencv", 1), "benchmark"]


def test_lightweight_suite_does_not_initialize_cuda():
    assert not torch.cuda.is_initialized()
