"""PBS admission/provenance fixtures only; no CUDA/model/data benchmark."""

import importlib.util
import json
from pathlib import Path

import pytest

ROOT = Path(__file__).parents[1]
spec = importlib.util.spec_from_file_location(
    "pbs_bench", ROOT / "scripts/benchmark_unified_bf16.py"
)
b = importlib.util.module_from_spec(spec)
spec.loader.exec_module(b)


@pytest.fixture
def context(tmp_path, monkeypatch):
    root = tmp_path / "frozen72"
    root.mkdir()
    proof = tmp_path / "witness.json"
    uuid = "GPU-11111111-2222-3333-4444-555555555555"
    args = b.parser().parse_args(
        [
            "--execute",
            "--execution-target",
            "pbs",
            "--source-root",
            str(root),
            "--pbs-proof",
            str(proof),
            "--pbs-proof-sha256",
            "pending",
            "--expected-gpu-uuid",
            uuid,
            "--parent",
            str(tmp_path / "parent"),
            "--manifest",
            str(tmp_path / "manifest"),
            "--cache",
            str(tmp_path / "cache"),
            "--output",
            str(tmp_path / "report.json"),
        ]
    )
    document = {
        "schema_version": 1,
        "kind": "verified_host_source_attestation",
        "bindings": {
            "source_commit": b.TRAIN_SOURCE,
            "execution_code_sha256": b.TRAIN_EXECUTION,
            "training_source_sha256": b.TRAIN_HASH,
            "image_sha256": b.SIF,
            "git_status_porcelain": "",
            "container_source": str(root),
            "host_source": str(root),
        },
        "execution": {
            "kind": "pbs_bf16_core_benchmark",
            "pbs_job_id": "fixture.job",
            "original_cuda_visible_devices": "2",
            "gpu_uuid": uuid,
            "parent_sha256": b.PARENT,
            "benchmark_helper_sha256": b.digest(Path(b.__file__).read_bytes()),
            "torch_threads": 8,
            "torch_interop_threads": 1,
        },
    }
    raw = json.dumps(document).encode()
    proof.write_bytes(raw)
    proof.chmod(0o444)
    args.pbs_proof_sha256 = b.digest(raw)
    monkeypatch.setenv("PBS_JOBID", "fixture.job")
    monkeypatch.setenv("RETINA_ORIGINAL_PBS_CUDA_MASK", "2")
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", uuid)
    monkeypatch.setattr(b, "executable_digest", lambda p: b.TRAIN_EXECUTION)
    monkeypatch.setattr(b.u, "_training_source_sha256", lambda: b.TRAIN_HASH)
    return args, document


def test_explicit_pbs_source_and_proof_keep_training72_separate(context):
    args, _ = context
    checked = b.pbs_context(args)
    assert checked["root"] == args.source_root
    assert checked["proof"]["bindings"]["source_commit"] == b.TRAIN_SOURCE


@pytest.mark.parametrize(
    "change", ["sha", "writable", "job", "mask", "source", "helper", "threads"]
)
def test_bad_pbs_proof_rejects_before_gpu_or_data(context, change):
    args, doc = context
    if change == "sha":
        args.pbs_proof_sha256 = "0" * 64
    elif change == "writable":
        args.pbs_proof.chmod(0o644)
    else:
        if change == "job":
            doc["execution"]["pbs_job_id"] = "wrong"
        if change == "mask":
            doc["execution"]["original_cuda_visible_devices"] = "0"
        if change == "source":
            doc["bindings"]["source_commit"] = "0" * 40
        if change == "helper":
            doc["execution"]["benchmark_helper_sha256"] = "0" * 64
        if change == "threads":
            doc["execution"]["torch_threads"] = 20
        args.pbs_proof.chmod(0o644)
        raw = json.dumps(doc).encode()
        args.pbs_proof.write_bytes(raw)
        args.pbs_proof.chmod(0o444)
        args.pbs_proof_sha256 = b.digest(raw)
    with pytest.raises(ValueError):
        b.pbs_context(args)
    assert not args.output.exists()


def test_ada_default_is_still_explicit_and_does_not_accept_pbs_flags(tmp_path, monkeypatch):
    args = b.parser().parse_args(
        [
            "--execute",
            "--parent",
            "missing",
            "--manifest",
            "missing",
            "--cache",
            "missing",
            "--output",
            str(tmp_path / "out"),
        ]
    )
    assert args.execution_target == "ada" and args.expected_gpu_uuid == b.GPU
    monkeypatch.setattr(b.socket, "gethostname", lambda: "hpc-not-ada")
    with pytest.raises(ValueError, match="Own Ada only"):
        b.admission(args)
