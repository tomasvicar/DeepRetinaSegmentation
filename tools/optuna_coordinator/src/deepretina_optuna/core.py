"""Single-writer Optuna5 ask/report/prune/tell; no ML imports or transport."""

from __future__ import annotations

import copy
import csv
import datetime
import fcntl
import hashlib
import io
import json
import math
import re
import socket
import sqlite3
import subprocess
import threading
import uuid
from pathlib import Path

import optuna
from optuna.trial import TrialState

RUNGS = (3, 9, 27, 60)


def canonical_json(value):
    return (
        json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n"
    ).encode()


def digest(value):
    return hashlib.sha256(canonical_json(value)).hexdigest()


def require(ok, message):
    if not ok:
        raise ValueError(message)


def utcnow():
    return datetime.datetime.now(datetime.UTC)


def timestamp(value):
    t = datetime.datetime.fromisoformat(value)
    require(t.tzinfo is not None, "UTC timezone required")
    return t.astimezone(datetime.UTC)


def hex64(value):
    return isinstance(value, str) and bool(re.fullmatch("[0-9a-f]{64}", value))


def local_filesystem(path):
    kind = subprocess.check_output(
        ["findmnt", "-n", "-o", "FSTYPE", "--target", str(path)], text=True
    ).strip()
    require(
        kind in {"btrfs", "ext4", "xfs", "tmpfs", "overlay", "zfs"},
        "SQLite must be LOCAL filesystem, not NFS/network: " + kind,
    )
    require("owncloud" not in str(path).lower(), "No database in synchronized ownCloud tree")
    return kind


def validate_params(params, p):
    require(set(params) == set(p["search_space"]), "Search parameter inventory differs")
    for name, spec in p["search_space"].items():
        value = params[name]
        if spec["kind"] == "categorical":
            require(
                type(value) is type(spec["choices"][0]) and value in spec["choices"],
                "Categorical bound: " + name,
            )
        else:
            require(
                type(value) is float
                and math.isfinite(value)
                and spec["low"] <= value <= spec["high"],
                "Continuous bound: " + name,
            )


def config_for(params, profile, p):
    validate_params(params, p)
    require(profile in p["profiles"], "Unknown profile kind")
    config = copy.deepcopy(p["config_template"])
    for name, value in params.items():
        if name != "geometry":
            config[name] = value
    config["patch_size"], config["batch_size"] = map(int, params["geometry"].split("x"))
    config["workers"] = p["profiles"][profile]["workers"]
    config["device"] = "cuda:0"
    require(
        (config["patch_size"], config["batch_size"]) in {(256, 4), (256, 8), (384, 4)},
        "Disallowed coupled geometry",
    )
    return config


def csv_scores(text, epoch, config):
    require(isinstance(text, str), "CSV text required")
    rows = list(csv.DictReader(io.StringIO(text)))
    require(
        len(rows) == epoch and [int(r["epoch"]) for r in rows] == list(range(1, epoch + 1)),
        "CSV not contiguous through actual epoch",
    )
    values = []
    for i, row in enumerate(rows, 1):
        require(all(math.isfinite(float(v)) for v in row.values()), "Nonfinite CSV field")
        require(int(row["steps"]) == i * 8192 // config["batch_size"], "CSV step budget differs")
        value = float(row["val_selection_score"])
        require(
            0 <= value <= 1 and value == float(row["val_macro_dataset_mean_dice"]),
            "Wrong fixed0.5 source-macro objective",
        )
        values.append(value)
    best = max(values)
    return values, best, values.index(best) + 1


class Coordinator:
    def __init__(self, db_path, protocol, profiles, *, clock=utcnow):
        self.clock = clock
        self.protocol = copy.deepcopy(protocol)
        self.profiles = copy.deepcopy(profiles)
        self.guard = threading.RLock()
        self.closed = False
        p = self.protocol
        require(
            p["rungs"] == list(RUNGS) and 1 <= p["max_new_configs"] <= 48, "Invalid study bounds"
        )
        require(
            p["maximum_meta_gpus"] <= 5 and p["maximum_own_gpus"] <= 3,
            "GPU cap exceeds authorization",
        )
        require(
            p["sampler"]["seed"] == 42
            and p["sampler"]["constant_liar"] is True
            and p["pruner"]["min_resource"] == 3
            and p["pruner"]["reduction_factor"] == 3,
            "Wrong TPE/ASHA protocol",
        )
        require(
            p["config_template"]["epochs"] == 60
            and p["config_template"]["epoch_samples"] == 8192
            and p["config_template"]["coverage_first"] is True
            and p["config_template"]["legacy_initialization"] is None
            and p["config_template"]["allow_historical_exposure"] is False,
            "Invalid fixed training budget/lineage",
        )
        require(
            digest(p["plan_template"])
            == "3362cfbd178a3f8ffe4c2a725a3a78e654476b06e9c41e01cf4ec3857cf20982",
            "Approved3157 source/data template changed",
        )
        fixed = {
            k: v
            for k, v in p["config_template"].items()
            if k not in {*p["search_space"], "patch_size", "batch_size", "workers"}
        }
        require(
            digest(fixed) == "a54f19799773f336f0e7286400cd33eb604d3c6c9a46b9ecfdbb624ac80197c9",
            "Fixed nonsearch configuration changed",
        )
        for name, low, high in [
            ("lr", 1e-4, 1e-3),
            ("weight_decay", 1e-6, 1e-3),
            ("av_weight", 0.5, 3.0),
        ]:
            require(
                p["search_space"][name] == {"kind": "float", "low": low, "high": high, "log": True},
                "Continuous search bounds changed",
            )
        require(
            p["search_space"]["geometry"]["choices"] == ["256x4", "256x8", "384x4"],
            "Geometry search bounds changed",
        )
        self.protocol_sha = digest(p)
        self.search_sha = digest(p["search_space"])
        self.deadline = timestamp(p["deadline_utc"])
        for worker, b in self.profiles.items():
            require(
                worker
                and b["profile"] in p["profiles"]
                and hex64(b["profile_sha256"])
                and Path(b["output_root"]).is_absolute(),
                "Invalid approved worker binding",
            )
            if "config_fixed" in b:
                require(
                    b["config_fixed"] == {"workers": p["profiles"][b["profile"]]["workers"]},
                    "Profile worker count differs",
                )
        path = Path(db_path).absolute()
        path.parent.mkdir(parents=True, exist_ok=True)
        self.filesystem = local_filesystem(path.parent)
        self.lock = (path.parent / (path.name + ".lockfile")).open("a+")
        fcntl.flock(self.lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        try:
            with sqlite3.connect(path) as db:
                db.execute("PRAGMA journal_mode=WAL")
                db.execute("PRAGMA synchronous=FULL")
            path.chmod(0o600)
            self.storage = optuna.storages.RDBStorage(
                "sqlite:///" + str(path), engine_kwargs={"connect_args": {"timeout": 10}}
            )
            optuna.logging.set_verbosity(optuna.logging.WARNING)
            self.study = optuna.create_study(
                study_name=p["study_id"],
                storage=self.storage,
                load_if_exists=True,
                direction="maximize",
                sampler=optuna.samplers.TPESampler(seed=42, constant_liar=True, multivariate=True),
                pruner=optuna.pruners.SuccessiveHalvingPruner(
                    min_resource=3, reduction_factor=3, min_early_stopping_rate=0, bootstrap_count=0
                ),
            )
            previous = self.study.user_attrs.get("protocol_sha256")
            require(previous in (None, self.protocol_sha), "Existing study protocol changed")
            bindings = digest(self.profiles)
            require(
                self.study.user_attrs.get("bindings_sha256") in (None, bindings),
                "Approved worker bindings changed",
            )
            self._restore_sampler_state()
            self.study.set_user_attr("protocol_sha256", self.protocol_sha)
            self.study.set_user_attr("bindings_sha256", bindings)
            self.study.set_user_attr(
                "last_session",
                {
                    "uuid": uuid.uuid4().hex,
                    "host": socket.gethostname(),
                    "started_utc": clock().isoformat(),
                    "optuna": optuna.__version__,
                    "sampler_restart_replay_claim": False,
                    "sampler_state_policy": "JSON restore of both pinned Optuna5 RandomState streams; asynchronous history/order is not replayed",
                    "sampler_state_restoration": self.sampler_restoration,
                },
            )
        except Exception:
            self.close()
            raise

    def _rngs(self):
        require(optuna.__version__ == "5.0.0", "Sampler state requires pinned Optuna5.0.0")
        sampler = self.study.sampler
        return {"tpe": sampler._rng.rng, "random_sampler": sampler._random_sampler._rng.rng}

    def _parameter_history(self):
        # Reporting, promotions and session timestamps do not consume sampler RNG.
        return [{"number": t.number, "params": t.params} for t in self._trials()]

    @staticmethod
    def _encode_rng(rng):
        algorithm, keys, position, has_gauss, cached = rng.get_state()
        return {
            "algorithm": algorithm,
            "keys": keys.tolist(),
            "position": position,
            "has_gauss": has_gauss,
            "cached_gaussian": float(cached),
        }

    @staticmethod
    def _decode_rng(value):
        require(
            set(value) == {"algorithm", "keys", "position", "has_gauss", "cached_gaussian"},
            "Sampler RNG field inventory differs",
        )
        require(value["algorithm"] == "MT19937", "Unsupported sampler RNG algorithm")
        keys = value["keys"]
        require(
            isinstance(keys, list)
            and len(keys) == 624
            and all(type(k) is int and 0 <= k < 2**32 for k in keys),
            "Invalid MT19937 keys",
        )
        require(
            type(value["position"]) is int and 0 <= value["position"] <= 624,
            "Invalid MT19937 position",
        )
        require(
            type(value["has_gauss"]) is int and value["has_gauss"] in (0, 1),
            "Invalid cached Gaussian flag",
        )
        require(
            type(value["cached_gaussian"]) is float and math.isfinite(value["cached_gaussian"]),
            "Invalid cached Gaussian value",
        )
        return ("MT19937", keys, value["position"], value["has_gauss"], value["cached_gaussian"])

    def _persist_sampler_state(self, phase="committed"):
        value = {
            "schema_version": 1,
            "optuna_version": optuna.__version__,
            "protocol_sha256": self.protocol_sha,
            "bindings_sha256": digest(self.profiles),
            "history_sha256": digest(self._parameter_history()),
            "phase": phase,
            "streams": {name: self._encode_rng(rng) for name, rng in self._rngs().items()},
        }
        value["sha256"] = digest(value)
        # One SQLite-backed JSON attribute commits both streams together.
        self.study.set_user_attr("sampler_state", value)

    def _restore_sampler_state(self):
        rngs = self._rngs()
        saved = self.study.user_attrs.get("sampler_state")
        history = self._parameter_history()
        if saved is not None:
            require(
                set(saved)
                == {
                    "schema_version",
                    "optuna_version",
                    "protocol_sha256",
                    "bindings_sha256",
                    "history_sha256",
                    "phase",
                    "streams",
                    "sha256",
                },
                "Sampler state field inventory differs",
            )
            payload = {k: v for k, v in saved.items() if k != "sha256"}
            require(saved.get("sha256") == digest(payload), "Sampler state checksum differs")
            require(
                saved["schema_version"] == 1
                and saved["optuna_version"] == optuna.__version__
                and saved["protocol_sha256"] == self.protocol_sha
                and saved["bindings_sha256"] == digest(self.profiles)
                and saved["phase"] in {"committed", "before_ask"},
                "Sampler state pins differ",
            )
            require(set(saved["streams"]) == set(rngs), "Sampler stream inventory differs")
            decoded = {name: self._decode_rng(saved["streams"][name]) for name in rngs}
            if saved["history_sha256"] == digest(history):
                for name, rng in rngs.items():
                    rng.set_state(decoded[name])
                self.sampler_restoration = "exact_saved_streams"
                return
        if not history:
            require(saved is None, "Unexpected sampler state/history mismatch for empty study")
            self.sampler_restoration = "new_study_seed42"
        else:
            # Old versions saved no RNG. An interrupted ask can also leave trial
            # parameters newer than its write-ahead RNG snapshot. Neither case
            # can honestly recover the original streams: derive and record new
            # streams from immutable stored history and the preceding session.
            context = {
                "study_id": self.protocol["study_id"],
                "protocol_sha256": self.protocol_sha,
                "bindings_sha256": digest(self.profiles),
                "history": history,
                "last_session": self.study.user_attrs.get("last_session"),
                "prior_state_sha256": saved.get("sha256") if saved else None,
            }
            context_sha = digest(context)
            seeds = {
                name: int(digest({"context": context_sha, "stream": name})[:8], 16) for name in rngs
            }
            for name, rng in rngs.items():
                if seeds[name] == 42:
                    seeds[name] ^= 0x9E3779B9
                rng.seed(seeds[name])
            event = {
                "schema_version": 1,
                "reason": "missing_legacy_state"
                if saved is None
                else "unclean_ask_history_mismatch",
                "recorded_utc": self.clock().isoformat(),
                "history_sha256": digest(history),
                "context_sha256": context_sha,
                "derivation_context": context,
                "derived_seeds": seeds,
                "original_stream_recovery_claim": False,
            }
            events = self.study.user_attrs.get("sampler_state_migrations", [])
            self.study.set_user_attr("sampler_state_migrations", [*events, event])
            self.sampler_restoration = event["reason"]
        self._persist_sampler_state()

    def close(self):
        if not self.closed:
            self.closed = True
            if hasattr(self, "storage"):
                self.storage.remove_session()
            if hasattr(self, "lock"):
                fcntl.flock(self.lock, fcntl.LOCK_UN)
                self.lock.close()

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()

    def _trial(self, number):
        sid = self.storage.get_study_id_from_name(self.protocol["study_id"])
        tid = self.storage.get_trial_id_from_study_id_trial_number(sid, number)
        return optuna.trial.Trial(self.study, tid)

    def _trials(self):
        return self.study.get_trials(deepcopy=True)

    def _task(self, t, meta, target, previous):
        p = self.protocol
        config = config_for(t.params, meta["profile"], p)
        name = f"trial{t.number:04d}"
        plan = copy.deepcopy(p["plan_template"])
        plan.update(
            manifest="/data/manifest.json",
            screen_epochs=3,
            total_epochs=60,
            candidates={name: config},
        )
        if not p["profiles"][meta["profile"]]["cache"]:
            plan["derived_cache"] = None
        task = {
            "schema_version": 1,
            "study_id": p["study_id"],
            "trial_number": t.number,
            "trial": name,
            "task_id": f"{name}-e{target:03d}",
            "lease": f"{meta['lease']}-e{target:03d}",
            "worker_id": meta["worker_id"],
            "profile": meta["profile"],
            "profile_sha256": meta["profile_sha256"],
            "gpu_uuid": meta.get("gpu_uuid"),
            "target_epoch": target,
            "previous_epoch": previous,
            "resume": previous > 0,
            "plan": plan,
            "plan_sha256": digest(plan),
            "config_sha256": digest(config),
            "protocol_sha256": self.protocol_sha,
            "search_space_sha256": self.search_sha,
            "source_commit": plan["source_commit"],
            "execution_code_sha256": plan["execution_code_sha256"],
            "canonical_sha256": plan["prepared_content_sha256"],
            "cache_sha256": plan["derived_cache"]["fingerprint"] if plan["derived_cache"] else None,
            "deadline_utc": p["deadline_utc"],
            "output": str(Path(meta["output_root"]) / p["study_id"] / name),
            "runtime": copy.deepcopy(p["profiles"][meta["profile"]]),
        }
        return task

    def ask(self, worker_id, gpu_uuid=None):
        with self.guard:
            require(worker_id in self.profiles, "Worker not approved")
            binding = self.profiles[worker_id]
            active = [t for t in self._trials() if t.state == TrialState.RUNNING]
            for frozen in active:
                if frozen.user_attrs.get("meta", {}).get("worker_id") == worker_id:
                    pinned = frozen.user_attrs["meta"].get("gpu_uuid")
                    require(
                        gpu_uuid is None or pinned is None or gpu_uuid == pinned,
                        "Physical GPU migration refused",
                    )
                    return frozen.user_attrs["pending_task"]
            if self.clock() >= self.deadline:
                return None
            require(
                any(
                    t.user_attrs.get("origin") == "manual_baseline"
                    and t.state == TrialState.COMPLETE
                    for t in self._trials()
                ),
                "Import actual reviewed full60 baseline before asking",
            )
            new = [t for t in self._trials() if t.user_attrs.get("origin") == "optuna_new"]
            if len(new) >= self.protocol["max_new_configs"]:
                return None
            kind = binding["profile"]
            meta_count = sum(t.user_attrs.get("meta", {}).get("profile") == "meta" for t in active)
            own_count = len(active) - meta_count
            if (kind == "meta" and meta_count >= self.protocol["maximum_meta_gpus"]) or (
                kind != "meta" and own_count >= self.protocol["maximum_own_gpus"]
            ):
                return None
            if gpu_uuid:
                require(
                    not any(
                        t.user_attrs.get("meta", {}).get("gpu_uuid") == gpu_uuid for t in active
                    ),
                    "Physical GPU already leased",
                )
            seed_used = any(t.user_attrs.get("seed_candidate") for t in self._trials())
            if kind == self.protocol["seed_candidate_profile"] and not seed_used:
                self.study.enqueue_trial(
                    self.protocol["seed_candidate"], user_attrs={"seed_candidate": True}
                )
            self._persist_sampler_state(phase="before_ask")
            t = self.study.ask()
            t.set_user_attr("origin", "optuna_new")
            params = {}
            for name, spec in sorted(self.protocol["search_space"].items()):
                params[name] = (
                    t.suggest_categorical(name, spec["choices"])
                    if spec["kind"] == "categorical"
                    else t.suggest_float(name, spec["low"], spec["high"], log=spec["log"])
                )
            validate_params(params, self.protocol)
            meta = {
                "worker_id": worker_id,
                "profile": kind,
                "profile_sha256": binding["profile_sha256"],
                "output_root": binding["output_root"],
                "gpu_uuid": gpu_uuid or binding.get("gpu_uuid"),
                "lease": uuid.uuid4().hex,
            }
            t.set_user_attr("origin", "optuna_new")
            t.set_user_attr("meta", meta)
            task = self._task(t, meta, 3, 0)
            t.set_user_attr("pending_task", task)
            self._persist_sampler_state()
            return task

    def _validate_receipt(self, r, task):
        for key in [
            "schema_version",
            "study_id",
            "task_id",
            "trial_number",
            "trial",
            "lease",
            "worker_id",
            "profile_sha256",
            "plan_sha256",
            "config_sha256",
            "protocol_sha256",
            "search_space_sha256",
            "target_epoch",
        ]:
            require(r.get(key) == task[key], "Receipt identity/pin differs: " + key)
        require(
            isinstance(r.get("gpu_uuid"), str) and bool(r["gpu_uuid"]), "Observed GPU UUID required"
        )
        if task.get("gpu_uuid"):
            require(r["gpu_uuid"] == task["gpu_uuid"], "Physical GPU migration refused")
        require(r["status"] in {"COMPLETE", "FAILED"}, "Wrong worker stage status")
        if r["status"] == "FAILED":
            return None
        q = r["checkpoint_qa"]
        require(
            type(q["completed_budget"]) is bool
            and type(q["finite_weights"]) is bool
            and type(q["finite_optimizer"]) is bool,
            "QA flags must be actual booleans",
        )
        epoch = task["target_epoch"]
        cfg = task["plan"]["candidates"][task["trial"]]
        require(
            q["status"] == "PASS" and q["actual_epoch"] == r["actual_epoch"] == epoch,
            "Actual rung QA missing",
        )
        for key, value in {
            "finite_weights": True,
            "finite_optimizer": True,
            "scheduler_T_max": 60,
            "scheduler_last_epoch": epoch,
            "steps": epoch * 8192 // cfg["batch_size"],
            "delivered_samples": epoch * 8192,
            "eligible_photos": 2221,
            "seen_photos": 2221,
            "completed_budget": epoch == 60,
            "plan_sha256": task["plan_sha256"],
            "config_sha256": task["config_sha256"],
            "source_commit": task["source_commit"],
            "execution_code_sha256": task["execution_code_sha256"],
            "training_source_sha256": task["plan"]["training_source_sha256"],
            "manifest_sha256": task["plan"]["manifest_sha256"],
            "canonical_sha256": task["canonical_sha256"],
            "cache_sha256": task["cache_sha256"],
            "profile_sha256": task["profile_sha256"],
            "gpu_uuid": r["gpu_uuid"],
            "worker_id": task["worker_id"],
            "output": task["output"],
        }.items():
            require(q.get(key) == value, "CheckpointQA differs: " + key)
        require(q["minimum_photo_visits"] >= epoch, "Coverage minimum below epoch")
        for key in ["checkpoint_sha256", "best_checkpoint_sha256", "csv_sha256"]:
            require(hex64(q[key]), "Missing artifact SHA: " + key)
        require(
            hashlib.sha256(q["csv_text"].encode()).hexdigest() == q["csv_sha256"],
            "CSV bytes differ",
        )
        _, score, best_epoch = csv_scores(q["csv_text"], epoch, cfg)
        require(
            type(q["metric"]) is float
            and math.isfinite(q["metric"])
            and q["metric"] == score == r["metric"]
            and q["best_epoch"] == best_epoch,
            "Best-so-far score/checkpoint disagrees with actual CSV",
        )
        return score

    def report(self, r):
        with self.guard:
            require(type(r.get("trial_number")) is int, "Integer trial number required")
            t = self._trial(r["trial_number"])
            receipt_sha = digest(r)
            attrs = t.user_attrs
            cached = attrs.get("last_decision")
            if attrs.get("last_receipt_sha256") == receipt_sha:
                if cached["action"] in {"pruned", "finished"}:
                    frozen = self.storage.get_trial(
                        self.storage.get_trial_id_from_study_id_trial_number(
                            self.storage.get_study_id_from_name(self.protocol["study_id"]), t.number
                        )
                    )
                    if frozen.state == TrialState.RUNNING:
                        self.study.tell(
                            t.number,
                            cached.get("value") if cached["trial_state"] == "COMPLETE" else None,
                            state=TrialState[cached["trial_state"]],
                        )
                return cached
            require(
                self.storage.get_trial(
                    self.storage.get_trial_id_from_study_id_trial_number(
                        self.storage.get_study_id_from_name(self.protocol["study_id"]), t.number
                    )
                ).state
                == TrialState.RUNNING,
                "Trial already finished",
            )
            task = attrs["pending_task"]
            score = self._validate_receipt(r, task)
            meta = attrs["meta"]
            meta["gpu_uuid"] = r["gpu_uuid"]
            t.set_user_attr("meta", meta)
            ended = timestamp(r["completed_utc"])
            require(
                ended <= self.clock() + datetime.timedelta(minutes=5), "Future worker timestamp"
            )
            state = None
            if score is None:
                action = "pruned"
                state = TrialState.FAIL
                reason = "worker_failed"
            else:
                epoch = task["target_epoch"]
                t.report(score, step=epoch)
                if epoch == 60 and ended <= self.deadline:
                    action = "finished"
                    state = TrialState.COMPLETE
                    reason = "actual60_QA"
                elif self.clock() >= self.deadline or ended > self.deadline:
                    action = "pruned"
                    state = TrialState.PRUNED
                    reason = "deadline_partial_not_final"
                elif t.should_prune():
                    action = "pruned"
                    state = TrialState.PRUNED
                    reason = "successive_halving"
                else:
                    action = "continue"
                    reason = "promoted"
            decision = {
                "schema_version": 1,
                "study_id": self.protocol["study_id"],
                "trial_number": t.number,
                "task_id": task["task_id"],
                "lease": task["lease"],
                "action": action,
                "reason": reason,
                "trial_state": state.name if state else "RUNNING",
                "value": score,
                "receipt_sha256": receipt_sha,
            }
            if action == "continue":
                i = RUNGS.index(task["target_epoch"])
                next_task = self._task(t, meta, RUNGS[i + 1], task["target_epoch"])
                require(
                    next_task["plan_sha256"] == task["plan_sha256"]
                    and next_task["config_sha256"] == task["config_sha256"],
                    "Plan/config changed across rungs",
                )
                t.set_user_attr("pending_task", next_task)
                decision["next_task"] = next_task
            t.set_user_attr("last_receipt_sha256", receipt_sha)
            t.set_user_attr("last_decision", decision)
            if state is not None:
                self.study.tell(
                    t.number, score if state == TrialState.COMPLETE else None, state=state
                )
            return decision

    def import_baseline(self, csv_text, qa, qa_sha256):
        with self.guard:
            b = self.protocol["manual_baseline"]
            payload = (json.dumps(qa, indent=2) + "\n").encode() if isinstance(qa, dict) else qa
            require(
                hashlib.sha256(payload).hexdigest() == qa_sha256 == b["qa_sha256"],
                "Require exact actual full60 QA bytes",
            )
            q = json.loads(payload)
            require(
                q["source_commit"] == self.protocol["plan_template"]["source_commit"]
                and q["last_epoch"] == 60
                and q["steps"] == 122880
                and q["delivered_samples"] == 491520
                and q["eligible_photos"] == 2221
                and q["minimum_photo_visits"] >= 60
                and q["finite_best_last_weights_and_last_optimizer"] is True,
                "Incomplete/manual legacy baseline refused",
            )
            require(
                q["checkpoint_sha256"]["best.pt"] == b["best_checkpoint_sha256"]
                and q["checkpoint_sha256"]["last.pt"] == b["last_checkpoint_sha256"],
                "Baseline checkpoint pins differ",
            )
            require(
                hashlib.sha256(csv_text.encode()).hexdigest() == q["sidecar_sha256"]["metrics.csv"],
                "Baseline60 CSV bytes differ",
            )
            values, score, best_epoch = csv_scores(csv_text, 60, self.protocol["config_template"])
            require(
                score == q["best_score"] and best_epoch == q["best_epoch"],
                "Baseline objective differs",
            )
            for frozen in self._trials():
                if frozen.user_attrs.get("origin") == "manual_baseline":
                    require(frozen.user_attrs["qa_sha256"] == qa_sha256, "Baseline already differs")
                    return frozen.number
            params = {
                k: self.protocol["config_template"][k]
                for k in self.protocol["search_space"]
                if k != "geometry"
            }
            params["geometry"] = "384x4"
            validate_params(params, self.protocol)
            distributions = {
                name: optuna.distributions.CategoricalDistribution(s["choices"])
                if s["kind"] == "categorical"
                else optuna.distributions.FloatDistribution(s["low"], s["high"], log=s["log"])
                for name, s in self.protocol["search_space"].items()
            }
            frozen = optuna.trial.create_trial(
                state=TrialState.COMPLETE,
                value=score,
                params=params,
                distributions=distributions,
                intermediate_values={i: max(values[:i]) for i in range(1, 61)},
                user_attrs={
                    "origin": "manual_baseline",
                    "qa_sha256": qa_sha256,
                    "best_checkpoint_sha256": b["best_checkpoint_sha256"],
                    "last_checkpoint_sha256": b["last_checkpoint_sha256"],
                    "best_epoch": best_epoch,
                    "fixed_threshold": 0.5,
                    "excluded_from_new_config_budget": True,
                },
            )
            self.study.add_trial(frozen)
            self._persist_sampler_state()
            return self._trials()[-1].number

    def status(self):
        with self.guard:
            return {
                "study_id": self.protocol["study_id"],
                "protocol_sha256": self.protocol_sha,
                "search_space_sha256": self.search_sha,
                "deadline_utc": self.protocol["deadline_utc"],
                "sampler_state_sha256": self.study.user_attrs["sampler_state"]["sha256"],
                "sampler_state_restoration": self.sampler_restoration,
                "sampler_state_migrations": self.study.user_attrs.get(
                    "sampler_state_migrations", []
                ),
                "new_configs": sum(
                    t.user_attrs.get("origin") == "optuna_new" for t in self._trials()
                ),
                "trials": [
                    {
                        "number": t.number,
                        "state": t.state.name,
                        "value": t.value,
                        "origin": t.user_attrs.get("origin"),
                        "pending_task": t.user_attrs.get("pending_task")
                        if t.state == TrialState.RUNNING
                        else None,
                    }
                    for t in self._trials()
                ],
            }
