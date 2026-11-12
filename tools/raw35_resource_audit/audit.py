"""Disabled-by-default retained RAW35 resource hashing. Stdlib only; no decode/extract."""

import argparse
import hashlib
import importlib.util
import json
import os
import platform
import re
import resource
import stat
import subprocess
import time
import zipfile
from contextlib import contextmanager
from pathlib import Path

CHUNK = 1024**2
MAX_BYTES = 128 * 1024**3
MAX_OUTPUT = 32 * 1024**2
MAX_CD = 64 * 1024**2
MAX_ENTRIES = 200000
MAX_NAMES = 32 * 1024**2
RESOLVER_SHA = "840c482a1a9fcb531b0fea0271e20405d51ac1d3a705fa5ba3c242bb488cde92"
PRODUCER = "f9d5e666f6ff789be44cbefbaf8d7a0f2284caf6"
MEMBERSHIP = "08ef48ec32109c8971d5690066ac48781b768e2b3b8241ee1880513f66b828b5"
TRAIN = {
    "rimone",
    "onhsd",
    "gamma",
    "riga",
    "iostar",
    "refuge",
    "erlangen",
    "rite",
    "avrdb",
    "g1020",
    "chasedb1",
    "uoadr",
    "idrid_disc",
    "drive",
    "biomisa",
    "drishti",
    "hrf",
    "drhagis",
    "aria",
    "drions",
    "stare",
}
VAL = TRAIN - {"biomisa", "drions", "gamma", "idrid_disc", "riga", "uoadr"}


def need(ok, message):
    if not ok:
        raise ValueError(message)


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def resolver():
    p = Path(__file__).with_name("resolver.py")
    need(digest(p) == RESOLVER_SHA, "Frozen resolver bytes differ")
    s = importlib.util.spec_from_file_location("f9d_resolver", p)
    m = importlib.util.module_from_spec(s)
    s.loader.exec_module(m)
    return m


def duration(value):
    parts = re.findall(r"([0-9.]+)(h|min|ms|us|s)", value)
    need(
        parts and not re.sub(r"[0-9.]+(?:h|min|ms|us|s)|\s+", "", value),
        "Unsupported unit duration",
    )
    return sum(
        float(n) * {"h": 3600, "min": 60, "s": 1, "ms": 0.001, "us": 0.000001}[u] for n, u in parts
    )


class Runtime:
    def __init__(self, profile):
        unit = profile["unit"]
        need(
            re.fullmatch(r"deepretina-raw35-resource-audit-[A-Za-z0-9-]+\.service", unit),
            "Own bounded unit required",
        )
        props = dict(
            line.split("=", 1)
            for line in subprocess.check_output(
                [
                    "systemctl",
                    "--user",
                    "show",
                    unit,
                    "--property=ControlGroup,MemoryMax,MemorySwapMax,CPUQuotaPerSecUSec,RuntimeMaxUSec,KillMode",
                ],
                text=True,
            ).splitlines()
        )
        group = next(
            x.split(":", 2)[2]
            for x in Path("/proc/self/cgroup").read_text().splitlines()
            if x.startswith("0::")
        )
        need(
            group == props["ControlGroup"] and group.endswith("/" + unit),
            "Not inside reviewed unit",
        )
        cg = Path("/sys/fs/cgroup") / group.lstrip("/")
        actual = {
            n: (cg / n).read_text().strip() for n in ["memory.max", "memory.swap.max", "cpu.max"]
        }
        quota, period = map(int, actual["cpu.max"].split())
        need(
            props["MemoryMax"] == actual["memory.max"] == str(2 * 1024**3)
            and props["MemorySwapMax"] == actual["memory.swap.max"] == "0",
            "Actual2GiB/noSwap required",
        )
        need(
            duration(props["CPUQuotaPerSecUSec"]) == 1
            and quota == period
            and duration(props["RuntimeMaxUSec"]) == 1800
            and props["KillMode"] == "control-group",
            "Actual CPU1/external1800s required",
        )
        need(
            list(os.sched_getaffinity(0)) == [profile["cpu_core"]]
            and os.getpriority(os.PRIO_PROCESS, 0) == 17,
            "Actual affinity1/Nice17 required",
        )
        need(
            subprocess.check_output(["ionice", "-p", str(os.getpid())], text=True).strip()
            == "idle",
            "Actual idle IO required",
        )
        need(os.environ.get("CUDA_VISIBLE_DEVICES") == "", "CUDA must be hidden")
        self.proof = {
            "unit": unit,
            "actual_systemd": props,
            "actual_cgroup": actual,
            "pid": os.getpid(),
            "core": profile["cpu_core"],
            "nice": 17,
            "IO": "idle",
        }

    def check(self):
        need(
            resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024 <= 2 * 1024**3,
            "RSS ceiling exceeded",
        )


class Budget:
    def __init__(self, runtime, *, clock=time.monotonic, maximum=MAX_BYTES):
        self.runtime, self.clock, self.started = runtime, clock, clock()
        self.maximum = maximum
        self.payload = self.container = 0

    def before(self, size, kind):
        need(type(size) is int and 0 <= size <= CHUNK, "Read chunk exceeds1MiB")
        self.runtime.check()
        need(self.clock() - self.started < 1800, "Audit time ceiling exceeded")
        need(getattr(self, kind) + size <= self.maximum, "Audit byte ceiling exceeded")

    def after(self, size, kind):
        setattr(self, kind, getattr(self, kind) + size)
        need(getattr(self, kind) <= self.maximum, "Audit byte ceiling exceeded")


class CountedFile:
    """Bound every stdlib ZIP container read, including central-directory reads."""

    def __init__(self, stream, budget):
        self.stream, self.budget = stream, budget

    def read(self, size=-1):
        need(type(size) is int, "Invalid container read")
        if size < 0:
            here = self.stream.tell()
            self.stream.seek(0, os.SEEK_END)
            size = self.stream.tell() - here
            self.stream.seek(here)
            need(size <= MAX_CD, "Unbounded metadata read exceeds central-directory ceiling")
        chunks = []
        while size:
            n = min(size, CHUNK)
            self.budget.before(n, "container")
            b = self.stream.read(n)
            self.budget.after(len(b), "container")
            chunks.append(b)
            if len(b) < n:
                break
            size -= len(b)
        return b"".join(chunks)

    def seek(self, *args):
        return self.stream.seek(*args)

    def tell(self):
        return self.stream.tell()

    def seekable(self):
        return True


def clean_relative(name):
    p = Path(name)
    need(
        isinstance(name, str)
        and bool(name)
        and not p.is_absolute()
        and ".." not in p.parts
        and "\x00" not in name,
        "Resource path escape",
    )
    return p


def file_stat(st):
    return [st.st_dev, st.st_ino, st.st_size, st.st_mtime_ns, st.st_ctime_ns]


@contextmanager
def open_member(root, relative):
    """Walk directory components with O_NOFOLLOW; never follow an intermediate link."""
    parts = clean_relative(relative).parts
    fd = os.open(root, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
    try:
        for part in parts[:-1]:
            child = os.open(part, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW, dir_fd=fd)
            os.close(fd)
            fd = child
        before = os.stat(parts[-1], dir_fd=fd, follow_symlinks=False)
        need(stat.S_ISREG(before.st_mode), "Regular retained resource required before open")
        final = os.open(parts[-1], os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK, dir_fd=fd)
        with os.fdopen(final, "rb") as stream:
            actual = os.fstat(stream.fileno())
            need(
                stat.S_ISREG(actual.st_mode) and file_stat(actual) == file_stat(before),
                "Regular retained resource type/stat changed at open",
            )
            yield stream
    finally:
        os.close(fd)


def hash_stream(stream, size, budget):
    need(type(size) is int and 0 < size <= budget.maximum, "Invalid/overbudget member size")
    need(budget.payload + size <= budget.maximum, "Retained payload exceeds remaining byte ceiling")
    h = hashlib.sha256()
    remaining = size
    while remaining:
        n = min(CHUNK, remaining)
        budget.before(n, "payload")
        block = stream.read(n)
        budget.after(len(block), "payload")
        need(0 < len(block) <= n, "Truncated member payload")
        h.update(block)
        remaining -= len(block)
    return {"size": size, "sha256": h.hexdigest()}


def zip_index(stream, budget, r):
    counted = CountedFile(stream, budget)
    # Same stdlib EOCD/ZIP64 interpretation as ZipFile, before allocating its index.
    end = zipfile._EndRecData(counted)
    need(end is not None, "Missing ZIP EOCD")
    need(
        end[zipfile._ECD_DISK_NUMBER] == end[zipfile._ECD_DISK_START] == 0,
        "Multi-disk ZIP unsupported",
    )
    need(
        end[zipfile._ECD_ENTRIES_TOTAL] <= MAX_ENTRIES and end[zipfile._ECD_SIZE] <= MAX_CD,
        "Central directory exceeds metadata ceiling",
    )
    z = zipfile.ZipFile(counted)
    entries = {}
    identities = {}
    name_bytes = 0
    try:
        infos = z.infolist()
        need(len(infos) <= MAX_ENTRIES, "ZIP entry ceiling exceeded")
        for info in infos:
            budget.before(0, "container")
            need(not stat.S_ISLNK(info.external_attr >> 16), "ZIP symlink forbidden")
            name = r.canonical_path(info.filename)
            clean_relative(name)
            name_bytes += len(name.encode())
            need(name_bytes <= MAX_NAMES, "Index name ceiling exceeded")
            if (
                info.is_dir()
                or "__MACOSX" in name
                or any(p.startswith(".") for p in Path(name).parts)
            ):
                continue
            identity = (info.file_size, info.CRC)
            need(
                name not in identities or identities[name] == identity,
                "Conflicting canonical ZIP duplicate CRC/size",
            )
            entries.setdefault(name, info)
            identities.setdefault(name, identity)
        return z, entries, identities
    except BaseException:
        z.close()
        raise


def directory_index(root, r, budget):
    entries = {}
    total_names = 0
    count = 0
    for folder, dirs, files in os.walk(root, followlinks=False):
        budget.before(0, "container")
        for leaf in [*dirs, *files]:
            budget.before(0, "container")
            p = Path(folder) / leaf
            name = r.canonical_path(p.relative_to(root).as_posix())
            clean_relative(name)
            count += 1
            total_names += len(name.encode())
            need(
                count <= MAX_ENTRIES and total_names <= MAX_NAMES,
                "Directory metadata ceiling exceeded",
            )
            need(not p.is_symlink(), "Directory symlink forbidden")
        for leaf in sorted(files):
            p = Path(folder) / leaf
            relative = p.relative_to(root).as_posix()
            name = r.canonical_path(relative)
            clean_relative(name)
            if "__MACOSX" in name or any(x.startswith(".") for x in Path(name).parts):
                continue
            # Unlike ZIP there is no CRC metadata to prove a duplicate harmless.
            need(name not in entries, "Ambiguous canonical directory duplicate")
            entries[name] = relative
    return entries


def audit_requests(requests, source_root, budget):
    r = resolver()
    root = Path(source_root).absolute()
    need(root == root.resolve() and root.is_dir(), "Source root must be real, nonsymlink directory")
    resources = {}
    stats = {}
    for archive, wanted in sorted(requests.items()):
        clean_relative(archive)
        path = root / archive
        need(
            not path.is_symlink()
            and path == path.resolve()
            and path.resolve().is_relative_to(root),
            "Archive path escape/symlink",
        )
        before = file_stat(path.stat())
        if path.is_dir():
            index = directory_index(path, r, budget)
            mapping = r.resolve_requests(list(index), wanted)
            members = {}
            for name in sorted(set(mapping.values())):
                with open_member(path, index[name]) as f:
                    old = file_stat(os.fstat(f.fileno()))
                    members[name] = hash_stream(f, old[2], budget)
                    need(
                        old == file_stat(os.fstat(f.fileno())),
                        "Directory member changed during audit",
                    )
                need(
                    old == file_stat((path / index[name]).stat(follow_symlinks=False)),
                    "Directory member path changed",
                )
        else:
            with open_member(root, archive) as f:
                need(before == file_stat(os.fstat(f.fileno())), "Archive changed before open")
                z, index, identities = zip_index(f, budget, r)
                with z:
                    mapping = r.resolve_requests(list(index), wanted, identities)
                    members = {}
                    for name in sorted(set(mapping.values())):
                        info = index[name]
                        need(not info.flag_bits & 1, "Encrypted retained member forbidden")
                        need(
                            budget.payload + info.file_size <= budget.maximum,
                            "Retained member byte ceiling exceeded",
                        )
                        budget.before(0, "payload")
                        with z.open(info, "r") as member:
                            members[name] = hash_stream(member, info.file_size, budget)
                need(before == file_stat(os.fstat(f.fileno())), "Archive changed during audit")
        need(before == file_stat(path.stat()), "Source path changed during audit")
        resources[archive] = {"request_to_member": mapping, "members": members}
        stats[archive] = {"before": before, "after": file_stat(path.stat())}
    return {
        "resources": resources,
        "source_stats": stats,
        "member_payload_bytes": budget.payload,
        "container_bytes_read": budget.container,
        "seconds": budget.clock() - budget.started,
    }


def check_plan(plan):
    need(
        plan["producer_commit"] == PRODUCER and plan["membership_sha256"] == MEMBERSHIP,
        "Frozen producer/membership differs",
    )
    need(
        plan["manifest_sha256"]
        == "17a2277c39cb37b7eee1b2063cfd13f2cf4adf4ebdf86cf449a1f51367c5a767"
        and plan["photos_sha256"]
        == "765709a1ddc08035565ebf3d54d414dd1f53a9882a5cfa9c0fd03811b8e48a4d",
        "Frozen metadata pins differ",
    )
    need(
        plan["records"] == {"train": 5700, "val": 993}
        and plan["photos"] == {"train": 2529, "val": 770}
        and plan["VAL_groups"] == 612,
        "Frozen TRAIN/VAL scope differs",
    )
    need(
        set(plan["row_sources"]["train"]) == TRAIN and set(plan["row_sources"]["val"]) == VAL,
        "Source inventory differs",
    )
    requests = plan["requests"]
    need(
        len(requests) == 22 and sum(map(len, requests.values())) == 10999,
        "Exact22/10999 request scope required",
    )
    for archive, items in requests.items():
        clean_relative(archive)
        need(isinstance(items, dict) and items, "Explicit requested-resource map required")
        for name, roles in items.items():
            clean_relative(name)
            need(
                isinstance(roles, list) and roles and all(isinstance(x, str) and x for x in roles),
                "Explicit resource roles required",
            )


def save_exclusive(output, value):
    output = Path(output)
    need(not output.exists() and not output.is_symlink(), "Never overwrite audit output")
    partial = output.with_name(output.name + ".partial")
    total = 0
    with partial.open("x") as f:
        for text in json.JSONEncoder(indent=2, allow_nan=False).iterencode(value):
            b = text.encode()
            total += len(b)
            need(total <= MAX_OUTPUT, "Output32MiB ceiling exceeded")
            f.write(text)
        f.write("\n")
        need(total + 1 <= MAX_OUTPUT, "Output32MiB ceiling exceeded")
        f.flush()
        os.fsync(f.fileno())
    os.link(partial, output)
    partial.unlink()
    output.chmod(0o444)


def small_json(path, limit):
    path = Path(path)
    need(
        not path.is_symlink() and path.stat().st_size <= limit,
        "Metadata file exceeds ceiling/symlink",
    )
    with path.open("rb") as f:
        raw = f.read(limit + 1)
    need(len(raw) <= limit, "Metadata file grew beyond ceiling")
    return raw, json.loads(raw)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    for name in ["profile", "trusted-profile-sha256", "plan", "source-root", "output"]:
        ap.add_argument("--" + name, required=True)
    a = ap.parse_args(argv)
    raw, p = small_json(a.profile, CHUNK)
    need(
        hashlib.sha256(raw).hexdigest() == a.trusted_profile_sha256, "External profile pin differs"
    )
    need(
        p.get("execution_authorized") is True and p.get("root_reviewed") is True,
        "Disabled; no plan or source read",
    )
    need(p.get("kind") == "raw35_retained_encoded_resource_audit", "Wrong audit profile kind")
    need(
        p["tool_sha256"] == digest(__file__) and p["resolver_sha256"] == RESOLVER_SHA,
        "Reviewed tool/resolver differs",
    )
    need(
        p["limits"]
        == {
            "external_seconds": 1800,
            "memory_bytes": 2 * 1024**3,
            "swap_bytes": 0,
            "CPU_quota_percent": 100,
            "payload_bytes": MAX_BYTES,
            "output_bytes": MAX_OUTPUT,
        },
        "Exact ceiling contract differs",
    )
    need(platform.python_version() == p["python_version"], "Reviewed stdlib Python version differs")
    need(
        all(type(v) is int for v in p["limits"].values()),
        "Integer ceilings required; no bool aliases",
    )
    runtime = Runtime(p)
    budget = Budget(runtime)
    raw, plan = small_json(a.plan, MAX_OUTPUT)
    need(hashlib.sha256(raw).hexdigest() == p["plan_sha256"], "Externally pinned plan differs")
    check_plan(plan)
    need(
        str(Path(a.source_root).absolute()) == p["source_root"]
        and str(Path(a.output).absolute()) == p["output"],
        "Reviewed source/output paths differ",
    )
    out = Path(a.output).absolute()
    source = Path(a.source_root).resolve()
    need(
        out.parent.resolve() == Path(p["output_parent"]).resolve()
        and out.parent.is_dir()
        and not out.resolve().is_relative_to(source),
        "Exclusive local output outside source required",
    )
    need(
        out.parent.absolute() == out.parent.resolve() and "owncloud" not in str(out).lower(),
        "Output must be nonsymlink local audit namespace",
    )
    need(
        not out.exists()
        and not out.is_symlink()
        and not out.with_name(out.name + ".partial").exists()
        and not out.with_name(out.name + ".partial").is_symlink(),
        "Exclusive output and partial required before audit",
    )
    result = audit_requests(plan["requests"], a.source_root, budget)
    result.update(
        kind="actual_retained_RAW35_encoded_resource_audit_NOT_preparation",
        profile_sha256=a.trusted_profile_sha256,
        tool_sha256=digest(__file__),
        source_root=str(source),
        scope_provenance="Externally reviewed producer TRAIN/VAL request plan; no filename-based TEST inference",
        plan_sha256=p["plan_sha256"],
        producer_commit=PRODUCER,
        resolver_sha256=RESOLVER_SHA,
        actual_runtime=runtime.proof,
        python=platform.python_version(),
        zipfile_source_sha256=digest(zipfile.__file__),
        no_payload_decoding_or_extraction=True,
        TEST_payload_members_requested=False,
    )
    save_exclusive(out, result)
    print(
        json.dumps(
            {"output": str(out), "sha256": digest(out), "member_payload_bytes": budget.payload}
        )
    )


if __name__ == "__main__":
    main()
