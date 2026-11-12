# Retained RAW35 encoded-resource audit — disabled preparation

This standalone stdlib tool produces **only** the exact encoded byte identities
needed by producer f9d5e666. It never imports the producer, NumPy, PIL or Torch,
extracts archives, parses an image/mask, prepares35 data or changes a source.
The committed profile is False and incomplete. No actual resource audit, source
archive access, remote query or RAW35 build occurred in this task.

## Resolver and scope

`resolver.py` contains exact function-source copies of `canonical_path`,
`resolve_requests` and its `require` helper from frozen f9d5e666. Source/function
and copied-file SHA pins are in `resolver-source-pins.json`; the tool verifies
the copied file before use. The numerical/resource resolver is not reimplemented.
Exact case-insensitive names win, then canonical suffix matches. Multiple ZIP
suffix aliases select `min(len,name)` **only** when CRC/size metadata agrees;
conflicting identities fail before any member is opened. Canonical ZIP duplicate
names with different CRC/size fail while indexing. Equal CRC/size is the original
resolver's metadata rule, not proof that all duplicate payload bytes were hashed.
Only the selected retained payload is hashed. Directory aliases have no such CRC
proof, so ambiguous matches or canonical collisions fail rather than probing
additional files. Symlinks and paths escaping the explicit root fail.

Input is an externally SHA-pinned JSON plan with `requests` (archive → requested
resource → role list), exported from f9d5e666's **frozen TRAIN/VAL** metadata plan.
Its envelope must include exact producer commit, manifest17a, photos765,
membership08ef, TRAIN2529/5700/21sources, VAL770/993/15sources/612groups and exactly
22 archives /10999 resource requests. The producer summary calls the map
`resource_requests`; the reviewed metadata export explicitly renames that field
`requests` and adds the immutable input pins. It must preserve the full map,
not regenerate it by searching archives. `check_plan` enforces those headers;
the external reviewed plan SHA is what binds the exact retained requests.

TEST inventory names may appear in an archive's metadata index. No unselected
member stream is opened, decompressed or hashed. ZIP EOF/central-directory
container reads are bounded and counted separately. The tool does not infer
canonical splits from filenames: an official source folder named `Test` can
legitimately supply a frozen canonical VAL row. A self-declared header alone is
not proof that arbitrary requests came from the frozen manifest; the plan export
and its external review remain required.

## Finite IO and ownership

Before source access the CLI requires a real owned user **service** with actual
2GiB memory cap, swap0, CPU quota100%, external RuntimeMax1800s, KillMode control-group,
one reviewed CPU core, Nice17, idle IO and CUDA hidden. Actual systemd/cgroup facts
are checked/logged; unsupported or different limits fail. They are not relaxed.
Every container/member read checks the cooperative monotonic deadline/RSS/byte
ceiling, and each physical read and hash chunk is at most1MiB. The service is the
independent hard containment if a filesystem operation blocks. No timeout retry
or child-launch framework is included.

ZIP EOCD/ZIP64 metadata is checked before `ZipFile` allocates its index: at most
64MiB central directory,200000 entries and32MiB canonical names per archive.
The Python3.12.13 stdlib ZIP parser is used, including its EOCD/ZIP64 interpretation;
its actual module hash is recorded. Directories are indexed without following
links; retained files are opened via O_NOFOLLOW component traversal. Selected
member payload hashes use SHA256 after outer ZIP decompression, while the bytes
remain their original encoded JPEG/TIFF/JSON/etc form. There is no pixel decode.
Declared and actual member sizes and source before/after stat identities must
agree. The128GiB ceiling independently bounds retained payload bytes and physical
container reads (metadata included), rather than hiding compression bombs.

Output is an exclusive local JSON, at most32MiB, published only after completion,
mode0444. Existing final/partial paths reject before audit. Failure preserves a
partial owned output if one was being written; it never overwrites or resumes it.
The result's `resources` mapping is directly producer-compatible:

```json
{"resources":{"ARCHIVE.zip":{"request_to_member":{"suffix":"exact/member"},
"members":{"exact/member":{"size":123,"sha256":"64 lowercase hex"}}}}}
```

The audit also retains source stat checks, read counters, resolver/tool/profile/
plan pins and filtered runtime facts. It does not attest to pixel compatibility
or successful25 replay; those remain the producer's future independently bounded
work. Time/storage caps are ceilings, **not predictions** from the three-photo canary.

## Future activation command (not executed or approved here)

After Root reviews the exact plan, fills all null fields, and creates a separate
True profile without changing the committed False bytes:

```sh
systemd-run --user --wait --pipe --service-type=exec --unit=REVIEWED.service \
  -p MemoryMax=2G -p MemorySwapMax=0 -p CPUQuota=100% -p RuntimeMaxSec=1800 \
  -p KillMode=control-group -p TimeoutStopSec=20 -p CPUAffinity=REVIEWED_CORE \
  -p Nice=17 -p IOSchedulingClass=idle --setenv=CUDA_VISIBLE_DEVICES= \
  EXISTING_PYTHON tools/raw35_resource_audit/audit.py \
  --profile ACTUAL_TRUE.json --trusted-profile-sha256 ACTUAL_SHA \
  --plan REVIEWED_REQUESTS.json --source-root ORIGINAL_READONLY_RAW_ROOT \
  --output NEW_LOCAL_AUDIT.json
```

The operator must preserve the actual service/subprocess exit and output SHA.
This tool does not perform that activation, renew credentials, prepare data or
submit training. No installations are needed; existing Python stdlib suffices.

## Actual synthetic verification

Twenty-one small synthetic tests passed, final0.24s/exit0; Ruff check0. Fixtures
cover exact/suffix/CRC alias semantics, canonical conflict before payload, actual
small ZIP64 EOCD, unselected TEST sentinel never opened, directory filtering,
symlink/escape failures, central-directory ceilings,1MiB chunks, byte/time limits,
source stability, output no-overwrite/size failure, False refusal before source,
full metadata-scope validation and mocked correct/incorrect actual unit limits.
The two unit-property outcomes are fixtures, not an actual activated audit unit.
First15 tests passed0.19s with one remaining lint C405 issue; it was corrected.
An intermediate21-case run passed0.13s; the final source was rerun as above.

```sh
CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  taskset -c4 nice -n17 EXISTING_PYTHON -m pytest -q tools/raw35_resource_audit/test_audit.py
```
