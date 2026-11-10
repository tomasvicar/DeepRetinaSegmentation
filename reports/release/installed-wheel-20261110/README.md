# Installed Python wheel verification

The release source `ad72647` built a37,454,699-byte wheel via `uv build --wheel`,
then installed it with `uv pip install --no-deps --target .../site` using the
existing pinned environment. The package was imported from that target while
running outside the source checkout. The wheel stayed local; this is an
installation check, not a sourceZIP handover.

All nine model resources match their catalog SHA256 in both the wheel and the
actual installed target. The real public source-AV CLI ran on the previously
verified canonical TRAIN STARE im0002, automatically computing both vessel
support and source FOV, CPU only, without MATLAB/oracle/GT scoring/TEST.

The process completed in30.38seconds, peak RSS1.81GB, actualnice10. Every saved
probability/support array and PNG pixel exactly matches the source-release CLI
result; all output metadata paths exist. The installed module origin and
`cuda_initialized=false` are recorded in `proof.json`. The bounded verifier is
retained, with actual local paths for this one-off evidence; to reproduce elsewhere
replace ROOT/REFERENCE and use the same verified raw TRAIN resource.

Raw images, masks, predictions and wheel remain outside Git. No final quality or
latency comparison follows from this functional packaging check.
