# Real raw source A/V CLI release check

One actual public `deepretina predict --model legacy-av --av-raw-matlab-source` call used raw STARE im0002, whose decodedRGB SHA and canonical manifest membership were verified as TRAIN before inference. Both original vessel prediction and source automatic FOV ran normally; no supplied vessel/FOV, MATLAB runtime, oracle values, GPU, GT scoring or TEST access. Source8835f76 and unchanged bundled weights were used.

The call returned0 in30.68seconds, kernel child peakRSS1,877,573,632bytes, under the180s/6GiB external watchdog. CPU numerical libraries used one thread and CUDA was hidden. Actual child niceness was19: the monitor inherited10 and its child added10, clamped by the kernel. This lower priority is recorded rather than misreported as10; the call was not repeated.

[Proof](proof.json) retains exact command, input/source/model/output hashes, profile and shape checks. Original605×700 outputs include artery, vein, classes, retained input vessel, overlay, probabilities and metadata. Finite conditional/gated probability planes, exclusive A/V masks and exact subset of saved NPZ vessel_support passed. Every exported path, including nested vessel provenance, exists and no deleted temporary output path is retained. Input vessel.png and AV restored support can differ at nearest-resampling boundaries; support comparisons use the actual NPZ vessel_support.

Artifacts remain outside the repository at the paths in the proof. Only metadata is committed, no source photograph, masks, probability arrays or weights. This is a release functionality check, not MATLAB numerical parity, an accuracy measurement, or clinical evidence. Existing measured source-photometry counterexamples remain explicit.
