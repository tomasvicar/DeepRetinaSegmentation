# Optional PBS execution of the BF16 compute benchmark

The default benchmark still requires Ada. `--execution-target pbs` is an explicit
alternative for a scheduler-assigned native-BF16 GPU. This helper runs outside the
frozen training checkout; it does not change its source or create trained weights.

Before CUDA initialization, the PBS route requires a read-only host-source
attestation and its external original-byte SHA. Its bindings identify genuine
clean Git source 72fb321, execution/training hashes, dependency SIF, original host
path and container alias. A trusted host adapter must observe these facts; a
standalone JSON witness does not authenticate a host. The container independently
checks the mounted execution and training bytes. It also binds this helper's
bytes, the exact terminal E60 parent, PBS job ID, original scheduler CUDA mask,
assigned GPU UUID, and the reviewed Torch 8/1 thread profile. The actual single
visible GPU must have that UUID and support native BF16 without emulation.

A launcher preserves the original scheduler mask in
`RETINA_ORIGINAL_PBS_CUDA_MASK` before selecting the verified assigned UUID for
the container. Pass `--expected-gpu-uuid`, `--source-root /workspace`,
`--pbs-proof`, and `--pbs-proof-sha256` explicitly. The helper must be mounted
outside `/workspace`; `PYTHONPATH=/workspace/src` selects the frozen package.
The actual host source and genuine Git metadata stay read-only mounted.

The benchmark remains TRAIN-only, same-start ABBA FP32/BF16, with the original
terminal Adam state and learning rate. It measures numerical/compute behavior,
not validation quality or a newly trained model. Before a GPU allocation, run the
separate CPU container import, full canonical/cache staging and capacity proofs.
The metadata tests do not constitute an actual PBS/SIF or GPU performance proof.
