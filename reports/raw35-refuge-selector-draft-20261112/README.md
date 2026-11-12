# Deferred REFUGE typed-selector draft

Work stopped at user direction on 2026-11-12. This isolated research draft is
incomplete and is not required for the selected raw512/originals handover.
Do not execute a resource audit or producer from this draft as a reviewed packet.

The shared stdlib selector reproduces the original REFUGE decoder's sorted-name,
folder-and-basename matching and `min(matches, key=len)` selection. Existing suffix
requests still delegate to the unchanged resolver. The producer records typed
selectors separately from unchanged frozen rows/photos/membership and uses them
before decoding. No category folder is inferred from an image path.

Completed exact focused check (synthetic inputs only):

```
CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONPATH=src \
  /var/home/tom/code/DeepRetinaSegmentation_hpo/.venv/bin/python -m pytest -q \
  tests/test_raw_resource_selectors.py tests/test_raw35_producer.py
```

Observed result: exit 0, **32 passed in 0.82s**. Ruff check and format also completed
successfully on the changed files. No real archive/index/member/pixel/GT/model
read, GPU operation, resource-audit retry, build or training was performed here.

Remaining deferred work: paired adoption of this exact shared module by the
standalone audit tool; new producer/source/helper identities and reviewed frozen
request-plan/profile regeneration; any actual resource readiness audit. The old
f9d producer, old 0bab plan and failed audit remain historical inputs, not evidence
that this draft has been activated or validated on the corpus.
