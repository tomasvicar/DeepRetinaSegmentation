# Work log

## 2026-11-10 — Leipzig cleanup

Requested: a clean `for_Leipzig` branch, Python only, `uv` installation, usable and documented
training and prediction for vessels, artery/vein classification, optic disc and cup. No dependency
on the missing large pretraining corpus. Keep investigation notes in this repository. Use
parallel agents; run training on Ada / Pascal / old where suitable.

- Cloned `https://github.com/tomasvicar/DeepRetinaSegmentation.git` into
  `/var/home/tom/code/DeepRetinaSegmentation_clean`, branched from master `06ae8f5`.
- All original remote branches are available. Original source is preserved in git history.
- Dataset root: `/var/home/tom/ownCloud/fundus_databases`. Existing `NOTES.md` provides helpful
  archive and annotation provenance; archive contents are being independently checked.
- Parallel tasks: source/branch audit; Python archive adapters; training/inference engine.
- Master requires a pretraining HDF5 even for supervised splitting; single_net contains an
  inverted missing-validation check, set-order-dependent splitting and evaluates the last
  checkpoint rather than the selected best checkpoint. Full details in REPOSITORY_MAP.md.
- Local machine is Pascal (GTX 1070 + GT 1030). Ada is reachable over SSH, RTX 4070 Ti SUPER
  16 GB and 64 GB RAM; a Windows VM allocates 24 GB RAM. old is reachable, GTX 1060 6 GB
  and 16 GB RAM, but does not have the dataset archives at the given path.
- Chose Python 3.12 and PyTorch 2.6 / CUDA 12.4 wheels via uv to cover both Pascal GPUs
  and Ada without downloading pretrained encoders. `uv.lock` records the environment.

## Evidence policy

A passing test or an executable training loop demonstrates functionality. A lower loss on
training crops does not demonstrate better generalization. Accuracy changes require an
identical held-out split, preprocessing and inference settings, with checkpoint selection using
validation only. Test data must never be used for hyperparameter or epoch selection. Small
pilot runs and untrained predictions will be explicitly identified as such.

- Initial full test suite: 12 tests passed, including synthetic end-to-end training for all tasks.
- Real adapters produced correct expected counts for DRIVE/RITE, CHASEDB1, IOSTAR and
  Drishti-GS. DRIVE/RITE share exactly matching source splits.
- Reviewed scale consistency: checkpoints must carry preparation max size; raw prediction
  uses that size before restoring probability maps to original image geometry.
- Removed obsolete master scripts and MATLAB files from this branch. Their exact source
  remains available at the original branch SHAs in REPOSITORY_MAP.md, avoiding a second
  undocumented installation or execution path.
- Correction at 07:54 CET: the initial `qemu:///session` check connected to a fresh
  libvirt instance and incorrectly suggested Ada's Windows VM was already off. The original
  GNOME Boxes QEMU process (PID 1772148) was running with about 23 GB resident memory.
  Connecting explicitly to its original Flatpak runtime socket confirmed `windows` running.
  The authorized graceful shutdown command below succeeded; follow-up reported `shut off`,
  no QEMU process, and 53 GiB available memory. No force kill or domain destroy was used.
  Ada training completed before this corrected shutdown check, so the VM must not be
  described as off before that training. old lacks raw archives; prepared data can be copied.

```bash
ssh ada 'flatpak run --filesystem=/run/user/1000/.flatpak/org.gnome.Boxes/xdg-run/libvirt --command=virsh org.gnome.Boxes -c "qemu+unix:///session?socket=/run/user/1000/.flatpak/org.gnome.Boxes/xdg-run/libvirt/virtqemud-sock" shutdown windows --mode agent,acpi'
```

The same Flatpak invocation with `list --all` instead of `shutdown windows --mode agent,acpi`
inspects the original instance. A default session connection alone was misleading here.

## Completed pilot and integration checks

- 18 tests pass, Ruff check/format pass, wheel/source builds pass. Explicit source
  archive allowlist excludes licensed images, labels, weights and local environments.
- All 8 adapters passed real-data checks across 1,615 prepared records. Main training
  manifest has 181 task records for DRIVE/RITE/Drishti, with shared DRIVE/RITE splits.
- Completed 6 controlled 30-epoch runs across old/Pascal/Ada. BCE+Dice improved vessel
  validation Dice by 0.011254 and disc/cup by 0.021284, but reduced AV mean Dice by 0.009028.
  Presets select BCE+Dice for vessels/disc_cup and BCE for AV; generic engine default BCE.
- Held-out tests were run only after validation selection. Vessel Dice 0.798755;
  AV model vessel Dice 0.807128, conditional AV accuracy 0.721889; disc Dice 0.879785,
  cup Dice 0.764842. Exact JSON metrics and all curves are in reports/2026-11-10.
- Inspected real predictions. Geometry restoration confirmed; disc/cup pilot has bright
  boundary false positives, documented without tuning against test examples.
- Selected weights backed up to a new ownCloud model folder, preserving all existing
  archives/models. Full commands, hashes, metric support and limitations in EXPERIMENTS.md.

## Storage correction

The user clarified that ownCloud has limited space. Removed only the newly added
`models/for_Leipzig_2026-11-10` backup (15 files, 6,005,545 bytes). The three checkpoint
copies in local `runs/` were checked byte-identical before deletion. No existing datasets
or historical models were removed. Future outputs stay in the local repository directory.
