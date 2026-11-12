# Full-V2 native topology from scratch: bounded follow-up

The existing release and its frozen856TEST remain unchanged. This follow-up is
motivated by prior TRAIN/VAL evidence: the native topology had only V1/E8, whereas
compact controls completed V2/E60. Historical-teacher initialization cannot erase
unknown split exposure, and a head copy does not preserve the complete original
prediction operator. This arm therefore uses no pretrained weights or parent.
The previous TEST has already been reported; it is not an unseen test for future
research. No TEST scoring or tuning is part of this activation.

## Controlled question

Does the original linear-width native topology improve the vessel/all-head tradeoff
when trained from scratch with the genuine firstTPE5 control recipe and fullV2?
Effective widths32/64/96/128, depth4, fiveoutputs,1,304,421parameters, compared with
compact16/depth4's1,965,637. RawJSON changes two fields: architecture and requested
base_channels16→32. The native builder fixes its width; the latter records the
actual first width and is not a second independently varied factor. All other36
config fields match the actual control, including workers8, intra8/inter1, raw25,
patch384/batch4, BF16/basic, optimizer/loss/sampling/seed42 and60epoch cosine horizon.
Only one seed is studied; paired photograph intervals do not measure training-seed
variance or prove complete patient independence.

Frozen training source72fb321187eca517823083aca80cedace9605570; executable digest
347f78ffd2f4a352c0870faad34327945f7d1c64b865dacb7b7d86e5ca4e0aaa;
training digest3725ad32a76449b2a3383485cbf14ef050a75f945b4c25e98f4b5b511a332f28.
ConfigfileSHAb699212778f703f2138fe76499b6d02a59971f7d986ef740751a5da576d8536c.
Canonical manifest17a2277c39cb37b7eee1b2063cfd13f2cf4adf4ebdf86cf449a1f51367c5a767;
content00d9ef19d262eef74f0ca1350fec260c687dff5cc3a53f8911cbe3403e821e52;
cache18c3af6ce64e23c896d7befc52e5ba8e8ebda8e4058c2f156574fa755480766e.
There are2529TRAINphotos/5700annotationrecords/21sources and770VALphotos.
Existing encoded data/cache are read-only; no synthetic35ppd resize is introduced.

## First activation and prospective continuation policy

E1only is an operational gate, with stop_after_epoch1 but epochs60/schedulerTmax60.
Its actual stop must deliver8192samples/2048optimizer updates, finite used Adam
moments/RNG and resumable state. Report actual photo/annotation contribution counts;
E1 does not demonstrate complete60epoch training or all5700annotations by assumption.
The original model registers exactly two unused final BN parameters; no optimizer
states are fabricated for them. A distinct model-bound completion verifier and
QA pin are required; see the optimizer binding review report.

Resource gate: one own-Ada E1, maximum1800s including setup/postcheck/cleanup,
actual user-systemd8GiB memory/no swap/8CPU quota/Nice17/idleIO, short HOME worker
TMPDIR. The explicit desktop40 policy requires only exact reviewed actors, sufficient
freeVRAM/RAM and records desktop activity rather than claiming idle. Original
strict20 deferrals remain preserved. This policy is not an authorization for more
machines, Meta jobs, automatic retries, worker reduction or continuation.

Before a continuation, use observed setup/TRAIN/VAL times to forecast
actualE1elapsed +59×1.2×(E1TRAIN+E1VAL) +1800scompletionQA +300scleanup.
Maximum projected full arm is12hours; if it does not fit, defer rather than change
the scheduler, validation cadence, workers or batch. A new bounded resume profile
must be reviewed before activation. If admitted, the intended scientific comparison
is full60epochs: **no quality-based pruning from E1 or E27**. E27 is a retained
resumable diagnostic, not evidence of terminal inferiority; continuation60 still
requires actual source/state/resource/budget checks. NaN/OOM/infrastructure failures
remain incomplete operations, not bad quality scores. No automatic continuation
is enabled by this document.

## Evaluation after genuine completion

Primary controlled contrast: fixed LAST60 versus the genuine compact firstTPE5
LAST60, same770VAL/sources/geometry, FP32 NONE/.5/tile512/overlap64. BEST is separate
and must disclose its own epoch/coverage. Dense vessel87photos/55known groups/8sources
(exclude only positive-only AVRDB) and each source's precision/recall/counts are
mandatory alongside every head, conditional/end-to-endAV, macro and paired group
intervals. Also retain the already selected raw512 and original references as
external anchors, without calling their different recipes a one-factor contrast.
Future formal validation must use a newly reviewed evaluation source lease for the
legacy optimizer revision; no historical6d/f13 report is relabelled. Do not select
new checkpoints, thresholds or protocols from the previously observed TEST.

At this commit, the E1 operational run has started; no completed epoch, quality
improvement, full60 eligibility or new model release is claimed. Detailed actual
unit/identity/exit evidence stays in the private operations repository.
