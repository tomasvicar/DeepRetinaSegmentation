# Actual full-image photometry port parity

On 11 November 2026, both CPU comparisons on one actual DRIVE TRAIN photograph
passed with bitwise identical finite FP32 probabilities and decoded masks
(maximum absolute error zero):

| Original checkpoint | Reference implementation | Compared output |
| --- | --- | --- |
| TPE BEST58, SHA20d4caab… | Previous product48dd081 | Actual raw prediction, restored geometry and metadata |
| Raw256 BEST52, SHA132c7879… | Original fd44cdf | Prepared and restored probabilities, masks, full saved config |

Candidate source1fb4a0c was merged unchanged into producta64c15c. The numerical
summary preserves exact source/model/output hashes and resource limits. Only the
RGB input and official image-derived FOV were used: no annotation, VAL/TEST image
or GPU. No claim of nonraw trained-model parity, new model quality, whole-corpus
parity or latency improvement follows from these two checks. Raw256 is not a
bundled or promoted model.

The successful comparison children exited0, taking27.314s and24.919s including
imports, with sampled resident memory below0.8GiB. Earlier bookkeeping failures
are explicitly retained in the summary: an initial output-metadata lookup failed
after inference; a later raw256 input preflight incorrectly assumed unframed pixel
hashes and stopped before inference. The successful TPE child belongs to that
combined attempt with outer exit1; successful raw256 ran separately with outer
exit0. Corrected checks preserved original inputs and source bytes. The private
operations repository retains complete original attempts and resource observers;
this report exports numerical evidence without operator host paths.
