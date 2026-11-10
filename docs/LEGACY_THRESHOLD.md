# Strict original default threshold

At the default threshold 0.5, public single_net masks now use the original strict
logit>0 decision **before nearest restoration**, matching the original validation
reference and strong-pipeline benchmark. In float32, sigmoid(+1e-8) rounds to 0.5;
thresholding that rounded probability would incorrectly remove a positive logit.
Zero and negative logits remain background. Metadata records the mask decision.

Explicit nondefault thresholds retain the existing probability>threshold contract.
Probability outputs remain sigmoid followed by continuous restoration, with zero
outside the crop. Independent heads and original cup policy do not change.
The raw A/V convenience route obtains automatic vessel support through this same
public single_net path, so its default vessel decisions receive the fix as well.

Regression tests use actual float32 ±1e-8/0 and float64 ±1e-16/0 and odd original/crop geometry, preserve
probability export and arbitrary thresholds 0.3/0.7, and run the public raw-A/V CLI
with automatic tiny-logit vessels plus actual bundled A/V classification. No
model weights, preprocessing profile, training math, or running immutable source is changed.

Current original tiling blends into float64, whose analogous rounding edge is
+1e-16. The float32 case additionally protects that accepted helper output dtype.
Actual occurrence/frequency in real model logits has not been measured; this
is a numerical decision-contract fix, not a claimed accuracy improvement.

Validation:36 targeted integration/CLI cases passed on Ada CPU, nice10, two
threads, CUDA hidden; four existing morphology warnings. Local heavy tests were
not run during the desktop usability incident.
