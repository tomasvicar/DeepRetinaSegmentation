# Actual architecture meaning of the matched-config follow-up

Static inspection of frozen training commit72fb321: `model.UNet` builds
`depth+1` feature widths and pools between encoders; at depth4/base16 this means
16/32/64/128/256 with four pooling operations. `LegacyUniversalStudent` fixes
native32/64/96/128 and the original `single_net.Unet` has three pooling stages.
Thus both saved configs say depth4, but these constructor parameters count
levels differently. This is **not** a matched-downsampling or width-only ablation.

| Property | Compact control | Native scratch candidate |
| --- | --- | --- |
| Width schedule | 16/32/64/128/256 | 32/64/96/128 |
| Pooling operations | 4 | 3 |
| Upsampling | Bilinear interpolation and concatenation | Learned transposed convolutions and concatenation |
| First high-resolution block | Two convolutions | Three convolutions |
| Registered parameters | 1,965,637 | 1,304,421 |

The controlled factor is the **whole network family**. All36otherrecipe fields
remain fixed, and requested base_channels16→32 records the native fixedwidth
truthfully. In particular, the prospective protocol does not claim that keeping
the number4 fixes receptive field, effective feature resolution or compute.
A possible advantage for small structures is a hypothesis, not a measured gain;
no evaluation/pruning rule is changed by this clarification. The hash-pinned
original PROTOCOL.md and running configuration/source remain unchanged.

The native implementation retains two registered final BN parameters whose
forward branch is disabled. Its source-bound optimizer validation treats exactly
those two as inactive and keeps every other native Adam state strict; their
presence in parameter count is disclosed rather than silently removed.
