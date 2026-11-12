# Native legacy optimizer binding: synthetic readiness only

The strictCPU loader reconstructs the actual LegacyUniversalStudent. Only when
its up_conv1.1.do_batch is0 are the two registered affine parameters
up_conv1.1.bn.weight/bias legitimately missing AdamW state. Native ordered,
unique integer group IDs still cover EVERY registered parameter; state IDs must
be EXACTLY that inventory minus the two proven unused IDs. Fabricated unused
states, missing used state, wrong order/shape/dtype/nonfinite/stale state reject.
No arbitrary skiplist or exception for compact models is introduced.

Future source72 legacy standalone QA must use the separately reviewed new
verifier SHA412a11bb92c438446f70b5b6aa471e5aa38f2bb35c802723b823f1a89a638d47
and explicit optimizer_binding_schema `legacy_universal_native_order_unused_bn_v1`.
Compact0b/72 retains original4b verifier pin. Crossed/unknown pins/tags/source
families reject; original terminal/protocol/config/data/owncoverage rules remain.
Historical QA is not retagged. No original source72, frozen6d or pending E1
verifier was edited; no live artifact was read or changed.

Actual focused CPU metadata result: **43PASS in5.89s**, actualchild0; see
[scope and provenance](verification.json). Tests use strict synthetic CPU model
serialization and genuine AdamW states generated from assigned gradients, with
no forward, realdata, GT, GPU or TEST. Both public/private bindings see the same
corruption cases. Enabled BN flag, actual compact full-state requirement and
architecture-conditioned verifier pin/schema are also checked.

Initial28pass+1tmp_path setup failure was caused by /tmpquota. The original
failure and external receipts are retained privately; owned shortHOME TMPDIR
resolved the harness. No gate was loosened to make a fixture pass.

This experimental gate source has a new evaluation fingerprint. An actual future
legacy candidate needs original NEW-verifier completion, reviewed freshVAL under
the appropriate new source/runtime, then ordinary selection review. No current
frozen release, historical validation, family or heldout entry is silently opened.
This report proves synthetic binding behavior, not an actuallegacy60 result.
