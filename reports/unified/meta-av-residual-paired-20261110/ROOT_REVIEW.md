# Root scientific protocol review

The candidate configs match approved Ada6be48 exactly; architecture is the only
between-candidate change. Source3157, cache4e39, canonical6103 and60-epoch
scheduler remain fixed. Both fresh E8 trials run sequentially in one GPU
allocation under common4/1 thread policy and PYTHONHASHSEED42, with a45-minute
setup/control/QA budget gate before starting the residual trial. No checkpoint
migration or test tuning. Timing is order-confounded; quality is a single-seed
short screen, not final evidence.

The original draft remains byte-identical. Its inherited Ada review timestamp
and status are retained only under parent_ada_review in the separately hashed
root-reviewed copy. This fixes misleading approval metadata without rewriting
the published draft. Execution still requires actual successful CPU12 and an
independently reviewed private wrapper; scientific plan review alone submits
no job. Private resource/submission details stay in metacentrum-workflows.

Root-reviewed plan SHA256: `520facf9532722d3f6592a3e518ee3088ab297f31f96b44b64ba87ddb7aa5751`
