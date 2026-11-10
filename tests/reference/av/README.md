# Immutable original A/V reference

`Network.py` is copied byte-for-byte from:

- Repository: https://github.com/tomasvicar/DeepRetinaSegmentation
- Commit: `7bb0539b90dfce2ed6eaeef1de812072be1155d0`
- Path: `Ves_Class/Network.py`
- SHA256: `8f99b450f38a3d01dbeb75a110f81ae13a39c0eeb1d4a8f785b38a33eaf8d7a9`

It is a reference fixture, not an imported runtime module. Tests extract the five
model classes by AST, avoiding its unrelated training/data imports. The independent
CenterCrop adapter in `test_legacy_av.py` implements the dependency's round-half-to-even
and padding contract and was additionally checked against actual torchvision 0.21.0.
Do not format or modernize this historical fixture; its byte identity is checked.
