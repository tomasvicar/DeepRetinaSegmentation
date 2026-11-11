# Root independent verification

Source reviewed:1775a2a; production diff only the validated sampling scalar.
Root repeated all41 new tests:41 passed in1.88seconds, CPU1/nice10/threads1,
CUDA hidden, PYTHONHASHSEED42. It also independently reran the actual5700-record /
2529-photo task-flag audit using exact manifest17a2 and flags62a4, with no further
RGB/label decoding or model execution. Default float64 weight bits, all9 sampler
sequences and final RNG states match frozen6b; priority2 matches the independent
key oracle, total mass35→50. The full132 tests were previously run by the author.

This proves the implementation/default compatibility and intended mixture change.
It does not prove better model quality. No GPU job, training, validation, TEST,
current protocol mutation or stable merge was performed.
