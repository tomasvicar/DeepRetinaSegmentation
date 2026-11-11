# Python-only independent corpus copy

Followup to RAW-only usability commit `a9598e6`. The V2 append producer no longer
runs an external copy command. A Python tree copier creates destination files
exclusively, attempts Linux FICLONE, and falls back to `shutil.copy2` for unsupported
filesystems or other platforms. Actual receipt counters distinguish reflink,
ordinary-copy and clone-internal hardlink files; no universal CoW claim is made.

Internal hardlink aliases are recreated from another **destination** file only.
No `os.link` call uses a source-parent path. Unsupported internal hardlinks fall
back to an independent copy. Zero/unavailable inode identities are not merged.
Mode/timestamps use copy2/copystat semantics; ownership and complete ACL parity
are not promised. Destination directories gain owner access, enabling append
when the source is read-only; source permissions remain unchanged.

Before fingerprinting/copying, the tree is checked for symlinks and special files.
Existing/nested destinations and symlink roots are rejected. Data copy failures,
including ENOSPC, propagate without an automatic retry or overwrite. Existing
content fingerprint, per-file byte hashes, partition, and inode-independence
checks remain in the enclosing producer. Failed outputs remain for inspection.

CPU-only focused verification: **36 tests passed**, including the real synthetic
DRIVE/RITE/Drishti→DRIONS append/audit pipeline. New tests cover read-only source
files/directories, bytes/timestamps/modes, internal aliases, simulated unavailable
reflink, simulated non-Linux ordinary copies, unsupported hardlinks, existing
output protection, symlink/special-file rejection, and propagation of disk-full
errors. No complete corpus was rebuilt and no training/GPU/shared RAW was touched.
The non-Linux branch is simulated on Linux; this is not a full Windows/macOS
platform certification. Current skimage deprecation warnings are unchanged.
