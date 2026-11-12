"""Reproduce the source-scoped CPU suite; set a short TMPDIR before Python starts."""
import sys


def main():
    import pytest
    import torch

    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    return int(pytest.main(sys.argv[1:]))


if __name__ == "__main__":
    raise SystemExit(main())
