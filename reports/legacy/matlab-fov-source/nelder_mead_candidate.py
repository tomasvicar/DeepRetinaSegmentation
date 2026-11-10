"""Analysis-only access to private search state; public contract is first cutoff."""

from deepretina.legacy.matlab_firstcut import (
    _nelder_mead as nelder_mead,
    _search_thresholds as candidate_multithresh,
)

__all__ = ["nelder_mead", "candidate_multithresh"]
