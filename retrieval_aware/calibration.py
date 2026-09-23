"""Calibration contracts for matching local and global retrieval rankings."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

from .pool_builder import RetrievalPool


@dataclass(frozen=True)
class CalibrationResult:
    query_id: str
    top_k: int
    top_k_overlap: float
    spearman_rho: float | None
    target_rank_error: int | None
    scoring_rule: str
    passed: bool
    notes: str = ""


def calibrate_local_ranker(
    pools: Sequence[RetrievalPool],
    scoring_rules: Sequence[str],
    top_k: int,
) -> list[CalibrationResult]:
    """Compare local baseline ranks with fixed/global retriever ranks later."""
    del pools, scoring_rules, top_k
    raise NotImplementedError(
        "calibration needs retrieved scores/embeddings and an agreed pass threshold"
    )

