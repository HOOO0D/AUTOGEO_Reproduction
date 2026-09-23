"""Local counterfactual re-indexing interfaces for E7."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, Sequence

from .pool_builder import RetrievalPool
from .target_selector import TargetMetadata


class EmbeddingModel(Protocol):
    def encode_query(self, query: str) -> Sequence[float]: ...

    def encode_document(self, text: str) -> Sequence[float]: ...


@dataclass(frozen=True)
class CounterfactualRankResult:
    query_id: str
    target_document_id: str
    original_rank: int
    counterfactual_rank: int
    target_score: float
    ranked_document_ids: tuple[str, ...]


def rerank_single_target(
    pool: RetrievalPool,
    target: TargetMetadata,
    rewritten_target_text: str,
    embedding_model: EmbeddingModel,
) -> CounterfactualRankResult:
    """Re-embed only the target, keep the other 99 documents fixed, then sort.

    The scoring rule is intentionally deferred until calibration demonstrates
    that it reproduces the fixed retriever's baseline ordering closely enough.
    """
    del pool, target, rewritten_target_text, embedding_model
    raise NotImplementedError("local scoring must be chosen by calibration")

