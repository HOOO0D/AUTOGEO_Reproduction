"""End-to-end retrieval gate and GEO/GEU evaluation contracts for E7."""

from __future__ import annotations

from dataclasses import dataclass

from .config import E7Config
from .local_ranker import CounterfactualRankResult
from .protocol import PROTOCOL_SPECS, ProtocolId
from .ra_rewriter import RewriteResult


@dataclass(frozen=True)
class MethodSpecificGEContext:
    """The actual Top-5 after one method's counterfactual re-index."""

    protocol_id: ProtocolId
    document_ids: tuple[str, ...]
    target_source_index: int | None


@dataclass(frozen=True)
class E2EEvaluationResult:
    query_id: str
    target_document_id: str
    protocol_id: ProtocolId
    original_rank: int
    counterfactual_rank: int
    method_top5_document_ids: tuple[str, ...]
    target_source_index: int | None
    target_geo_score: dict[str, float] | None
    target_e2e_visibility: float
    geu_score: dict[str, float | None]
    response: str

    def validate(self, config: E7Config) -> None:
        spec = PROTOCOL_SPECS[self.protocol_id]
        if not spec.uses_e7_retrieval_pool:
            raise ValueError("E2EEvaluationResult is only for Track B")
        if len(self.method_top5_document_ids) != config.ge_top_k:
            raise ValueError("GE must receive the method-specific actual Top-5")
        entered = enters_ge_context(self.counterfactual_rank, config)
        expected_index = (
            self.method_top5_document_ids.index(self.target_document_id)
            if self.target_document_id in self.method_top5_document_ids
            else None
        )
        if entered != (expected_index is not None):
            raise ValueError("target rank and method-specific Top-5 disagree")
        if self.target_source_index != expected_index:
            raise ValueError("target_source_index must be the dynamic Top-5 index")
        if entered and expected_index != self.counterfactual_rank - 1:
            raise ValueError("dynamic source index must equal one-based rank minus one")
        if not entered:
            if self.target_e2e_visibility != 0.0:
                raise ValueError("target visibility must be zero outside Top-5")
            if self.target_geo_score is not None:
                raise ValueError("a non-source target has no within-context GEO score")
        elif self.target_geo_score is None:
            raise ValueError("a Top-5 target requires GEO scores at its dynamic index")
        if not self.response.strip():
            raise ValueError("GE response is required even when target misses Top-5")
        if not self.geu_score:
            raise ValueError("GEU is required even when target misses Top-5")


def enters_ge_context(rank: int, config: E7Config) -> bool:
    return 1 <= rank <= config.ge_top_k


def build_method_specific_ge_context(
    rewrite: RewriteResult,
    rank_result: CounterfactualRankResult,
    config: E7Config,
) -> MethodSpecificGEContext:
    """Build the actual Top-5; a retrieval miss does not skip GE or GEU."""
    if rewrite.target_document_id != rank_result.target_document_id:
        raise ValueError("rewrite and rank result must refer to the same D*")
    if rewrite.query_id != rank_result.query_id:
        raise ValueError("rewrite and rank result must refer to the same query")
    if not PROTOCOL_SPECS[rewrite.protocol_id].uses_e7_retrieval_pool:
        raise ValueError("method-specific re-index context is only for Track B")
    top_k_ids = rank_result.ranked_document_ids[: config.ge_top_k]
    if len(top_k_ids) != config.ge_top_k:
        raise ValueError("counterfactual ranking is shorter than ge_top_k")
    target_source_index = (
        top_k_ids.index(rewrite.target_document_id)
        if rewrite.target_document_id in top_k_ids
        else None
    )
    if enters_ge_context(rank_result.counterfactual_rank, config) != (
        target_source_index is not None
    ):
        raise ValueError("counterfactual rank and ranked document order disagree")
    if target_source_index is not None:
        if target_source_index != rank_result.counterfactual_rank - 1:
            raise ValueError("target position must equal one-based rank minus one")
    return MethodSpecificGEContext(
        protocol_id=rewrite.protocol_id,
        document_ids=top_k_ids,
        target_source_index=target_source_index,
    )


def evaluate_method_specific_top5(
    rewrite: RewriteResult,
    rank_result: CounterfactualRankResult,
    config: E7Config,
) -> E2EEvaluationResult:
    """Always generate and calculate GEU from the real method-specific Top-5."""
    del rewrite, rank_result, config
    raise NotImplementedError(
        "GE generation and GEO/GEU scoring are intentionally not run in this phase"
    )
