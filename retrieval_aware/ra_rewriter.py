"""Sibling rewrite contracts for the three Track B E2E protocols."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

from .config import E7Config
from .pool_builder import RetrievalPool
from .protocol import (
    AUTOGEO_FILTERED_RULES,
    PROTOCOL_SPECS,
    TRACK_B_PROTOCOLS,
    ProtocolId,
)
from .target_selector import TargetMetadata


@dataclass(frozen=True)
class RetrieverFeedback:
    round_index: int
    candidate_id: str
    target_rank: int
    target_score: float | None
    rank_delta_from_original: int


@dataclass(frozen=True)
class RewriteCandidate:
    """One RA candidate after local re-index supplies retriever feedback."""

    candidate_id: str
    round_index: int
    candidate_index: int
    rewritten_text: str
    target_rank: int
    target_score: float | None


@dataclass(frozen=True)
class RewriteRequest:
    protocol_id: ProtocolId
    query_id: str
    query: str
    target: TargetMetadata
    original_target_text: str
    pool: RetrievalPool
    ruleset_id: str | None
    feedback_history: tuple[RetrieverFeedback, ...] = ()


@dataclass(frozen=True)
class RewriteResult:
    protocol_id: ProtocolId
    query_id: str
    target_document_id: str
    pool_id: str
    original_text: str
    rewritten_text: str
    model_name: str | None
    prompt_version: str | None
    ruleset_id: str | None
    rules_sha256: str | None
    selected_round: int | None = None
    selected_candidate_id: str | None = None
    candidate_history: tuple[RewriteCandidate, ...] = ()


def validate_rewrite_request(request: RewriteRequest) -> None:
    """Enforce sibling inputs and protocol-specific feedback/rule boundaries."""
    spec = PROTOCOL_SPECS[request.protocol_id]
    if not spec.uses_e7_retrieval_pool:
        raise ValueError("Track A is evaluated by its unchanged original pipeline")
    if request.ruleset_id != spec.ruleset_id:
        raise ValueError(f"wrong ruleset for {request.protocol_id.value}")
    if request.target.document_id not in {
        document.document_id for document in request.pool.documents
    }:
        raise ValueError("the shared target must belong to the shared retrieval pool")
    pool_target = next(
        document
        for document in request.pool.documents
        if document.document_id == request.target.document_id
    )
    if request.original_target_text != pool_target.text:
        raise ValueError("every Track B sibling must start from original D*")
    if request.feedback_history and not spec.allows_retriever_feedback:
        raise ValueError("retrieval feedback is allowed only for ra_autogeo_e2e")
    if request.protocol_id is ProtocolId.RA_AUTOGEO_E2E:
        if request.ruleset_id != AUTOGEO_FILTERED_RULES:
            raise ValueError("RA-v1 must use the unchanged AutoGEO filtered rules")


def validate_rewrite_result(result: RewriteResult, config: E7Config) -> None:
    """Validate a completed Track B rewrite artifact before persistence."""
    spec = PROTOCOL_SPECS[result.protocol_id]
    if not spec.uses_e7_retrieval_pool:
        raise ValueError("Track A results must remain in the original pipeline")
    if not result.query_id or not result.target_document_id or not result.pool_id:
        raise ValueError("rewrite results require query, target, and pool identity")
    if result.ruleset_id != spec.ruleset_id:
        raise ValueError("rewrite result ruleset does not match its protocol")
    if spec.ruleset_id is None:
        if result.rules_sha256 is not None:
            raise ValueError("original_e2e must not claim a rewrite rules hash")
    else:
        if result.rules_sha256 is None or len(result.rules_sha256) != 64:
            raise ValueError("rule-based rewrites require a SHA-256 rules fingerprint")
        try:
            int(result.rules_sha256, 16)
        except ValueError as error:
            raise ValueError("rules fingerprint must be hexadecimal") from error
    if result.protocol_id is ProtocolId.ORIGINAL_E2E:
        if result.rewritten_text != result.original_text:
            raise ValueError("original_e2e cannot modify D*")
        if result.selected_round is not None or result.selected_candidate_id is not None:
            raise ValueError("original_e2e cannot select a rewrite candidate")
        if result.candidate_history:
            raise ValueError("original_e2e cannot contain rewrite candidates")
    elif result.protocol_id is ProtocolId.AUTOGEO_API_E2E:
        if result.selected_round is not None or result.selected_candidate_id is not None:
            raise ValueError("AutoGEO_API-E2E is one-shot, not iterative")
        if result.candidate_history:
            raise ValueError("AutoGEO_API-E2E cannot contain RA candidate history")
    else:
        if result.selected_round is None or result.selected_candidate_id is None:
            raise ValueError("RA-AutoGEO requires a selected round and candidate")
        if not 1 <= result.selected_round <= config.max_rounds:
            raise ValueError("RA selected round exceeds max_rounds")
        if not result.candidate_history:
            raise ValueError("RA-AutoGEO requires candidate history")
        if len(result.candidate_history) > (
            config.num_candidates_per_round * config.max_rounds
        ):
            raise ValueError("RA candidate history exceeds the configured budget")
        candidate_ids: set[str] = set()
        selected_candidate = None
        for candidate in result.candidate_history:
            if candidate.candidate_id in candidate_ids:
                raise ValueError("RA candidate identifiers must be unique")
            candidate_ids.add(candidate.candidate_id)
            if not 1 <= candidate.round_index <= config.max_rounds:
                raise ValueError("candidate round exceeds max_rounds")
            if not 1 <= candidate.candidate_index <= config.num_candidates_per_round:
                raise ValueError("candidate index exceeds num_candidates_per_round")
            if not 1 <= candidate.target_rank <= config.pool_size:
                raise ValueError("candidate target rank is outside the retrieval pool")
            if candidate.candidate_id == result.selected_candidate_id:
                selected_candidate = candidate
        if selected_candidate is None:
            raise ValueError("selected RA candidate is absent from candidate history")
        if selected_candidate.round_index != result.selected_round:
            raise ValueError("selected candidate and selected round disagree")
        if selected_candidate.rewritten_text != result.rewritten_text:
            raise ValueError("selected candidate text must be the persisted rewrite")


def validate_track_b_sibling_results(
    results: tuple[RewriteResult, ...],
    config: E7Config,
) -> None:
    """Require the three Track B arms to share query, pool, original D*, and rules."""
    if {result.protocol_id for result in results} != set(TRACK_B_PROTOCOLS):
        raise ValueError("a paired unit requires exactly the three Track B results")
    if len(results) != len(TRACK_B_PROTOCOLS):
        raise ValueError("paired Track B results must contain no duplicates")
    for result in results:
        validate_rewrite_result(result, config)
    shared_identity = {
        (result.query_id, result.pool_id, result.target_document_id, result.original_text)
        for result in results
    }
    if len(shared_identity) != 1:
        raise ValueError("Track B siblings must share query, pool, target, and original D*")
    by_protocol = {result.protocol_id: result for result in results}
    autogeo = by_protocol[ProtocolId.AUTOGEO_API_E2E]
    ra = by_protocol[ProtocolId.RA_AUTOGEO_E2E]
    if (autogeo.ruleset_id, autogeo.rules_sha256) != (
        ra.ruleset_id,
        ra.rules_sha256,
    ):
        raise ValueError("RA-v1 and AutoGEO-E2E must use identical filtered rules")


def original_e2e_variant(request: RewriteRequest) -> RewriteResult:
    """Represent the no-rewrite Track B arm without any model call."""
    validate_rewrite_request(request)
    if request.protocol_id is not ProtocolId.ORIGINAL_E2E:
        raise ValueError("original_e2e_variant requires protocol original_e2e")
    return RewriteResult(
        protocol_id=ProtocolId.ORIGINAL_E2E,
        query_id=request.query_id,
        target_document_id=request.target.document_id,
        pool_id=request.pool.pool_id,
        original_text=request.original_target_text,
        rewritten_text=request.original_target_text,
        model_name=None,
        prompt_version=None,
        ruleset_id=None,
        rules_sha256=None,
    )


def autogeo_baseline_variant(
    request: RewriteRequest,
    rewrite_document: Callable[..., str],
) -> RewriteResult:
    """Adapter contract for unchanged, one-shot AutoGEO without feedback.

    The callable is injected so importing the E7 skeleton never initializes an
    LLM client. The concrete adapter remains deferred.
    """
    validate_rewrite_request(request)
    if request.protocol_id is not ProtocolId.AUTOGEO_API_E2E:
        raise ValueError("AutoGEO E2E adapter requires protocol autogeo_api_e2e")
    del request, rewrite_document
    raise NotImplementedError("AutoGEO baseline adapter is not wired in this phase")


def retrieval_aware_variant(request: RewriteRequest) -> RewriteResult:
    """Produce an iterative retriever-aware rewrite in a later phase."""
    validate_rewrite_request(request)
    if request.protocol_id is not ProtocolId.RA_AUTOGEO_E2E:
        raise ValueError("RA adapter requires protocol ra_autogeo_e2e")
    del request
    raise NotImplementedError("RA-AutoGEO prompting is outside the skeleton phase")
