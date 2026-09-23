"""Eligibility-only audit for frozen R1 over immutable TEST50 Pool V2.

The module never selects a target.  It exact-ranks the complete local corpus,
deduplicates the five original slots by physical identity, and applies the
pre-registered rank 6--100 eligibility boundary.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Sequence

from .config import E7Config
from .frozen_local_retriever import (
    FrozenLocalRetriever,
    RetrievalHit,
    load_embedding_cache,
    load_local_corpus,
)
from .independent_retriever import BGEBaseEnV15Encoder
from .test50_pool_v2 import (
    EXPECTED_TEST_QUERY_COUNT,
    _persist_immutable_json,
    test_pool_v2_paths,
)
from .test50_pool_v2_embeddings import (
    FROZEN_R1_CONFIG_SHA256,
    TEST_POOL_V2_CACHE_NAMESPACE,
    assert_frozen_r1_contract,
)
from .utils import canonical_json_sha256, read_json, sha256_file


RANK_BUCKETS = ("top5", "easy", "medium", "hard", "out_of_pool")


@dataclass(frozen=True)
class TestQueryCandidates:
    query_id: str
    query: str
    original_candidate_ids: tuple[str, ...]
    original_target_text_index: int
    original_target_document_id: str


@dataclass(frozen=True)
class PhysicalCandidateAudit:
    physical_document_id: str
    source_slots: tuple[int, ...]
    contains_original_autogeo_target_id: bool
    rank: int
    score: float
    eligible: bool
    rank_bucket: str

    def to_dict(self) -> dict[str, object]:
        value = asdict(self)
        value["source_slots"] = list(self.source_slots)
        return value


@dataclass(frozen=True)
class TestQueryAudit:
    query_id: str
    query: str
    corpus_id: str
    corpus_size: int
    retriever_id: str
    top_10: tuple[RetrievalHit, ...]
    top_100: tuple[RetrievalHit, ...]
    original_candidate_slots: tuple[dict[str, object], ...]
    unique_original_candidates: tuple[PhysicalCandidateAudit, ...]
    eligible: bool
    eligible_physical_candidate_count: int
    full_ranking_document_count: int
    full_ranking_deterministic: bool

    def to_dict(self) -> dict[str, object]:
        return {
            "query_id": self.query_id,
            "query": self.query,
            "corpus_id": self.corpus_id,
            "corpus_size": self.corpus_size,
            "retriever_id": self.retriever_id,
            "top_10": [hit.to_dict() for hit in self.top_10],
            "top_100": [hit.to_dict() for hit in self.top_100],
            "original_candidate_slots": list(self.original_candidate_slots),
            "unique_original_candidates": [
                candidate.to_dict() for candidate in self.unique_original_candidates
            ],
            "eligible": self.eligible,
            "eligible_physical_candidate_count": self.eligible_physical_candidate_count,
            "full_ranking_document_count": self.full_ranking_document_count,
            "full_ranking_deterministic": self.full_ranking_deterministic,
        }


def is_eligible_rank(rank: int, config: E7Config) -> bool:
    if rank < 1:
        raise ValueError("rank must be positive")
    return config.preferred_target_rank_min <= rank <= config.max_target_rank


def rank_bucket(rank: int, config: E7Config) -> str:
    if rank < 1:
        raise ValueError("rank must be positive")
    if rank <= config.ge_top_k:
        return "top5"
    if rank <= config.preferred_target_rank_max:
        return "easy"
    if rank <= config.medium_target_rank_max:
        return "medium"
    if rank <= config.max_target_rank:
        return "hard"
    return "out_of_pool"


def load_test_queries(
    manifest_path: Path,
    *,
    expected_query_count: int = EXPECTED_TEST_QUERY_COUNT,
) -> tuple[TestQueryCandidates, ...]:
    raw = read_json(manifest_path)
    if raw.get("schema_version") != "e7_researchy_pooled_corpus_v2":
        raise ValueError("TEST eligibility requires Pool V2 schema")
    if raw.get("corpus_split") != "test":
        raise ValueError("TEST eligibility cannot consume a DEV corpus")
    rows = raw.get("query_candidate_sets")
    if not isinstance(rows, list) or len(rows) != expected_query_count:
        raise ValueError("TEST eligibility query count changed")
    queries: list[TestQueryCandidates] = []
    for row in rows:
        if not isinstance(row, dict) or row.get("source_split") != "test":
            raise ValueError("invalid TEST query provenance row")
        query_id = row.get("question_id")
        query = row.get("query")
        candidate_ids = row.get("original_candidate_ids")
        target_index = row.get("original_target_text_index")
        target_document_id = row.get("original_target_id_document")
        if not isinstance(query_id, str) or not isinstance(query, str):
            raise ValueError("TEST row lacks query identity")
        if not isinstance(candidate_ids, list) or len(candidate_ids) != 5:
            raise ValueError("TEST row must preserve five original slots")
        if not all(isinstance(value, str) and value for value in candidate_ids):
            raise ValueError("TEST original candidate identity is invalid")
        if not isinstance(target_index, int) or not 0 <= target_index < 5:
            raise ValueError("TEST original target_id is invalid")
        if target_document_id != candidate_ids[target_index]:
            raise ValueError("TEST original target provenance changed")
        queries.append(
            TestQueryCandidates(
                query_id=query_id,
                query=query,
                original_candidate_ids=tuple(candidate_ids),
                original_target_text_index=target_index,
                original_target_document_id=target_document_id,
            )
        )
    if len({query.query_id for query in queries}) != len(queries):
        raise ValueError("TEST query IDs are not unique")
    return tuple(queries)


def audit_query(
    query: TestQueryCandidates,
    full_ranking: Sequence[RetrievalHit],
    *,
    corpus_id: str,
    retriever_id: str,
    config: E7Config,
    deterministic_repeat: Sequence[RetrievalHit] | None = None,
) -> TestQueryAudit:
    if len(full_ranking) < config.pool_size:
        raise ValueError("full ranking is shorter than frozen Top-100")
    if any(hit.rank != index for index, hit in enumerate(full_ranking, start=1)):
        raise ValueError("full ranking ranks are not contiguous")
    hits_by_id = {hit.document_id: hit for hit in full_ranking}
    if len(hits_by_id) != len(full_ranking):
        raise ValueError("full ranking contains duplicate physical identities")
    if deterministic_repeat is not None and tuple(full_ranking) != tuple(deterministic_repeat):
        raise AssertionError("repeated full exact ranking is not deterministic")

    slots = tuple(
        {
            "source_slot": slot,
            "physical_document_id": document_id,
            "is_original_autogeo_target_id": slot == query.original_target_text_index,
        }
        for slot, document_id in enumerate(query.original_candidate_ids)
    )
    grouped: dict[str, list[int]] = {}
    for slot, document_id in enumerate(query.original_candidate_ids):
        grouped.setdefault(document_id, []).append(slot)
    physical_candidates: list[PhysicalCandidateAudit] = []
    for document_id, source_slots in grouped.items():
        hit = hits_by_id.get(document_id)
        if hit is None:
            raise ValueError(f"original candidate is absent from full ranking: {document_id}")
        physical_candidates.append(
            PhysicalCandidateAudit(
                physical_document_id=document_id,
                source_slots=tuple(source_slots),
                contains_original_autogeo_target_id=(
                    query.original_target_text_index in source_slots
                ),
                rank=hit.rank,
                score=hit.score,
                eligible=is_eligible_rank(hit.rank, config),
                rank_bucket=rank_bucket(hit.rank, config),
            )
        )
    target_candidates = [
        candidate
        for candidate in physical_candidates
        if candidate.contains_original_autogeo_target_id
    ]
    if len(target_candidates) != 1:
        raise AssertionError("exactly one physical document must contain target_id")
    if target_candidates[0].physical_document_id != query.original_target_document_id:
        raise AssertionError("original target provenance changed during physical dedup")
    eligible_count = sum(candidate.eligible for candidate in physical_candidates)
    return TestQueryAudit(
        query_id=query.query_id,
        query=query.query,
        corpus_id=corpus_id,
        corpus_size=len(full_ranking),
        retriever_id=retriever_id,
        top_10=tuple(full_ranking[:10]),
        top_100=tuple(full_ranking[: config.pool_size]),
        original_candidate_slots=slots,
        unique_original_candidates=tuple(physical_candidates),
        eligible=eligible_count > 0,
        eligible_physical_candidate_count=eligible_count,
        full_ranking_document_count=len(full_ranking),
        full_ranking_deterministic=(
            deterministic_repeat is None
            or tuple(full_ranking) == tuple(deterministic_repeat)
        ),
    )


def summarize_audits(records: Sequence[TestQueryAudit]) -> dict[str, object]:
    candidates = [
        candidate
        for record in records
        for candidate in record.unique_original_candidates
    ]
    original_targets = [
        candidate
        for candidate in candidates
        if candidate.contains_original_autogeo_target_id
    ]
    if len(original_targets) != len(records):
        raise AssertionError("every query must retain one target_id physical document")
    eligible_queries = sum(record.eligible for record in records)
    return {
        "test_query_count": len(records),
        "original_candidate_slots_total": sum(
            len(record.original_candidate_slots) for record in records
        ),
        "unique_original_physical_candidates_total": len(candidates),
        "unique_physical_candidate_rank_distribution": {
            bucket: sum(candidate.rank_bucket == bucket for candidate in candidates)
            for bucket in RANK_BUCKETS
        },
        "eligible_query_count": eligible_queries,
        "ineligible_query_count": len(records) - eligible_queries,
        "eligible_physical_candidate_count_by_query": {
            record.query_id: record.eligible_physical_candidate_count
            for record in records
        },
        "original_autogeo_target_rank_distribution": {
            bucket: sum(candidate.rank_bucket == bucket for candidate in original_targets)
            for bucket in RANK_BUCKETS
        },
    }


def run_test50_eligibility_audit(config: E7Config | None = None) -> dict[str, object]:
    config = config or E7Config()
    config_hash = assert_frozen_r1_contract(config)
    manifest_path = test_pool_v2_paths(config)["manifest"]
    cache_directory = (
        config.output_paths["embeddings"]
        / TEST_POOL_V2_CACHE_NAMESPACE
        / "test_corpus_cache"
    )
    embedding_path = cache_directory / "embeddings.npy"
    metadata_path = cache_directory / "metadata.json"
    frozen_before = {
        path: sha256_file(path)
        for path in (manifest_path, embedding_path, metadata_path)
    }
    corpus = load_local_corpus(manifest_path)
    queries = load_test_queries(manifest_path)
    encoder = BGEBaseEnV15Encoder(config, execution_device="cpu")
    if canonical_json_sha256(encoder.retriever_specification.to_dict()) != config_hash:
        raise AssertionError("loaded query encoder differs from frozen R1")
    cache = load_embedding_cache(
        corpus,
        encoder.specification,
        config,
        cache_namespace=TEST_POOL_V2_CACHE_NAMESPACE,
        retriever_specification=encoder.retriever_specification,
    )
    if canonical_json_sha256(cache.retriever.to_dict()) != FROZEN_R1_CONFIG_SHA256:
        raise ValueError("TEST embedding cache uses a different R1 contract")
    retriever = FrozenLocalRetriever(encoder, cache)
    records: list[TestQueryAudit] = []
    for query in queries:
        first = retriever.retrieve(query.query, corpus, top_k=len(corpus.documents))
        second = retriever.retrieve(query.query, corpus, top_k=len(corpus.documents))
        records.append(
            audit_query(
                query,
                first,
                corpus_id=corpus.corpus_id,
                retriever_id=retriever.retriever_id,
                config=config,
                deterministic_repeat=second,
            )
        )
    statistics = summarize_audits(records)
    payload: dict[str, object] = {
        "schema_version": "e7_test50_r1_pool_v2_eligibility_audit_v1",
        "protocol": "researchy_geo_pooled_local",
        "split": "test",
        "audit_scope": "eligibility_only_no_target_selection",
        "ranking_scope": "full_exact_corpus_ranking_before_top_100_truncation",
        "retriever_id": retriever.retriever_id,
        "retriever_config_sha256": config_hash,
        "retriever_config": cache.retriever.to_dict(),
        "corpus_id": corpus.corpus_id,
        "corpus_size": len(corpus.documents),
        "corpus_manifest_path": str(manifest_path.resolve()),
        "corpus_manifest_sha256": corpus.manifest_sha256,
        "embedding_cache_id": cache.cache_id,
        "embedding_cache_path": str(cache.cache_directory.resolve()),
        "embedding_cache_sha256": cache.embeddings_sha256,
        "embedding_cache_metadata_sha256": sha256_file(metadata_path),
        "eligibility_definition": {
            "candidate_scope": "unique_physical_documents_from_query_original_five",
            "minimum_rank": config.preferred_target_rank_min,
            "maximum_rank": config.max_target_rank,
            "ge_top_k": config.ge_top_k,
            "duplicate_slots_counted_once_for_retrieval_rank": True,
            "slot_provenance_preserved": True,
        },
        "eligibility_statistics": statistics,
        "per_query_candidate_ranks": [record.to_dict() for record in records],
        "selection_policy_executed": False,
        "formal_target_manifest_created": False,
        "rewrite_calls_made": False,
        "ge_geo_geu_calls_made": False,
        "llm_or_external_api_calls_made": False,
        "test50_processed": True,
    }
    destination = config.output_paths["targets"] / "test50_eligibility_r1_pool_v2.json"
    action, digest = _persist_immutable_json(destination, payload, config)
    frozen_after = {path: sha256_file(path) for path in frozen_before}
    if frozen_after != frozen_before:
        raise AssertionError("eligibility audit modified TEST corpus/cache artifacts")
    return {
        "output_path": str(destination),
        "action": action,
        "output_sha256": digest,
        "eligibility_statistics": statistics,
    }


def main() -> None:
    print(json.dumps(run_test50_eligibility_audit(), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
