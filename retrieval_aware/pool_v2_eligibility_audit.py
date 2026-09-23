"""Eligibility-only audit for frozen R1 over the frozen DEV Pool V2 corpus."""

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
from .utils import (
    canonical_json_sha256,
    read_json,
    sha256_file,
    write_e7_json_once,
)


EXPECTED_POOL_V2_MANIFEST_SHA256 = (
    "423e4cd932222fca8e16ba47c230393f5c9a67a00b8c10f05a0292dbe27f55d8"
)
EXPECTED_POOL_V2_EMBEDDINGS_SHA256 = (
    "a2d53f643cf4a3e34e9082c7bb32aacf54dd9396951830500fa743469caacfef"
)
EXPECTED_POOL_V2_METADATA_SHA256 = (
    "3964d6fac25079eff963493273aad7245c91bcb0ee242bb3d43102111576fd63"
)
EXPECTED_POOL_V2_CORPUS_SIZE = 44_264

RANK_BUCKETS = ("top5", "easy", "medium", "hard", "out_of_pool")


@dataclass(frozen=True)
class PoolV2QueryCandidates:
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
class QueryPoolV2Audit:
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
            "eligible_physical_candidate_count": (
                self.eligible_physical_candidate_count
            ),
            "full_ranking_document_count": self.full_ranking_document_count,
            "full_ranking_deterministic": self.full_ranking_deterministic,
        }


def is_eligible_rank(rank: int, config: E7Config) -> bool:
    """Apply the frozen E7 v1 eligibility boundary exactly."""
    if rank < 1:
        raise ValueError("rank must be positive")
    return config.preferred_target_rank_min <= rank <= config.max_target_rank


def rank_bucket(rank: int, config: E7Config) -> str:
    """Map a positive full-corpus rank to the frozen descriptive bucket."""
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


def load_pool_v2_dev_queries(
    manifest_path: Path,
) -> tuple[PoolV2QueryCandidates, ...]:
    """Read DEV20 query provenance from Pool V2 without changing V1 loaders."""
    raw = read_json(manifest_path)
    if raw.get("schema_version") != "e7_researchy_pooled_corpus_v2":
        raise ValueError("Pool V2 eligibility requires the V2 corpus manifest")
    if raw.get("corpus_split") != "dev":
        raise ValueError("Pool V2 eligibility is restricted to DEV20")
    rows = raw.get("query_candidate_sets")
    if not isinstance(rows, list) or len(rows) != 20:
        raise ValueError("Pool V2 eligibility requires exactly DEV20 queries")

    queries: list[PoolV2QueryCandidates] = []
    for row in rows:
        if not isinstance(row, dict) or row.get("source_split") != "dev":
            raise ValueError("Pool V2 contains an invalid DEV query row")
        query_id = row.get("question_id")
        query = row.get("query")
        candidate_ids = row.get("original_candidate_ids")
        target_index = row.get("original_target_text_index")
        target_document_id = row.get("original_target_id_document")
        if not isinstance(query_id, str) or not isinstance(query, str):
            raise ValueError("Pool V2 DEV row lacks query identity")
        if not isinstance(candidate_ids, list) or len(candidate_ids) != 5:
            raise ValueError("every DEV query must preserve five source slots")
        if not all(isinstance(value, str) and value for value in candidate_ids):
            raise ValueError("original candidate identity must be a string")
        if not isinstance(target_index, int) or not 0 <= target_index < 5:
            raise ValueError("original AutoGEO target_id must be within 0--4")
        if target_document_id != candidate_ids[target_index]:
            raise ValueError("target_id provenance disagrees with its source slot")
        queries.append(
            PoolV2QueryCandidates(
                query_id=query_id,
                query=query,
                original_candidate_ids=tuple(candidate_ids),
                original_target_text_index=target_index,
                original_target_document_id=target_document_id,
            )
        )
    query_ids = [query.query_id for query in queries]
    if len(query_ids) != len(set(query_ids)):
        raise ValueError("Pool V2 DEV query IDs must be unique")
    return tuple(queries)


def audit_query(
    query: PoolV2QueryCandidates,
    full_ranking: Sequence[RetrievalHit],
    *,
    corpus_id: str,
    retriever_id: str,
    config: E7Config,
    deterministic_repeat: Sequence[RetrievalHit] | None = None,
) -> QueryPoolV2Audit:
    """Deduplicate slots by physical ID and audit candidates without selection."""
    if len(full_ranking) < config.pool_size:
        raise ValueError("full ranking is shorter than the frozen Top-100 pool")
    if any(hit.rank != index for index, hit in enumerate(full_ranking, start=1)):
        raise ValueError("full ranking has non-contiguous rank values")
    hits_by_id = {hit.document_id: hit for hit in full_ranking}
    if len(hits_by_id) != len(full_ranking):
        raise ValueError("full ranking contains duplicate physical identities")
    if deterministic_repeat is not None and tuple(full_ranking) != tuple(
        deterministic_repeat
    ):
        raise AssertionError("repeated full exact ranking is not deterministic")

    slots = tuple(
        {
            "source_slot": source_slot,
            "physical_document_id": document_id,
            "is_original_autogeo_target_id": (
                source_slot == query.original_target_text_index
            ),
        }
        for source_slot, document_id in enumerate(query.original_candidate_ids)
    )
    grouped_slots: dict[str, list[int]] = {}
    for source_slot, document_id in enumerate(query.original_candidate_ids):
        grouped_slots.setdefault(document_id, []).append(source_slot)

    physical_candidates: list[PhysicalCandidateAudit] = []
    for document_id, source_slots in grouped_slots.items():
        try:
            hit = hits_by_id[document_id]
        except KeyError as error:
            raise ValueError(
                f"original candidate is absent from Pool V2: {document_id}"
            ) from error
        contains_target = query.original_target_text_index in source_slots
        physical_candidates.append(
            PhysicalCandidateAudit(
                physical_document_id=document_id,
                source_slots=tuple(source_slots),
                contains_original_autogeo_target_id=contains_target,
                rank=hit.rank,
                score=hit.score,
                eligible=is_eligible_rank(hit.rank, config),
                rank_bucket=rank_bucket(hit.rank, config),
            )
        )
    if len(slots) != 5:
        raise AssertionError("slot provenance must retain all five source slots")
    target_candidates = [
        candidate
        for candidate in physical_candidates
        if candidate.contains_original_autogeo_target_id
    ]
    if len(target_candidates) != 1:
        raise AssertionError("exactly one physical candidate must contain target_id")
    if target_candidates[0].physical_document_id != query.original_target_document_id:
        raise AssertionError("physical target provenance changed during deduplication")

    eligible_count = sum(candidate.eligible for candidate in physical_candidates)
    return QueryPoolV2Audit(
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


def summarize_audits(records: Sequence[QueryPoolV2Audit]) -> dict[str, object]:
    """Aggregate query-scoped unique physical candidates and target provenance."""
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
        raise AssertionError("each query must contribute one physical target_id document")
    eligible_queries = sum(record.eligible for record in records)
    return {
        "dev_query_count": len(records),
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


def _assert_frozen_pool_v2_files(
    manifest_path: Path,
    embedding_path: Path,
    metadata_path: Path,
) -> None:
    expected = {
        manifest_path: EXPECTED_POOL_V2_MANIFEST_SHA256,
        embedding_path: EXPECTED_POOL_V2_EMBEDDINGS_SHA256,
        metadata_path: EXPECTED_POOL_V2_METADATA_SHA256,
    }
    for path, expected_sha256 in expected.items():
        actual_sha256 = sha256_file(path)
        if actual_sha256 != expected_sha256:
            raise ValueError(
                f"frozen Pool V2 artifact hash mismatch: {path} "
                f"({actual_sha256} != {expected_sha256})"
            )


def run_pool_v2_r1_dev_eligibility_audit(
    config: E7Config | None = None,
) -> dict[str, object]:
    """Run DEV20 exact retrieval and write one immutable eligibility artifact."""
    config = config or E7Config()
    config.validate()
    manifest_path = (
        config.output_paths["corpus"] / "pool_v2" / "dev_corpus_manifest.json"
    )
    cache_directory = (
        config.output_paths["embeddings"]
        / config.pool_v2_cache_namespace
        / "dev_corpus_cache"
    )
    embedding_path = cache_directory / "embeddings.npy"
    metadata_path = cache_directory / "metadata.json"
    _assert_frozen_pool_v2_files(manifest_path, embedding_path, metadata_path)

    corpus = load_local_corpus(manifest_path)
    if corpus.corpus_split != "dev" or len(corpus.documents) != EXPECTED_POOL_V2_CORPUS_SIZE:
        raise ValueError("frozen DEV Pool V2 corpus identity/size changed")
    queries = load_pool_v2_dev_queries(manifest_path)
    encoder = BGEBaseEnV15Encoder(config)
    cache = load_embedding_cache(
        corpus,
        encoder.specification,
        config,
        cache_namespace=config.pool_v2_cache_namespace,
        retriever_specification=encoder.retriever_specification,
    )
    retriever = FrozenLocalRetriever(encoder, cache)
    if retriever.retriever_id != config.r1_retriever_id:
        raise AssertionError("Pool V2 eligibility must use frozen R1")

    records: list[QueryPoolV2Audit] = []
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
    retriever_config = cache.retriever.to_dict()
    payload: dict[str, object] = {
        "schema_version": "e7_dev_r1_pool_v2_eligibility_audit_v1",
        "protocol": "researchy_geo_pooled_local",
        "split": "dev",
        "audit_scope": "eligibility_only_no_target_selection",
        "ranking_scope": "full_exact_corpus_ranking_before_top_100_truncation",
        "retriever_id": retriever.retriever_id,
        "retriever_config_sha256": canonical_json_sha256(retriever_config),
        "retriever_config": retriever_config,
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
        "pool_v1_vs_pool_v2_descriptive_comparison": {
            "comparison_type": "descriptive_only_no_protocol_change",
            "pool_v1": {
                "corpus_size": 597,
                "eligible_queries": 3,
                "dev_query_count": 20,
                "original_candidate_slots_top5": 97,
                "original_candidate_slots_total": 100,
                "eligible_candidate_rank_observation": "all_rank_6",
            },
            "pool_v2": {
                "corpus_size": len(corpus.documents),
                "eligible_queries": statistics["eligible_query_count"],
                "dev_query_count": statistics["dev_query_count"],
                "unique_physical_candidate_rank_distribution": statistics[
                    "unique_physical_candidate_rank_distribution"
                ],
            },
            "retriever_or_protocol_modified_after_comparison": False,
        },
        "per_query_candidate_ranks": [record.to_dict() for record in records],
        "selection_policy_executed": False,
        "formal_target_manifest_created": False,
        "rewrite_calls_made": False,
        "ge_geo_geu_calls_made": False,
        "llm_or_external_api_calls_made": False,
        "test50_processed": False,
    }
    destination = (
        config.output_paths["targets"] / "dev_eligibility_r1_pool_v2.json"
    )
    if destination.exists():
        if read_json(destination) != payload:
            raise FileExistsError(
                f"refusing to replace changed Pool V2 eligibility audit: {destination}"
            )
        action = "reused_identical"
    else:
        write_e7_json_once(destination, payload, config)
        destination.chmod(0o444)
        action = "created"

    _assert_frozen_pool_v2_files(manifest_path, embedding_path, metadata_path)
    return {
        "output_path": str(destination),
        "action": action,
        "eligibility_statistics": statistics,
        "pool_v1_vs_pool_v2_descriptive_comparison": payload[
            "pool_v1_vs_pool_v2_descriptive_comparison"
        ],
    }


def main() -> None:
    print(
        json.dumps(
            run_pool_v2_r1_dev_eligibility_audit(),
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
