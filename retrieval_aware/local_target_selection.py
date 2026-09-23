"""DEV-only eligibility audit and target selection for pooled-local E7."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from enum import Enum
from pathlib import Path
from typing import Sequence

from .config import E7Config
from .frozen_local_retriever import (
    RetrievalHit,
    load_local_corpus,
)
from .retriever_registry import load_frozen_retriever
from .utils import read_json, write_e7_json_once


class LocalRankBucket(str, Enum):
    EASY = "easy"
    MEDIUM = "medium"
    HARD = "hard"


class LocalTargetSource(str, Enum):
    ORIGINAL_TARGET_ID = "original_target_id"
    ALTERNATE_ORIGINAL_CANDIDATE = "alternate_original_candidate"


@dataclass(frozen=True)
class OriginalCandidateRanking:
    document_id: str
    original_text_index: int
    is_original_autogeo_target: bool
    local_rank: int
    local_score: float
    eligible: bool
    rank_bucket: LocalRankBucket | None

    def to_dict(self) -> dict[str, object]:
        result = asdict(self)
        result["rank_bucket"] = (
            self.rank_bucket.value if self.rank_bucket is not None else None
        )
        return result


@dataclass(frozen=True)
class DevTargetSelection:
    query_id: str
    query: str
    target_document_id: str | None
    target_source: LocalTargetSource | None
    original_text_index: int | None
    original_autogeo_target_id: int
    initial_local_rank: int | None
    initial_local_score: float | None
    rank_bucket: LocalRankBucket | None
    eligible: bool
    selection_reason: str
    original_candidate_rankings: tuple[OriginalCandidateRanking, ...]

    def to_dict(self) -> dict[str, object]:
        return {
            "query_id": self.query_id,
            "query": self.query,
            "target_document_id": self.target_document_id,
            "target_source": (
                self.target_source.value if self.target_source is not None else None
            ),
            "original_text_index": self.original_text_index,
            "original_autogeo_target_id": self.original_autogeo_target_id,
            "initial_local_rank": self.initial_local_rank,
            "initial_local_score": self.initial_local_score,
            "rank_bucket": (
                self.rank_bucket.value if self.rank_bucket is not None else None
            ),
            "eligible": self.eligible,
            "selection_reason": self.selection_reason,
            "original_candidate_rankings": [
                candidate.to_dict()
                for candidate in self.original_candidate_rankings
            ],
        }


@dataclass(frozen=True)
class DevQueryCandidates:
    query_id: str
    query: str
    original_candidate_ids: tuple[str, ...]
    original_target_text_index: int
    original_target_document_id: str


def classify_rank_bucket(rank: int, config: E7Config) -> LocalRankBucket:
    if config.preferred_target_rank_min <= rank <= config.preferred_target_rank_max:
        return LocalRankBucket.EASY
    if config.preferred_target_rank_max < rank <= config.medium_target_rank_max:
        return LocalRankBucket.MEDIUM
    if config.medium_target_rank_max < rank <= config.max_target_rank:
        return LocalRankBucket.HARD
    raise ValueError(f"rank is outside E7 eligibility: {rank}")


def load_dev_query_candidates(manifest_path: Path) -> tuple[DevQueryCandidates, ...]:
    raw = read_json(manifest_path)
    if raw.get("schema_version") != "e7_researchy_pooled_corpus_v1":
        raise ValueError("unsupported Researchy-GEO corpus manifest")
    if raw.get("corpus_split") != "dev":
        raise ValueError("formal eligibility audit is restricted to DEV20")
    raw_queries = raw.get("query_candidate_sets")
    if not isinstance(raw_queries, list) or len(raw_queries) != 20:
        raise ValueError("DEV eligibility audit requires exactly 20 queries")

    queries: list[DevQueryCandidates] = []
    for raw_query in raw_queries:
        if not isinstance(raw_query, dict) or raw_query.get("source_split") != "dev":
            raise ValueError("DEV candidate row has invalid source split")
        query_id = raw_query.get("question_id")
        query = raw_query.get("query")
        candidate_ids = raw_query.get("original_candidate_ids")
        target_index = raw_query.get("original_target_text_index")
        target_document_id = raw_query.get("original_target_id_document")
        if not isinstance(query_id, str) or not isinstance(query, str):
            raise ValueError("DEV candidate row lacks query identity")
        if not isinstance(candidate_ids, list) or len(candidate_ids) != 5:
            raise ValueError("DEV query must retain five original candidates")
        if not all(isinstance(document_id, str) for document_id in candidate_ids):
            raise ValueError("original candidate identity must be a string")
        if not isinstance(target_index, int) or not 0 <= target_index < 5:
            raise ValueError("original AutoGEO target_id must be within 0--4")
        if target_document_id != candidate_ids[target_index]:
            raise ValueError("target_id document disagrees with original candidate index")
        queries.append(
            DevQueryCandidates(
                query_id=query_id,
                query=query,
                original_candidate_ids=tuple(candidate_ids),
                original_target_text_index=target_index,
                original_target_document_id=target_document_id,
            )
        )
    query_ids = [query.query_id for query in queries]
    if len(query_ids) != len(set(query_ids)):
        raise ValueError("DEV manifest contains duplicate query IDs")
    return tuple(queries)


def build_candidate_rankings(
    query: DevQueryCandidates,
    full_ranking: Sequence[RetrievalHit],
    config: E7Config,
) -> tuple[OriginalCandidateRanking, ...]:
    hits_by_id = {hit.document_id: hit for hit in full_ranking}
    if len(hits_by_id) != len(full_ranking):
        raise ValueError("full local ranking contains duplicate document IDs")
    rankings: list[OriginalCandidateRanking] = []
    for text_index, document_id in enumerate(query.original_candidate_ids):
        try:
            hit = hits_by_id[document_id]
        except KeyError as error:
            raise ValueError(
                f"original candidate is absent from local ranking: {document_id}"
            ) from error
        eligible = (
            config.preferred_target_rank_min
            <= hit.rank
            <= config.max_target_rank
        )
        rankings.append(
            OriginalCandidateRanking(
                document_id=document_id,
                original_text_index=text_index,
                is_original_autogeo_target=(
                    text_index == query.original_target_text_index
                ),
                local_rank=hit.rank,
                local_score=hit.score,
                eligible=eligible,
                rank_bucket=(
                    classify_rank_bucket(hit.rank, config) if eligible else None
                ),
            )
        )
    if len(rankings) != 5:
        raise AssertionError("candidate audit must preserve all five text slots")
    return tuple(rankings)


def select_dev_target(
    query: DevQueryCandidates,
    full_ranking: Sequence[RetrievalHit],
    config: E7Config | None = None,
) -> DevTargetSelection:
    config = config or E7Config()
    config.validate()
    candidates = build_candidate_rankings(query, full_ranking, config)
    original_target = candidates[query.original_target_text_index]
    if original_target.eligible:
        selected = original_target
        target_source = LocalTargetSource.ORIGINAL_TARGET_ID
        reason = "original_target_id_within_rank_6_100"
    else:
        alternates = [
            candidate
            for candidate in candidates
            if not candidate.is_original_autogeo_target and candidate.eligible
        ]
        if not alternates:
            return DevTargetSelection(
                query_id=query.query_id,
                query=query.query,
                target_document_id=None,
                target_source=None,
                original_text_index=None,
                original_autogeo_target_id=query.original_target_text_index,
                initial_local_rank=None,
                initial_local_score=None,
                rank_bucket=None,
                eligible=False,
                selection_reason="no_original_candidate_within_rank_6_100",
                original_candidate_rankings=candidates,
            )
        bucket_priority = {
            LocalRankBucket.EASY: 0,
            LocalRankBucket.MEDIUM: 1,
            LocalRankBucket.HARD: 2,
        }
        selected = sorted(
            alternates,
            key=lambda candidate: (
                bucket_priority[candidate.rank_bucket],
                candidate.local_rank,
                candidate.original_text_index,
                candidate.document_id,
            ),
        )[0]
        target_source = LocalTargetSource.ALTERNATE_ORIGINAL_CANDIDATE
        reason = f"highest_ranked_alternate_in_{selected.rank_bucket.value}_bucket"

    if selected.rank_bucket is None:
        raise AssertionError("selected target must have an eligibility bucket")
    return DevTargetSelection(
        query_id=query.query_id,
        query=query.query,
        target_document_id=selected.document_id,
        target_source=target_source,
        original_text_index=selected.original_text_index,
        original_autogeo_target_id=query.original_target_text_index,
        initial_local_rank=selected.local_rank,
        initial_local_score=selected.local_score,
        rank_bucket=selected.rank_bucket,
        eligible=True,
        selection_reason=reason,
        original_candidate_rankings=candidates,
    )


def summarize_selections(
    records: Sequence[DevTargetSelection],
) -> dict[str, object]:
    eligible = [record for record in records if record.eligible]
    return {
        "query_count": len(records),
        "eligible_count": len(eligible),
        "ineligible_count": len(records) - len(eligible),
        "original_target_id_eligible_count": sum(
            record.target_source is LocalTargetSource.ORIGINAL_TARGET_ID
            for record in eligible
        ),
        "alternate_candidate_count": sum(
            record.target_source
            is LocalTargetSource.ALTERNATE_ORIGINAL_CANDIDATE
            for record in eligible
        ),
        "rank_bucket_distribution": {
            bucket.value: sum(record.rank_bucket is bucket for record in eligible)
            for bucket in LocalRankBucket
        },
    }


def persist_dev_targets(
    payload: dict[str, object],
    config: E7Config,
) -> tuple[Path, str]:
    destination = config.output_paths["targets"] / "dev_targets.json"
    if destination.exists():
        if read_json(destination) != payload:
            raise FileExistsError(
                f"refusing to replace changed DEV target audit: {destination}"
            )
        return destination, "reused_identical"
    write_e7_json_once(destination, payload, config)
    destination.chmod(0o444)
    return destination, "created"


def run_dev_eligibility_audit(config: E7Config) -> dict[str, object]:
    manifest_path = config.output_paths["corpus"] / "dev_corpus_manifest.json"
    corpus = load_local_corpus(manifest_path)
    if corpus.corpus_split != "dev":
        raise ValueError("formal eligibility audit cannot run on TEST50")
    queries = load_dev_query_candidates(manifest_path)
    # This function preserves the already-frozen R0 audit. New consumers choose
    # a retriever through the registry rather than importing a model class.
    retriever = load_frozen_retriever(config.r0_retriever_id, corpus, config)
    cache = retriever.cache

    records: list[DevTargetSelection] = []
    for query in queries:
        full_ranking = retriever.retrieve(
            query.query,
            corpus,
            top_k=len(corpus.documents),
        )
        records.append(select_dev_target(query, full_ranking, config))
    summary = summarize_selections(records)
    payload: dict[str, object] = {
        "schema_version": "e7_dev_target_selection_v1",
        "protocol": "researchy_geo_pooled_local",
        "split": "dev",
        "corpus_id": corpus.corpus_id,
        "embedding_cache_id": cache.cache_id,
        "eligibility_rank_min": config.preferred_target_rank_min,
        "eligibility_rank_max": config.max_target_rank,
        "selection_policy": {
            "candidate_scope": "original_candidate_ids_only",
            "primary": "original_target_id_if_rank_6_100",
            "alternate_bucket_order": ["easy", "medium", "hard"],
            "within_bucket": "lowest_local_rank",
        },
        "summary": summary,
        "records": [record.to_dict() for record in records],
        "rewrite_calls_made": False,
        "test50_processed": False,
    }
    output_path, action = persist_dev_targets(payload, config)
    return {
        "output_path": str(output_path),
        "action": action,
        "summary": summary,
        "rewrite_calls_made": False,
        "test50_processed": False,
    }


def main() -> None:
    print(
        json.dumps(
            run_dev_eligibility_audit(E7Config()),
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
