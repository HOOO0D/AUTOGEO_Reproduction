"""DEV20 eligibility-only audit for the frozen independent R1 retriever."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Sequence

from .config import E7Config
from .frozen_local_retriever import RetrievalHit, load_local_corpus
from .local_target_selection import DevQueryCandidates, load_dev_query_candidates
from .retriever_registry import load_frozen_retriever
from .utils import (
    canonical_json_sha256,
    read_json,
    sha256_file,
    write_e7_json_once,
)


@dataclass(frozen=True)
class CandidateRankAudit:
    document_id: str
    original_text_index: int
    is_original_autogeo_target: bool
    r1_rank: int
    r1_score: float
    eligible: bool
    rank_range: str

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True)
class QueryEligibilityAudit:
    query_id: str
    query: str
    corpus_size: int
    top_10: tuple[RetrievalHit, ...]
    original_candidate_ids: tuple[str, ...]
    original_candidate_rankings: tuple[CandidateRankAudit, ...]
    eligible: bool
    eligible_candidate_ids: tuple[str, ...]
    eligible_slot_count: int

    def to_dict(self) -> dict[str, object]:
        return {
            "query_id": self.query_id,
            "query": self.query,
            "corpus_size": self.corpus_size,
            "top_10": [hit.to_dict() for hit in self.top_10],
            "original_candidate_ids": list(self.original_candidate_ids),
            "original_candidate_rankings": [
                candidate.to_dict()
                for candidate in self.original_candidate_rankings
            ],
            "eligible": self.eligible,
            "eligible_candidate_ids": list(self.eligible_candidate_ids),
            "eligible_slot_count": self.eligible_slot_count,
        }


def rank_range(rank: int, config: E7Config) -> str:
    """Return one of the five frozen descriptive rank ranges."""
    if 1 <= rank <= config.ge_top_k:
        return "rank_1_5"
    if config.preferred_target_rank_min <= rank <= config.preferred_target_rank_max:
        return "rank_6_20"
    if config.preferred_target_rank_max < rank <= config.medium_target_rank_max:
        return "rank_21_50"
    if config.medium_target_rank_max < rank <= config.max_target_rank:
        return "rank_51_100"
    if rank > config.max_target_rank:
        return "rank_over_100"
    raise ValueError(f"rank is outside the frozen E7 ranges: {rank}")


def audit_query(
    query: DevQueryCandidates,
    full_ranking: Sequence[RetrievalHit],
    config: E7Config,
) -> QueryEligibilityAudit:
    """Audit five original slots without selecting any target."""
    if len(full_ranking) < 10:
        raise ValueError("full ranking must contain at least ten documents")
    hits = {hit.document_id: hit for hit in full_ranking}
    if len(hits) != len(full_ranking):
        raise ValueError("full ranking contains duplicate physical identities")

    candidates: list[CandidateRankAudit] = []
    eligible_ids: list[str] = []
    for text_index, document_id in enumerate(query.original_candidate_ids):
        try:
            hit = hits[document_id]
        except KeyError as error:
            raise ValueError(
                f"original candidate is absent from R1 ranking: {document_id}"
            ) from error
        is_eligible = (
            config.preferred_target_rank_min
            <= hit.rank
            <= config.max_target_rank
        )
        if is_eligible and document_id not in eligible_ids:
            eligible_ids.append(document_id)
        candidates.append(
            CandidateRankAudit(
                document_id=document_id,
                original_text_index=text_index,
                is_original_autogeo_target=(
                    text_index == query.original_target_text_index
                ),
                r1_rank=hit.rank,
                r1_score=hit.score,
                eligible=is_eligible,
                rank_range=rank_range(hit.rank, config),
            )
        )
    if len(candidates) != 5:
        raise AssertionError("eligibility audit must preserve five original slots")
    return QueryEligibilityAudit(
        query_id=query.query_id,
        query=query.query,
        corpus_size=len(full_ranking),
        top_10=tuple(full_ranking[:10]),
        original_candidate_ids=query.original_candidate_ids,
        original_candidate_rankings=tuple(candidates),
        eligible=bool(eligible_ids),
        eligible_candidate_ids=tuple(eligible_ids),
        eligible_slot_count=sum(candidate.eligible for candidate in candidates),
    )


def summarize_audits(
    records: Sequence[QueryEligibilityAudit],
) -> dict[str, object]:
    candidates = [
        candidate
        for record in records
        for candidate in record.original_candidate_rankings
    ]
    range_keys = (
        "rank_1_5",
        "rank_6_20",
        "rank_21_50",
        "rank_51_100",
        "rank_over_100",
    )
    eligible_count = sum(record.eligible for record in records)
    return {
        "query_count": len(records),
        "eligible_query_count": eligible_count,
        "ineligible_query_count": len(records) - eligible_count,
        "total_original_candidate_slots": len(candidates),
        "candidate_slot_rank_distribution": {
            key: sum(candidate.rank_range == key for candidate in candidates)
            for key in range_keys
        },
    }


def load_r0_top5_retention(path: Path) -> dict[str, object]:
    """Read the frozen R0 audit descriptively; never rerun or rewrite it."""
    raw = read_json(path)
    records = raw.get("records")
    if not isinstance(records, list) or len(records) != 20:
        raise ValueError("frozen R0 audit must contain DEV20")
    ranks = [
        candidate["local_rank"]
        for record in records
        for candidate in record["original_candidate_rankings"]
    ]
    if len(ranks) != 100 or not all(isinstance(rank, int) for rank in ranks):
        raise ValueError("frozen R0 audit must preserve 100 candidate slots")
    retained = sum(1 <= rank <= 5 for rank in ranks)
    return {
        "retriever_id": "r0_local_aligned",
        "source_audit_path": str(path.resolve()),
        "source_audit_sha256": sha256_file(path),
        "original_candidate_slots": len(ranks),
        "top5_retained_slots": retained,
        "top5_retention_rate": retained / len(ranks),
    }


def run_r1_dev_eligibility_audit(
    config: E7Config | None = None,
) -> dict[str, object]:
    config = config or E7Config()
    config.validate()
    manifest_path = config.output_paths["corpus"] / "dev_corpus_manifest.json"
    corpus = load_local_corpus(manifest_path)
    if corpus.corpus_split != "dev":
        raise ValueError("R1 eligibility audit is restricted to DEV20")
    queries = load_dev_query_candidates(manifest_path)
    retriever = load_frozen_retriever(config.r1_retriever_id, corpus, config)
    if retriever.retriever_id != config.r1_retriever_id:
        raise AssertionError("eligibility audit loaded the wrong retriever")

    records: list[QueryEligibilityAudit] = []
    for query in queries:
        full_ranking = retriever.retrieve(
            query.query,
            corpus,
            top_k=len(corpus.documents),
        )
        records.append(audit_query(query, full_ranking, config))
    statistics = summarize_audits(records)
    r1_top5 = statistics["candidate_slot_rank_distribution"]["rank_1_5"]
    total_slots = statistics["total_original_candidate_slots"]
    r0_audit_path = config.output_paths["targets"] / "dev_targets.json"

    retriever_config = retriever.cache.retriever.to_dict()
    payload: dict[str, object] = {
        "schema_version": "e7_dev_r1_eligibility_audit_v1",
        "protocol": "researchy_geo_pooled_local",
        "split": "dev",
        "retriever_id": retriever.retriever_id,
        "retriever_config_hash": canonical_json_sha256(retriever_config),
        "retriever_config": retriever_config,
        "embedding_cache_id": retriever.cache.cache_id,
        "embedding_cache_sha256": retriever.cache.embeddings_sha256,
        "corpus_id": corpus.corpus_id,
        "corpus_size": len(corpus.documents),
        "corpus_manifest_path": str(manifest_path.resolve()),
        "corpus_manifest_hash": corpus.manifest_sha256,
        "corpus_composition": ["rule_pool", "dev"],
        "eligibility_definition": {
            "candidate_scope": "query_original_five_only",
            "minimum_rank": config.preferred_target_rank_min,
            "maximum_rank": config.max_target_rank,
            "ge_top_k": config.ge_top_k,
        },
        "eligibility_statistics": statistics,
        "r0_vs_r1_diagnostic": {
            "comparison_type": "descriptive_only",
            "r0": load_r0_top5_retention(r0_audit_path),
            "r1": {
                "retriever_id": config.r1_retriever_id,
                "original_candidate_slots": total_slots,
                "top5_retained_slots": r1_top5,
                "top5_retention_rate": r1_top5 / total_slots,
            },
            "retriever_selection_or_tuning_performed": False,
        },
        "per_query_candidate_ranks": [record.to_dict() for record in records],
        "selection_policy_executed": False,
        "formal_target_manifest_created": False,
        "rewrite_calls_made": False,
        "ge_geo_geu_calls_made": False,
        "llm_calls_made": False,
        "test50_processed": False,
    }
    destination = config.output_paths["targets"] / "dev_eligibility_r1.json"
    if destination.exists():
        if read_json(destination) != payload:
            raise FileExistsError(
                f"refusing to replace changed R1 eligibility audit: {destination}"
            )
        action = "reused_identical"
    else:
        write_e7_json_once(destination, payload, config)
        destination.chmod(0o444)
        action = "created"
    return {
        "output_path": str(destination),
        "action": action,
        "eligibility_statistics": statistics,
        "r0_vs_r1_diagnostic": payload["r0_vs_r1_diagnostic"],
    }


def main() -> None:
    print(
        json.dumps(
            run_r1_dev_eligibility_audit(),
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
