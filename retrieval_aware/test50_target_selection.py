"""Freeze TEST50 targets from an immutable eligibility-only audit.

Targets may only be unique physical documents represented by a query's five
original Researchy-GEO slots.  No retrieval, embedding, rewriting, or external
API call occurs in this stage.
"""

from __future__ import annotations

import json
import statistics
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Sequence

from .config import E7Config
from .test50_eligibility_audit import is_eligible_rank, rank_bucket
from .test50_pool_v2 import (
    EXPECTED_TEST_QUERY_COUNT,
    _persist_immutable_json,
    test_pool_v2_paths,
)
from .test50_pool_v2_embeddings import FROZEN_R1_CONFIG_SHA256
from .utils import read_json, sha256_file


TARGET_SOURCE_ORIGINAL = "original_target_id"
TARGET_SOURCE_ALTERNATE = "alternate_original_candidate"
BUCKET_PRIORITY = {"easy": 0, "medium": 1, "hard": 2}


@dataclass(frozen=True)
class CandidateForSelection:
    physical_document_id: str
    text_hash: str
    source_text_indices: tuple[int, ...]
    contains_original_autogeo_target_id: bool
    initial_r1_rank: int
    initial_r1_score: float
    eligible: bool
    rank_bucket: str

    def to_dict(self) -> dict[str, object]:
        return {
            "physical_document_id": self.physical_document_id,
            "text_hash": self.text_hash,
            "source_text_indices": list(self.source_text_indices),
            "contains_original_autogeo_target_id": self.contains_original_autogeo_target_id,
            "initial_r1_rank": self.initial_r1_rank,
            "initial_r1_score": self.initial_r1_score,
            "eligible": self.eligible,
            "rank_bucket": self.rank_bucket,
        }


def _require_int(value: object, field: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool):
        raise ValueError(f"{field} must be an integer")
    return value


def _candidate_rows(
    audit_row: Mapping[str, object],
    text_hash_by_id: Mapping[str, str],
    config: E7Config,
) -> tuple[CandidateForSelection, ...]:
    raw_slots = audit_row.get("original_candidate_slots")
    raw_candidates = audit_row.get("unique_original_candidates")
    if not isinstance(raw_slots, list) or len(raw_slots) != 5:
        raise ValueError("eligibility row must retain five source slots")
    if not isinstance(raw_candidates, list) or not raw_candidates:
        raise ValueError("eligibility row lacks physical candidates")
    slot_ids = {slot.get("physical_document_id") for slot in raw_slots}
    candidate_ids = {candidate.get("physical_document_id") for candidate in raw_candidates}
    if slot_ids != candidate_ids:
        raise ValueError("candidate scope differs from the original five slots")

    candidates: list[CandidateForSelection] = []
    for raw in raw_candidates:
        if not isinstance(raw, dict):
            raise ValueError("physical candidate must be an object")
        document_id = raw.get("physical_document_id")
        source_slots = raw.get("source_slots")
        rank = _require_int(raw.get("rank"), "candidate rank")
        score = raw.get("score")
        contains_target = raw.get("contains_original_autogeo_target_id")
        if not isinstance(document_id, str) or document_id not in text_hash_by_id:
            raise ValueError("candidate is absent from TEST Pool V2")
        if not isinstance(source_slots, list) or not source_slots:
            raise ValueError("candidate source slots are missing")
        if not all(isinstance(slot, int) and not isinstance(slot, bool) for slot in source_slots):
            raise ValueError("candidate source slots must be integers")
        expected_slots = [
            slot["source_slot"]
            for slot in raw_slots
            if slot.get("physical_document_id") == document_id
        ]
        if source_slots != expected_slots:
            raise ValueError("duplicate slot provenance changed")
        if not isinstance(score, (int, float)) or isinstance(score, bool):
            raise ValueError("candidate score must be numeric")
        if not isinstance(contains_target, bool):
            raise ValueError("candidate target provenance must be boolean")
        eligible = is_eligible_rank(rank, config)
        bucket = rank_bucket(rank, config)
        if raw.get("eligible") is not eligible or raw.get("rank_bucket") != bucket:
            raise ValueError("candidate eligibility or bucket changed")
        candidates.append(
            CandidateForSelection(
                physical_document_id=document_id,
                text_hash=text_hash_by_id[document_id],
                source_text_indices=tuple(source_slots),
                contains_original_autogeo_target_id=contains_target,
                initial_r1_rank=rank,
                initial_r1_score=float(score),
                eligible=eligible,
                rank_bucket=bucket,
            )
        )
    if len({candidate.physical_document_id for candidate in candidates}) != len(candidates):
        raise ValueError("duplicate physical candidates were not collapsed")
    if sum(candidate.contains_original_autogeo_target_id for candidate in candidates) != 1:
        raise ValueError("exactly one physical candidate must contain target_id")
    return tuple(candidates)


def select_query_target(
    audit_row: Mapping[str, object],
    text_hash_by_id: Mapping[str, str],
    *,
    retriever_id: str,
    retriever_config_sha256: str,
    corpus_manifest_sha256: str,
    eligibility_result_sha256: str,
    config: E7Config | None = None,
) -> dict[str, object]:
    config = config or E7Config()
    candidates = _candidate_rows(audit_row, text_hash_by_id, config)
    original_target = next(
        candidate
        for candidate in candidates
        if candidate.contains_original_autogeo_target_id
    )
    raw_slots = audit_row["original_candidate_slots"]
    target_slots = [
        slot["source_slot"]
        for slot in raw_slots
        if slot.get("is_original_autogeo_target_id") is True
    ]
    if len(target_slots) != 1:
        raise ValueError("exactly one original slot must carry target_id")
    original_target_id = target_slots[0]
    if original_target.eligible:
        selected = original_target
        target_source = TARGET_SOURCE_ORIGINAL
        reason = "original_target_id_physical_document_is_eligible_rank_6_100"
    else:
        alternates = [
            candidate
            for candidate in candidates
            if candidate.eligible
            and not candidate.contains_original_autogeo_target_id
        ]
        if alternates:
            selected = min(
                alternates,
                key=lambda candidate: (
                    BUCKET_PRIORITY[candidate.rank_bucket],
                    candidate.initial_r1_rank,
                    candidate.physical_document_id,
                ),
            )
            target_source = TARGET_SOURCE_ALTERNATE
            reason = (
                f"original_target_id_ineligible_selected_lowest_rank_"
                f"{selected.rank_bucket}_alternate_with_document_id_tiebreak"
            )
        else:
            selected = None
            target_source = None
            reason = "no_unique_original_candidate_physical_document_within_rank_6_100"
    if audit_row.get("eligible") is not (selected is not None):
        raise ValueError("target selection disagrees with eligibility audit")
    return {
        "query_id": audit_row.get("query_id"),
        "query": audit_row.get("query"),
        "eligible": selected is not None,
        "status": "selected" if selected is not None else "ineligible",
        "target_document_id": selected.physical_document_id if selected else None,
        "target_text_hash": selected.text_hash if selected else None,
        "target_source": target_source,
        "source_text_indices": list(selected.source_text_indices) if selected else None,
        "original_autogeo_target_id": original_target_id,
        "initial_r1_rank": selected.initial_r1_rank if selected else None,
        "initial_r1_score": selected.initial_r1_score if selected else None,
        "rank_bucket": selected.rank_bucket if selected else None,
        "original_candidate_rankings": [candidate.to_dict() for candidate in candidates],
        "retriever_id": retriever_id,
        "retriever_config_sha256": retriever_config_sha256,
        "corpus_manifest_sha256": corpus_manifest_sha256,
        "eligibility_result_sha256": eligibility_result_sha256,
        "selection_reason": reason,
    }


def summarize_targets(records: Sequence[Mapping[str, object]]) -> dict[str, object]:
    selected = [record for record in records if record.get("eligible") is True]
    ranks = [_require_int(record.get("initial_r1_rank"), "selected rank") for record in selected]
    rank_statistics: dict[str, object]
    if ranks:
        rank_statistics = {
            "min": min(ranks),
            "max": max(ranks),
            "mean": statistics.mean(ranks),
            "median": statistics.median(ranks),
        }
    else:
        rank_statistics = {"min": None, "max": None, "mean": None, "median": None}
    return {
        "test_query_count": len(records),
        "eligible_count": len(selected),
        "ineligible_count": len(records) - len(selected),
        "selected_original_target_id_count": sum(
            record.get("target_source") == TARGET_SOURCE_ORIGINAL for record in selected
        ),
        "selected_alternate_candidate_count": sum(
            record.get("target_source") == TARGET_SOURCE_ALTERNATE for record in selected
        ),
        "selected_target_rank_bucket_distribution": {
            bucket: sum(record.get("rank_bucket") == bucket for record in selected)
            for bucket in ("easy", "medium", "hard")
        },
        "selected_target_rank_statistics": rank_statistics,
    }


def _load_inputs(
    eligibility_path: Path,
    manifest_path: Path,
    config: E7Config,
    *,
    expected_query_count: int = EXPECTED_TEST_QUERY_COUNT,
) -> tuple[dict[str, object], dict[str, str]]:
    eligibility = read_json(eligibility_path)
    manifest = read_json(manifest_path)
    if eligibility.get("schema_version") != "e7_test50_r1_pool_v2_eligibility_audit_v1":
        raise ValueError("unsupported TEST eligibility schema")
    if eligibility.get("split") != "test":
        raise ValueError("TEST target selection cannot consume DEV eligibility")
    if eligibility.get("retriever_id") != config.r1_retriever_id:
        raise ValueError("TEST eligibility does not use frozen R1")
    if eligibility.get("retriever_config_sha256") != FROZEN_R1_CONFIG_SHA256:
        raise ValueError("TEST eligibility R1 config hash changed")
    manifest_sha256 = sha256_file(manifest_path)
    if eligibility.get("corpus_manifest_sha256") != manifest_sha256:
        raise ValueError("TEST eligibility and corpus manifest disagree")
    if eligibility.get("corpus_id") != manifest.get("corpus_id"):
        raise ValueError("TEST eligibility and corpus identity disagree")
    expected_definition = {
        "candidate_scope": "unique_physical_documents_from_query_original_five",
        "minimum_rank": config.preferred_target_rank_min,
        "maximum_rank": config.max_target_rank,
        "ge_top_k": config.ge_top_k,
        "duplicate_slots_counted_once_for_retrieval_rank": True,
        "slot_provenance_preserved": True,
    }
    if eligibility.get("eligibility_definition") != expected_definition:
        raise ValueError("TEST eligibility definition changed")
    if eligibility.get("selection_policy_executed") is not False:
        raise ValueError("eligibility input already performed target selection")
    documents = manifest.get("documents")
    query_rows = manifest.get("query_candidate_sets")
    audit_rows = eligibility.get("per_query_candidate_ranks")
    if not isinstance(documents, list) or not documents:
        raise ValueError("TEST Pool V2 documents are missing")
    if not isinstance(query_rows, list) or len(query_rows) != expected_query_count:
        raise ValueError("TEST Pool V2 query count changed")
    if not isinstance(audit_rows, list) or len(audit_rows) != expected_query_count:
        raise ValueError("TEST eligibility query count changed")
    text_hash_by_id: dict[str, str] = {}
    for document in documents:
        document_id = document.get("document_id")
        text_hash = document.get("text_hash")
        if not isinstance(document_id, str) or not isinstance(text_hash, str):
            raise ValueError("TEST Pool V2 document identity is invalid")
        if document_id in text_hash_by_id:
            raise ValueError("TEST Pool V2 contains duplicate physical IDs")
        text_hash_by_id[document_id] = text_hash
    manifest_queries = {row.get("question_id"): row for row in query_rows}
    if len(manifest_queries) != expected_query_count:
        raise ValueError("TEST Pool V2 query IDs are not unique")
    for audit_row in audit_rows:
        query_id = audit_row.get("query_id")
        manifest_row = manifest_queries.get(query_id)
        if not isinstance(manifest_row, dict):
            raise ValueError(f"TEST query absent from corpus manifest: {query_id}")
        if audit_row.get("query") != manifest_row.get("query"):
            raise ValueError("TEST query text changed")
        slots = audit_row.get("original_candidate_slots")
        ids = manifest_row.get("original_candidate_ids")
        if not isinstance(slots, list) or len(slots) != 5:
            raise ValueError("TEST eligibility slot provenance changed")
        if [slot.get("source_slot") for slot in slots] != list(range(5)):
            raise ValueError("TEST source slot order changed")
        if [slot.get("physical_document_id") for slot in slots] != ids:
            raise ValueError("TEST physical candidate slots changed")
        target_index = manifest_row.get("original_target_text_index")
        target_slots = [
            slot["source_slot"]
            for slot in slots
            if slot.get("is_original_autogeo_target_id") is True
        ]
        if target_slots != [target_index]:
            raise ValueError("TEST original target_id provenance changed")
    return eligibility, text_hash_by_id


def run_test50_target_selection(config: E7Config | None = None) -> dict[str, object]:
    config = config or E7Config()
    config.validate()
    eligibility_path = config.output_paths["targets"] / "test50_eligibility_r1_pool_v2.json"
    manifest_path = test_pool_v2_paths(config)["manifest"]
    input_hashes_before = {
        eligibility_path: sha256_file(eligibility_path),
        manifest_path: sha256_file(manifest_path),
    }
    eligibility, text_hash_by_id = _load_inputs(eligibility_path, manifest_path, config)
    eligibility_sha256 = input_hashes_before[eligibility_path]
    records = [
        select_query_target(
            row,
            text_hash_by_id,
            retriever_id=eligibility["retriever_id"],
            retriever_config_sha256=eligibility["retriever_config_sha256"],
            corpus_manifest_sha256=eligibility["corpus_manifest_sha256"],
            eligibility_result_sha256=eligibility_sha256,
            config=config,
        )
        for row in eligibility["per_query_candidate_ranks"]
    ]
    summary = summarize_targets(records)
    frozen_stats = eligibility.get("eligibility_statistics")
    if not isinstance(frozen_stats, dict):
        raise ValueError("TEST eligibility statistics are missing")
    if summary["eligible_count"] != frozen_stats.get("eligible_query_count"):
        raise ValueError("TEST selection and eligibility counts disagree")
    payload: dict[str, object] = {
        "schema_version": "e7_test50_targets_r1_pool_v2_frozen_v1",
        "protocol": "researchy_geo_pooled_local",
        "split": "test",
        "status": "frozen",
        "corpus_id": eligibility["corpus_id"],
        "corpus_size": eligibility["corpus_size"],
        "retriever_id": eligibility["retriever_id"],
        "retriever_config_sha256": eligibility["retriever_config_sha256"],
        "corpus_manifest_sha256": eligibility["corpus_manifest_sha256"],
        "eligibility_result_path": str(eligibility_path.resolve()),
        "eligibility_result_sha256": eligibility_sha256,
        "selection_policy": {
            "candidate_scope": "unique_physical_documents_from_query_original_five",
            "primary": "original_target_id_if_eligible_rank_6_100",
            "alternate_bucket_order": ["easy", "medium", "hard"],
            "within_bucket_order": ["lowest_r1_rank", "document_id_ascending"],
            "eligibility_rank_min": config.preferred_target_rank_min,
            "eligibility_rank_max": config.max_target_rank,
            "ge_top_k": config.ge_top_k,
        },
        "summary": summary,
        "records": records,
        "target_selection_executed": True,
        "rewrite_calls_made": False,
        "ge_geo_geu_calls_made": False,
        "llm_or_external_api_calls_made": False,
        "test50_processed": True,
    }
    destination = config.output_paths["targets"] / "test50_targets_r1_pool_v2.frozen.json"
    action, digest = _persist_immutable_json(destination, payload, config)
    if {path: sha256_file(path) for path in input_hashes_before} != input_hashes_before:
        raise AssertionError("target selection modified eligibility or corpus artifacts")
    return {
        "output_path": str(destination),
        "action": action,
        "output_sha256": digest,
        "summary": summary,
    }


def main() -> None:
    print(json.dumps(run_test50_target_selection(), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
