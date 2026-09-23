"""Freeze formal DEV targets from the immutable R1/Pool V2 eligibility audit."""

from __future__ import annotations

import json
import statistics
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Sequence

from .config import E7Config
from .pool_v2_eligibility_audit import is_eligible_rank, rank_bucket
from .utils import read_json, sha256_file, write_e7_json_once


EXPECTED_ELIGIBILITY_RESULT_SHA256 = (
    "fdfec14df6b8435890661de9c77ef900f1ff1d4bb0db08ec14263d21a2b614a8"
)
EXPECTED_POOL_V2_MANIFEST_SHA256 = (
    "423e4cd932222fca8e16ba47c230393f5c9a67a00b8c10f05a0292dbe27f55d8"
)
EXPECTED_RETRIEVER_CONFIG_SHA256 = (
    "6ac758e1ee097a0df3c4df14e8d3d1534a5374a6e933b4ea355158fb48477ba2"
)
EXPECTED_CORPUS_SIZE = 44_264
EXPECTED_DEV_QUERY_COUNT = 20

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
            "contains_original_autogeo_target_id": (
                self.contains_original_autogeo_target_id
            ),
            "initial_r1_rank": self.initial_r1_rank,
            "initial_r1_score": self.initial_r1_score,
            "eligible": self.eligible,
            "rank_bucket": self.rank_bucket,
        }


def _require_int(value: object, field_name: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool):
        raise ValueError(f"{field_name} must be an integer")
    return value


def _load_and_validate_inputs(
    eligibility_path: Path,
    manifest_path: Path,
    config: E7Config,
) -> tuple[dict[str, object], dict[str, str]]:
    """Load only frozen artifacts and validate every protocol identity."""
    eligibility_sha256 = sha256_file(eligibility_path)
    manifest_sha256 = sha256_file(manifest_path)
    if eligibility_sha256 != EXPECTED_ELIGIBILITY_RESULT_SHA256:
        raise ValueError("frozen Pool V2 eligibility result SHA256 changed")
    if manifest_sha256 != EXPECTED_POOL_V2_MANIFEST_SHA256:
        raise ValueError("frozen Pool V2 corpus manifest SHA256 changed")

    eligibility = read_json(eligibility_path)
    manifest = read_json(manifest_path)
    if eligibility.get("schema_version") != (
        "e7_dev_r1_pool_v2_eligibility_audit_v1"
    ):
        raise ValueError("unsupported Pool V2 eligibility schema")
    if eligibility.get("split") != "dev" or eligibility.get("test50_processed"):
        raise ValueError("formal target selection is restricted to frozen DEV20")
    if eligibility.get("retriever_id") != config.r1_retriever_id:
        raise ValueError("eligibility result does not use frozen R1")
    if eligibility.get("retriever_config_sha256") != (
        EXPECTED_RETRIEVER_CONFIG_SHA256
    ):
        raise ValueError("frozen R1 configuration SHA256 changed")
    if eligibility.get("corpus_manifest_sha256") != manifest_sha256:
        raise ValueError("eligibility result and Pool V2 manifest disagree")
    if eligibility.get("corpus_size") != EXPECTED_CORPUS_SIZE:
        raise ValueError("eligibility result corpus size changed")
    definition = eligibility.get("eligibility_definition")
    expected_definition = {
        "candidate_scope": "unique_physical_documents_from_query_original_five",
        "minimum_rank": config.preferred_target_rank_min,
        "maximum_rank": config.max_target_rank,
        "ge_top_k": config.ge_top_k,
        "duplicate_slots_counted_once_for_retrieval_rank": True,
        "slot_provenance_preserved": True,
    }
    if definition != expected_definition:
        raise ValueError("frozen eligibility definition changed")
    if eligibility.get("selection_policy_executed") is not False:
        raise ValueError("eligibility input must predate formal target selection")

    if manifest.get("schema_version") != "e7_researchy_pooled_corpus_v2":
        raise ValueError("formal selection requires Pool V2")
    if manifest.get("corpus_split") != "dev":
        raise ValueError("TEST50 corpus is forbidden during DEV target selection")
    if manifest.get("corpus_id") != eligibility.get("corpus_id"):
        raise ValueError("eligibility result and corpus ID disagree")
    documents = manifest.get("documents")
    if not isinstance(documents, list) or len(documents) != EXPECTED_CORPUS_SIZE:
        raise ValueError("Pool V2 physical document count changed")
    text_hash_by_id: dict[str, str] = {}
    for document in documents:
        if not isinstance(document, dict):
            raise ValueError("Pool V2 document row must be an object")
        document_id = document.get("document_id")
        text_hash = document.get("text_hash")
        if not isinstance(document_id, str) or not isinstance(text_hash, str):
            raise ValueError("Pool V2 document lacks physical identity or text hash")
        if document_id in text_hash_by_id:
            raise ValueError("Pool V2 contains duplicate physical document IDs")
        text_hash_by_id[document_id] = text_hash

    query_rows = manifest.get("query_candidate_sets")
    if not isinstance(query_rows, list) or len(query_rows) != EXPECTED_DEV_QUERY_COUNT:
        raise ValueError("Pool V2 manifest must retain DEV20 query provenance")
    manifest_queries = {row.get("question_id"): row for row in query_rows}
    if len(manifest_queries) != EXPECTED_DEV_QUERY_COUNT:
        raise ValueError("Pool V2 query IDs must be unique")
    audit_rows = eligibility.get("per_query_candidate_ranks")
    if not isinstance(audit_rows, list) or len(audit_rows) != EXPECTED_DEV_QUERY_COUNT:
        raise ValueError("eligibility result must contain DEV20 records")
    for audit_row in audit_rows:
        if not isinstance(audit_row, dict):
            raise ValueError("eligibility query row must be an object")
        query_id = audit_row.get("query_id")
        manifest_row = manifest_queries.get(query_id)
        if not isinstance(manifest_row, dict):
            raise ValueError(f"query absent from Pool V2 manifest: {query_id}")
        if audit_row.get("query") != manifest_row.get("query"):
            raise ValueError(f"query text changed for {query_id}")
        manifest_ids = manifest_row.get("original_candidate_ids")
        audit_slots = audit_row.get("original_candidate_slots")
        if not isinstance(manifest_ids, list) or len(manifest_ids) != 5:
            raise ValueError(f"manifest candidate slots changed for {query_id}")
        if not isinstance(audit_slots, list) or len(audit_slots) != 5:
            raise ValueError(f"eligibility candidate slots changed for {query_id}")
        slot_ids = [slot.get("physical_document_id") for slot in audit_slots]
        slot_indices = [slot.get("source_slot") for slot in audit_slots]
        if slot_ids != manifest_ids or slot_indices != list(range(5)):
            raise ValueError(f"slot provenance changed for {query_id}")
        target_index = manifest_row.get("original_target_text_index")
        if not isinstance(target_index, int) or not 0 <= target_index < 5:
            raise ValueError(f"invalid original target_id for {query_id}")
        target_slots = [
            slot["source_slot"]
            for slot in audit_slots
            if slot.get("is_original_autogeo_target_id") is True
        ]
        if target_slots != [target_index]:
            raise ValueError(f"original target_id provenance changed for {query_id}")
    return eligibility, text_hash_by_id


def _candidate_rows(
    audit_row: Mapping[str, object],
    text_hash_by_id: Mapping[str, str],
    config: E7Config,
) -> tuple[CandidateForSelection, ...]:
    raw_slots = audit_row.get("original_candidate_slots")
    raw_candidates = audit_row.get("unique_original_candidates")
    if not isinstance(raw_slots, list) or not isinstance(raw_candidates, list):
        raise ValueError("eligibility row lacks candidate provenance")
    unique_slot_ids = {slot.get("physical_document_id") for slot in raw_slots}
    candidate_ids = {row.get("physical_document_id") for row in raw_candidates}
    if unique_slot_ids != candidate_ids:
        raise ValueError("physical candidate set differs from the original five slots")

    candidates: list[CandidateForSelection] = []
    for raw in raw_candidates:
        if not isinstance(raw, dict):
            raise ValueError("physical candidate audit must be an object")
        document_id = raw.get("physical_document_id")
        source_slots = raw.get("source_slots")
        rank = _require_int(raw.get("rank"), "candidate rank")
        score = raw.get("score")
        contains_target = raw.get("contains_original_autogeo_target_id")
        if not isinstance(document_id, str) or document_id not in text_hash_by_id:
            raise ValueError("physical candidate is absent from Pool V2")
        if (
            not isinstance(source_slots, list)
            or not source_slots
            or not all(isinstance(value, int) for value in source_slots)
        ):
            raise ValueError("candidate source slots are invalid")
        expected_slots = [
            slot["source_slot"]
            for slot in raw_slots
            if slot["physical_document_id"] == document_id
        ]
        if source_slots != expected_slots:
            raise ValueError("physical candidate duplicate provenance changed")
        if not isinstance(score, (int, float)) or isinstance(score, bool):
            raise ValueError("candidate score must be numeric")
        if not isinstance(contains_target, bool):
            raise ValueError("candidate target provenance must be boolean")
        expected_eligible = is_eligible_rank(rank, config)
        expected_bucket = rank_bucket(rank, config)
        if raw.get("eligible") is not expected_eligible:
            raise ValueError("candidate eligibility differs from frozen boundary")
        if raw.get("rank_bucket") != expected_bucket:
            raise ValueError("candidate rank bucket differs from frozen boundary")
        candidates.append(
            CandidateForSelection(
                physical_document_id=document_id,
                text_hash=text_hash_by_id[document_id],
                source_text_indices=tuple(source_slots),
                contains_original_autogeo_target_id=contains_target,
                initial_r1_rank=rank,
                initial_r1_score=float(score),
                eligible=expected_eligible,
                rank_bucket=expected_bucket,
            )
        )
    if len({candidate.physical_document_id for candidate in candidates}) != len(
        candidates
    ):
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
    """Apply the frozen target policy to one already-audited DEV query."""
    config = config or E7Config()
    candidates = _candidate_rows(audit_row, text_hash_by_id, config)
    target_candidates = [
        candidate
        for candidate in candidates
        if candidate.contains_original_autogeo_target_id
    ]
    original_target = target_candidates[0]
    raw_slots = audit_row["original_candidate_slots"]
    original_target_slots = [
        slot["source_slot"]
        for slot in raw_slots
        if slot.get("is_original_autogeo_target_id") is True
    ]
    if len(original_target_slots) != 1:
        raise ValueError("exactly one original source slot must carry target_id")
    original_target_id = original_target_slots[0]
    if original_target.eligible:
        selected = original_target
        target_source = TARGET_SOURCE_ORIGINAL
        selection_reason = (
            "original_target_id_physical_document_is_eligible_rank_6_100"
        )
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
            selection_reason = (
                f"original_target_id_ineligible_selected_lowest_rank_"
                f"{selected.rank_bucket}_alternate_with_document_id_tiebreak"
            )
        else:
            selected = None
            target_source = None
            selection_reason = (
                "no_unique_original_candidate_physical_document_within_rank_6_100"
            )

    audit_eligible = audit_row.get("eligible")
    if audit_eligible is not (selected is not None):
        raise ValueError("selection eligibility disagrees with frozen audit")
    record: dict[str, object] = {
        "query_id": audit_row.get("query_id"),
        "query": audit_row.get("query"),
        "eligible": selected is not None,
        "status": "selected" if selected is not None else "ineligible",
        "target_document_id": (
            selected.physical_document_id if selected is not None else None
        ),
        "target_text_hash": selected.text_hash if selected is not None else None,
        "target_source": target_source,
        "source_text_indices": (
            list(selected.source_text_indices) if selected is not None else None
        ),
        "original_autogeo_target_id": original_target_id,
        "initial_r1_rank": (
            selected.initial_r1_rank if selected is not None else None
        ),
        "initial_r1_score": (
            selected.initial_r1_score if selected is not None else None
        ),
        "rank_bucket": selected.rank_bucket if selected is not None else None,
        "original_candidate_rankings": [
            candidate.to_dict() for candidate in candidates
        ],
        "retriever_id": retriever_id,
        "retriever_config_sha256": retriever_config_sha256,
        "corpus_manifest_sha256": corpus_manifest_sha256,
        "eligibility_result_sha256": eligibility_result_sha256,
        "selection_reason": selection_reason,
    }
    return record


def summarize_targets(records: Sequence[Mapping[str, object]]) -> dict[str, object]:
    selected = [record for record in records if record.get("eligible") is True]
    ranks = [_require_int(record.get("initial_r1_rank"), "selected rank") for record in selected]
    return {
        "dev_query_count": len(records),
        "eligible_count": len(selected),
        "ineligible_count": len(records) - len(selected),
        "selected_original_target_id_count": sum(
            record.get("target_source") == TARGET_SOURCE_ORIGINAL
            for record in selected
        ),
        "selected_alternate_candidate_count": sum(
            record.get("target_source") == TARGET_SOURCE_ALTERNATE
            for record in selected
        ),
        "selected_target_rank_bucket_distribution": {
            bucket: sum(record.get("rank_bucket") == bucket for record in selected)
            for bucket in ("easy", "medium", "hard")
        },
        "selected_target_rank_statistics": {
            "min": min(ranks),
            "max": max(ranks),
            "mean": statistics.mean(ranks),
            "median": statistics.median(ranks),
        },
    }


def run_pool_v2_dev_target_selection(
    config: E7Config | None = None,
) -> dict[str, object]:
    """Create or identically reuse the immutable DEV R1/Pool V2 target manifest."""
    config = config or E7Config()
    config.validate()
    eligibility_path = (
        config.output_paths["targets"] / "dev_eligibility_r1_pool_v2.json"
    )
    manifest_path = (
        config.output_paths["corpus"] / "pool_v2" / "dev_corpus_manifest.json"
    )
    eligibility, text_hash_by_id = _load_and_validate_inputs(
        eligibility_path,
        manifest_path,
        config,
    )
    audit_rows = eligibility["per_query_candidate_ranks"]
    records = [
        select_query_target(
            row,
            text_hash_by_id,
            retriever_id=eligibility["retriever_id"],
            retriever_config_sha256=eligibility["retriever_config_sha256"],
            corpus_manifest_sha256=eligibility["corpus_manifest_sha256"],
            eligibility_result_sha256=EXPECTED_ELIGIBILITY_RESULT_SHA256,
            config=config,
        )
        for row in audit_rows
    ]
    summary = summarize_targets(records)
    frozen_statistics = eligibility.get("eligibility_statistics")
    if not isinstance(frozen_statistics, dict):
        raise ValueError("eligibility statistics are missing")
    if (
        summary["eligible_count"] != frozen_statistics.get("eligible_query_count")
        or summary["ineligible_count"]
        != frozen_statistics.get("ineligible_query_count")
    ):
        raise ValueError("selection counts disagree with frozen eligibility audit")

    payload: dict[str, object] = {
        "schema_version": "e7_dev_targets_r1_pool_v2_frozen_v1",
        "protocol": "researchy_geo_pooled_local",
        "split": "dev",
        "status": "frozen",
        "corpus_id": eligibility["corpus_id"],
        "corpus_size": eligibility["corpus_size"],
        "retriever_id": eligibility["retriever_id"],
        "retriever_config_sha256": eligibility["retriever_config_sha256"],
        "corpus_manifest_sha256": eligibility["corpus_manifest_sha256"],
        "eligibility_result_path": str(eligibility_path.resolve()),
        "eligibility_result_sha256": EXPECTED_ELIGIBILITY_RESULT_SHA256,
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
        "test50_processed": False,
    }
    destination = (
        config.output_paths["targets"]
        / "dev_targets_r1_pool_v2.frozen.json"
    )
    if destination.exists():
        if read_json(destination) != payload:
            raise FileExistsError(
                f"refusing to replace changed frozen target manifest: {destination}"
            )
        action = "reused_identical"
    else:
        write_e7_json_once(destination, payload, config)
        destination.chmod(0o444)
        action = "created"
    return {
        "output_path": str(destination),
        "action": action,
        "output_sha256": sha256_file(destination),
        "summary": summary,
    }


def main() -> None:
    print(
        json.dumps(
            run_pool_v2_dev_target_selection(),
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
