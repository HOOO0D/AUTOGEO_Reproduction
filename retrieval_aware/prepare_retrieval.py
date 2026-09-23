"""Optional future ClueWeb preparation entry point (no rewrite or GE).

Formal E7 corpus construction is implemented in ``corpus_builder``.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import replace
from pathlib import Path

from .config import E7Config
from .overlap_audit import (
    OriginalDocumentIdentity,
    audit_original_top5_overlap,
    persist_overlap_audit,
)
from .pool_builder import (
    DeepResearchGymClueWeb22B,
    RetrievalAuthenticationError,
    RetrievalPool,
    build_retrieval_pool,
    load_retrieval_pool,
    persist_retrieval_pool,
)
from .query_mapping import (
    AutoGEOQuestion,
    MappingStatus,
    ResearchyQuestionsJSONLSource,
    ResearchyQuestionRow,
    build_query_mapping,
    load_autogeo_questions,
)
from .target_selector import persist_target_selection, select_target
from .utils import ensure_output_layout, write_e7_json


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Prepare E7 query mappings and optional immutable Top-N pools."
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Number of local queries; defaults to config.smoke_query_limit.",
    )
    parser.add_argument(
        "--run-retrieval",
        action="store_true",
        help="Call the official ClueWeb22-B endpoint. Requires CLUEWEB_API_KEY.",
    )
    parser.add_argument(
        "--frozen-query-dir",
        type=Path,
        default=None,
        help="Read-only Researchy-GEO chunk directory override.",
    )
    return parser


def _official_researchy_rows(
    config: E7Config,
    autogeo_questions: list[AutoGEOQuestion],
) -> list[ResearchyQuestionRow]:
    source = ResearchyQuestionsJSONLSource(
        url=config.researchy_test_jsonl_url,
        timeout_seconds=config.request_timeout_seconds,
        max_retries=config.researchy_download_max_retries,
        retry_backoff_seconds=config.researchy_retry_backoff_seconds,
    )
    return source.load_matching_rows(question.query for question in autogeo_questions)


def _load_or_create_pool(
    mapping,
    backend: DeepResearchGymClueWeb22B,
    config: E7Config,
) -> tuple[RetrievalPool, str]:
    pool_path = config.output_paths["pools"] / f"{mapping.e7_query_id}.json"
    if pool_path.exists():
        pool = load_retrieval_pool(pool_path)
        expected = (
            pool.e7_query_id == mapping.e7_query_id
            and pool.autogeo_question_id == mapping.autogeo_question_id
            and pool.researchy_question_id == mapping.researchy_question_id
            and pool.query == mapping.original_query
            and pool.corpus == config.clueweb_retrieval_corpus
            and pool.retriever == config.retrieval_model
            and pool.requested_top_n == config.pool_size
            and len(pool.documents) == config.pool_size
        )
        if not expected:
            raise ValueError(
                "existing immutable pool does not match the mapping/config"
            )
        return pool, "reused"
    pool = build_retrieval_pool(mapping, backend, config)
    persist_retrieval_pool(pool, config)
    return pool, "created"


def run(
    config: E7Config,
    limit: int,
    run_retrieval: bool,
) -> dict[str, object]:
    """Run mapping and, only when explicitly enabled, retrieval/selection."""
    if not 1 <= limit <= 3:
        raise ValueError("smoke run limit must remain within 1--3")
    config.validate()
    ensure_output_layout(config)

    autogeo_questions = load_autogeo_questions(config.frozen_query_dir, limit=limit)
    researchy_rows = _official_researchy_rows(config, autogeo_questions)
    researchy_by_id = {row.researchy_question_id: row for row in researchy_rows}
    mappings = build_query_mapping(autogeo_questions, researchy_rows)
    mapping_path = config.output_paths["mappings"] / "smoke_query_mapping.json"
    write_e7_json(
        mapping_path,
        {
            "schema_version": "e7_query_mapping_v1",
            "source_dataset": config.researchy_dataset_id,
            "source_config": config.researchy_dataset_config,
            "source_split": config.researchy_dataset_split,
            "records": [mapping.to_dict() for mapping in mappings],
        },
        config,
    )

    summary: dict[str, object] = {
        "schema_version": "e7_retrieval_smoke_v1",
        "mapping_path": str(mapping_path),
        "mapping": [mapping.to_dict() for mapping in mappings],
        "retrieval_requested": run_retrieval,
        "queries": [],
        "rewrite_or_llm_or_ge_calls_made": False,
    }
    if not run_retrieval:
        summary["retrieval_status"] = "not_requested"
        summary_path = config.output_paths["logs"] / "retrieval_smoke_summary.json"
        write_e7_json(summary_path, summary, config)
        summary["summary_path"] = str(summary_path)
        return summary

    try:
        backend = DeepResearchGymClueWeb22B.from_config(config)
    except RetrievalAuthenticationError:
        summary["retrieval_status"] = "blocked_missing_api_key"
        summary["blocking_reason"] = (
            f"environment variable {config.retrieval_api_key_env} is not set"
        )
        summary_path = config.output_paths["logs"] / "retrieval_smoke_summary.json"
        write_e7_json(summary_path, summary, config)
        summary["summary_path"] = str(summary_path)
        return summary

    summary["retrieval_status"] = "running"
    local_by_id = {
        question.autogeo_question_id: question for question in autogeo_questions
    }
    query_summaries: list[dict[str, object]] = []
    for mapping in mappings:
        if mapping.mapping_status is not MappingStatus.MAPPED:
            query_summaries.append(
                {
                    "e7_query_id": mapping.e7_query_id,
                    "mapping_status": mapping.mapping_status.value,
                    "retrieval_status": "skipped",
                }
            )
            continue
        if mapping.researchy_question_id is None:
            raise ValueError("mapped record unexpectedly lacks Researchy id")
        researchy_row = researchy_by_id[mapping.researchy_question_id]
        pool, pool_action = _load_or_create_pool(mapping, backend, config)
        selection = select_target(pool, researchy_row, config)
        target_path = persist_target_selection(selection, config)

        local_question = local_by_id[mapping.autogeo_question_id]
        original_identities = tuple(
            OriginalDocumentIdentity(original_index=index, text=text)
            for index, text in enumerate(local_question.text_list)
        )
        overlap = audit_original_top5_overlap(
            mapping.e7_query_id,
            original_identities,
            pool,
        )
        overlap_path = persist_overlap_audit(overlap, config)
        query_summaries.append(
            {
                "e7_query_id": mapping.e7_query_id,
                "mapping_status": mapping.mapping_status.value,
                "pool_id": pool.pool_id,
                "pool_action": pool_action,
                "pool_size": len(pool.documents),
                "target": selection.to_dict(),
                "target_path": str(target_path),
                "original5_recall_at_100": overlap.original5_recall_at_100,
                "overlap_path": str(overlap_path),
            }
        )
    summary["queries"] = query_summaries
    summary["retrieval_status"] = "complete"
    summary_path = config.output_paths["logs"] / "retrieval_smoke_summary.json"
    write_e7_json(summary_path, summary, config)
    summary["summary_path"] = str(summary_path)
    return summary


def main() -> None:
    args = _build_parser().parse_args()
    config = E7Config()
    if args.frozen_query_dir is not None:
        config = replace(config, frozen_query_dir=args.frozen_query_dir)
    limit = args.limit if args.limit is not None else config.smoke_query_limit
    summary = run(config, limit=limit, run_retrieval=args.run_retrieval)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
