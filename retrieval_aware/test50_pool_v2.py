"""Build the immutable Researchy-GEO TEST50 pooled corpus V2.

This is a sibling of the frozen DEV Pool V2 implementation.  Importing this
module has no data-access side effects.  The CLI is intentionally split into a
source-audit stage and a build stage so the source definition is frozen before
any future Test ranking.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Mapping, Sequence

from .config import E7Config
from .corpus_builder import CorpusDocument, DocumentProvenance, QueryCandidateSet
from .utils import (
    canonical_json_sha256,
    normalize_document_text,
    normalized_text_sha256,
    read_json,
    sha256_file,
    write_e7_json_once,
)


TEST_POOL_V2_SOURCES = ("researchy_train", "rule_pool", "test")
TEST_POOL_V2_NAMESPACE = Path("pool_v2") / "test"
EXPECTED_TEST_QUERY_COUNT = 50


@dataclass
class _MutableDocument:
    document_id: str
    text_hash: str
    text: str
    primary_provenance: DocumentProvenance
    duplicate_provenance: list[DocumentProvenance]


def test_pool_v2_paths(config: E7Config) -> dict[str, Path]:
    root = config.output_paths["corpus"] / TEST_POOL_V2_NAMESPACE
    return {
        "root": root,
        "audit": root / "source_audit.frozen.json",
        "manifest": root / "test_corpus_manifest.json",
    }


def _source_directories(config: E7Config) -> dict[str, Path]:
    return {
        "researchy_train": config.data_root / "Researchy-GEO" / "train",
        "rule_pool": config.rule_pool_query_dir,
        "test": config.test_query_dir,
    }


def _display_path(path: Path, config: E7Config) -> str:
    resolved = path.resolve()
    try:
        return str(resolved.relative_to(config.project_root.resolve()))
    except ValueError:
        return str(resolved)


def _chunk_files(directory: Path) -> tuple[Path, ...]:
    paths = tuple(sorted(directory.glob("datachunk_*.json")))
    if not paths:
        raise FileNotFoundError(f"no Researchy-GEO chunks found in {directory}")
    return paths


def _chunk_rows(directory: Path) -> tuple[tuple[str, dict[str, object]], ...]:
    rows: list[tuple[str, dict[str, object]]] = []
    for path in _chunk_files(directory):
        raw = read_json(path)
        if not isinstance(raw, dict):
            raise ValueError(f"Researchy-GEO chunk must be an object: {path}")
        for question_id, row in raw.items():
            if not isinstance(row, dict):
                raise ValueError(f"Researchy-GEO row must be an object: {question_id}")
            rows.append((str(question_id), row))
    rows.sort(key=lambda item: item[0])
    question_ids = [question_id for question_id, _ in rows]
    if len(question_ids) != len(set(question_ids)):
        raise ValueError(f"duplicate query ID across chunks: {directory}")
    return tuple(rows)


def _validate_row(
    source_name: str,
    question_id: str,
    row: Mapping[str, object],
) -> tuple[str, list[str], int]:
    query = row.get("query")
    text_list = row.get("text_list")
    target_id = row.get("target_id")
    if not isinstance(query, str) or not query.strip():
        raise ValueError(f"missing query: {source_name}/{question_id}")
    if not isinstance(text_list, list) or not text_list:
        raise ValueError(f"missing documents: {source_name}/{question_id}")
    if not all(isinstance(text, str) and text.strip() for text in text_list):
        raise ValueError(f"invalid document: {source_name}/{question_id}")
    if not isinstance(target_id, int) or isinstance(target_id, bool):
        raise ValueError(f"invalid target_id: {source_name}/{question_id}")
    if not 0 <= target_id < len(text_list):
        raise ValueError(f"target_id is outside text_list: {source_name}/{question_id}")
    return query, text_list, target_id


def build_test_source_audit(
    config: E7Config,
    *,
    source_directories: Mapping[str, Path] | None = None,
    expected_test_query_count: int = EXPECTED_TEST_QUERY_COUNT,
) -> dict[str, object]:
    """Describe and hash the complete pre-registered TEST corpus sources."""
    config.validate()
    directories = dict(source_directories or _source_directories(config))
    if tuple(directories) != TEST_POOL_V2_SOURCES:
        raise ValueError("TEST Pool V2 source set or order changed")

    source_records: list[dict[str, object]] = []
    for source_name in TEST_POOL_V2_SOURCES:
        directory = directories[source_name]
        rows = _chunk_rows(directory)
        document_slots = 0
        for question_id, row in rows:
            _, text_list, _ = _validate_row(source_name, question_id, row)
            if source_name == "test" and len(text_list) != 5:
                raise ValueError("every TEST50 query must retain five original slots")
            document_slots += len(text_list)
        source_records.append(
            {
                "source_name": source_name,
                "location": _display_path(directory, config),
                "query_count": len(rows),
                "document_slots": document_slots,
                "files": [
                    {
                        "path": _display_path(path, config),
                        "sha256": sha256_file(path),
                    }
                    for path in _chunk_files(directory)
                ],
            }
        )
    test_record = next(
        record for record in source_records if record["source_name"] == "test"
    )
    if test_record["query_count"] != expected_test_query_count:
        raise ValueError(
            f"TEST source must contain exactly {expected_test_query_count} queries"
        )

    body: dict[str, object] = {
        "schema_version": "e7_test50_pool_v2_source_audit_v1",
        "status": "frozen_before_test_ranking",
        "source_definition": list(TEST_POOL_V2_SOURCES),
        "source_semantics": {
            "researchy_train": "canonical_training_background",
            "rule_pool": "fixed_shared_background_provenance",
            "test": "test50_original_candidate_documents",
        },
        "sources": source_records,
        "physical_identity": "normalized_text_sha256",
        "deduplication": "one_physical_document_preserve_all_provenance",
        "dev20_added_as_special_source": False,
        "forbidden_sources": [
            "generated_rewrites",
            "e0_e1_artifacts",
            "hijack_poison_outputs",
            "rl_artifacts",
            "synthetic_prompts",
            "ge_answers",
            "dev_method_outputs",
        ],
        "eligibility_or_ranking_consulted": False,
        "llm_or_external_api_calls_made": False,
    }
    return {**body, "audit_sha256": canonical_json_sha256(body)}


def _persist_immutable_json(
    destination: Path,
    payload: object,
    config: E7Config,
) -> tuple[str, str]:
    if destination.exists():
        if read_json(destination) != payload:
            raise FileExistsError(f"refusing to replace changed artifact: {destination}")
        if destination.stat().st_mode & 0o222:
            raise PermissionError(f"immutable artifact became writable: {destination}")
        return "reused_identical", sha256_file(destination)
    write_e7_json_once(destination, payload, config)
    destination.chmod(0o444)
    return "created", sha256_file(destination)


def persist_test_source_audit(
    audit: dict[str, object],
    config: E7Config,
) -> dict[str, object]:
    destination = test_pool_v2_paths(config)["audit"]
    action, digest = _persist_immutable_json(destination, audit, config)
    return {
        "stage": "audit_sources",
        "action": action,
        "output_path": str(destination),
        "output_sha256": digest,
        "test_ranking_run": False,
        "llm_or_external_api_calls_made": False,
    }


def _append_document(
    *,
    source_name: str,
    question_id: str,
    text_index: int,
    target_id: int,
    text: str,
    documents: dict[str, _MutableDocument],
) -> str:
    text_hash = normalized_text_sha256(text)
    document_id = f"e7doc_sha256_{text_hash}"
    provenance = DocumentProvenance(
        source_split=source_name,
        source_question_id=question_id,
        source_text_index=text_index,
        is_original_target_id=text_index == target_id,
    )
    existing = documents.get(document_id)
    if existing is None:
        documents[document_id] = _MutableDocument(
            document_id=document_id,
            text_hash=text_hash,
            text=text,
            primary_provenance=provenance,
            duplicate_provenance=[],
        )
    else:
        if normalize_document_text(existing.text) != normalize_document_text(text):
            raise RuntimeError(f"normalized-text SHA256 collision: {document_id}")
        existing.duplicate_provenance.append(provenance)
    return document_id


def _freeze_document(document: _MutableDocument) -> CorpusDocument:
    primary = document.primary_provenance
    return CorpusDocument(
        document_id=document.document_id,
        text_hash=document.text_hash,
        text=document.text,
        identity_method="normalized_text_sha256",
        source_split=primary.source_split,
        source_question_id=primary.source_question_id,
        source_text_index=primary.source_text_index,
        is_original_target_id=primary.is_original_target_id,
        duplicate_provenance=tuple(document.duplicate_provenance),
    )


def _validate_audit(audit: Mapping[str, object]) -> None:
    if audit.get("schema_version") != "e7_test50_pool_v2_source_audit_v1":
        raise ValueError("unsupported TEST Pool V2 source audit")
    body = {key: value for key, value in audit.items() if key != "audit_sha256"}
    if audit.get("audit_sha256") != canonical_json_sha256(body):
        raise ValueError("TEST Pool V2 source audit self-hash mismatch")
    if tuple(audit.get("source_definition", ())) != TEST_POOL_V2_SOURCES:
        raise ValueError("TEST Pool V2 source definition changed")
    if audit.get("eligibility_or_ranking_consulted") is not False:
        raise ValueError("source audit must predate all TEST ranking")


def build_test_pool_v2_manifest(
    config: E7Config,
    audit: Mapping[str, object],
    audit_path: Path,
    *,
    source_directories: Mapping[str, Path] | None = None,
    expected_test_query_count: int = EXPECTED_TEST_QUERY_COUNT,
) -> dict[str, object]:
    """Build a deterministic test corpus without retrieval or model loading."""
    config.validate()
    _validate_audit(audit)
    directories = dict(source_directories or _source_directories(config))
    if tuple(directories) != TEST_POOL_V2_SOURCES:
        raise ValueError("TEST Pool V2 source set or order changed")
    current_audit = build_test_source_audit(
        config,
        source_directories=directories,
        expected_test_query_count=expected_test_query_count,
    )
    if current_audit != audit:
        raise ValueError("TEST source files changed after source audit freeze")

    documents_by_id: dict[str, _MutableDocument] = {}
    query_candidate_sets: list[QueryCandidateSet] = []
    document_slots = 0
    source_files: list[dict[str, str]] = []
    for source_name in TEST_POOL_V2_SOURCES:
        directory = directories[source_name]
        for path in _chunk_files(directory):
            source_files.append(
                {
                    "source_split": source_name,
                    "path": _display_path(path, config),
                    "sha256": sha256_file(path),
                }
            )
        for question_id, row in _chunk_rows(directory):
            query, text_list, target_id = _validate_row(source_name, question_id, row)
            if source_name == "test" and len(text_list) != 5:
                raise ValueError("every TEST50 query must retain five original slots")
            candidate_ids = [
                _append_document(
                    source_name=source_name,
                    question_id=question_id,
                    text_index=text_index,
                    target_id=target_id,
                    text=text,
                    documents=documents_by_id,
                )
                for text_index, text in enumerate(text_list)
            ]
            document_slots += len(candidate_ids)
            if source_name == "test":
                query_candidate_sets.append(
                    QueryCandidateSet(
                        source_split="test",
                        question_id=question_id,
                        query=query,
                        original_candidate_ids=tuple(candidate_ids),
                        original_target_id_document=candidate_ids[target_id],
                        original_target_text_index=target_id,
                    )
                )

    documents = tuple(
        _freeze_document(documents_by_id[document_id])
        for document_id in sorted(documents_by_id)
    )
    queries = tuple(sorted(query_candidate_sets, key=lambda item: item.question_id))
    if len(queries) != expected_test_query_count:
        raise ValueError("TEST query count changed during corpus construction")
    identity: dict[str, object] = {
        "schema_version": "e7_researchy_pooled_corpus_v2",
        "corpus_split": "test",
        "corpus_version": "pool_v2",
        "retrieval_backend": config.retrieval_backend,
        "corpus_name": "researchy_geo_pooled_local_v2",
        "source_splits": list(TEST_POOL_V2_SOURCES),
        "source_files": source_files,
        "source_audit_sha256": audit["audit_sha256"],
        "document_slots": document_slots,
        "documents": [document.to_dict() for document in documents],
        "query_candidate_sets": [query.to_dict() for query in queries],
    }
    payload: dict[str, object] = {
        **identity,
        "corpus_id": f"e7corpus_test_pool_v2_{canonical_json_sha256(identity)}",
        "source_audit_path": _display_path(audit_path, config),
        "deduplicated_document_count": len(documents),
        "duplicate_count": document_slots - len(documents),
        "query_count": len(queries),
        "dev20_added_as_special_source": False,
        "selection_or_eligibility_used": False,
        "synthetic_documents_used": False,
        "llm_or_external_api_calls_made": False,
    }
    validate_test_pool_v2_manifest(
        payload,
        expected_test_query_count=expected_test_query_count,
    )
    return payload


def validate_test_pool_v2_manifest(
    manifest: Mapping[str, object],
    *,
    expected_test_query_count: int = EXPECTED_TEST_QUERY_COUNT,
) -> None:
    if manifest.get("schema_version") != "e7_researchy_pooled_corpus_v2":
        raise ValueError("invalid TEST Pool V2 schema")
    if manifest.get("corpus_split") != "test":
        raise ValueError("TEST Pool V2 manifest has the wrong split")
    if tuple(manifest.get("source_splits", ())) != TEST_POOL_V2_SOURCES:
        raise ValueError("TEST Pool V2 source composition changed")
    if "dev" in manifest.get("source_splits", ()):
        raise ValueError("DEV20 cannot be an extra TEST Pool V2 source")
    documents = manifest.get("documents")
    queries = manifest.get("query_candidate_sets")
    if not isinstance(documents, list) or not documents:
        raise ValueError("TEST Pool V2 has no physical documents")
    if not isinstance(queries, list) or len(queries) != expected_test_query_count:
        raise ValueError("TEST Pool V2 query provenance count changed")
    if manifest.get("deduplicated_document_count") != len(documents):
        raise ValueError("TEST Pool V2 physical document count mismatch")
    provenance_slots = sum(
        1 + len(document.get("duplicate_provenance", ())) for document in documents
    )
    if manifest.get("document_slots") != provenance_slots:
        raise ValueError("TEST Pool V2 provenance was not fully preserved")
    if manifest.get("duplicate_count") != provenance_slots - len(documents):
        raise ValueError("TEST Pool V2 duplicate count mismatch")
    ids: set[str] = set()
    for document in documents:
        if not isinstance(document, dict):
            raise ValueError("TEST Pool V2 document must be an object")
        document_id = document.get("document_id")
        text = document.get("text")
        text_hash = document.get("text_hash")
        if not all(isinstance(value, str) and value for value in (document_id, text, text_hash)):
            raise ValueError("TEST Pool V2 document lacks identity")
        expected_hash = normalized_text_sha256(text)
        if text_hash != expected_hash or document_id != f"e7doc_sha256_{expected_hash}":
            raise ValueError("TEST Pool V2 document identity mismatch")
        if document_id in ids:
            raise ValueError("TEST Pool V2 contains duplicate physical documents")
        ids.add(document_id)
        provenance = [
            {"source_split": document.get("source_split")},
            *document.get("duplicate_provenance", ()),
        ]
        if any(item.get("source_split") not in TEST_POOL_V2_SOURCES for item in provenance):
            raise ValueError("TEST Pool V2 provenance escapes source contract")
    for query in queries:
        candidate_ids = query.get("original_candidate_ids")
        target_index = query.get("original_target_text_index")
        if query.get("source_split") != "test":
            raise ValueError("only TEST50 rows may become evaluation queries")
        if not isinstance(candidate_ids, list) or len(candidate_ids) != 5:
            raise ValueError("TEST50 query did not preserve five source slots")
        if not isinstance(target_index, int) or not 0 <= target_index < 5:
            raise ValueError("TEST50 original target_id is invalid")
        if any(document_id not in ids for document_id in candidate_ids):
            raise ValueError("TEST50 original candidate is absent from corpus")
        if query.get("original_target_id_document") != candidate_ids[target_index]:
            raise ValueError("TEST50 original target provenance changed")


def persist_test_pool_v2_manifest(
    manifest: dict[str, object],
    config: E7Config,
) -> dict[str, object]:
    destination = test_pool_v2_paths(config)["manifest"]
    action, digest = _persist_immutable_json(destination, manifest, config)
    return {
        "stage": "build",
        "action": action,
        "output_path": str(destination),
        "output_sha256": digest,
        "corpus_id": manifest["corpus_id"],
        "document_slots": manifest["document_slots"],
        "deduplicated_document_count": manifest["deduplicated_document_count"],
        "duplicate_count": manifest["duplicate_count"],
        "query_count": manifest["query_count"],
        "test_ranking_run": False,
        "llm_or_external_api_calls_made": False,
    }


def run_audit_sources(config: E7Config | None = None) -> dict[str, object]:
    config = config or E7Config()
    audit = build_test_source_audit(config)
    return persist_test_source_audit(audit, config)


def run_build(config: E7Config | None = None) -> dict[str, object]:
    config = config or E7Config()
    paths = test_pool_v2_paths(config)
    if not paths["audit"].is_file():
        raise FileNotFoundError("freeze TEST Pool V2 source audit before build")
    audit = read_json(paths["audit"])
    manifest = build_test_pool_v2_manifest(config, audit, paths["audit"])
    return persist_test_pool_v2_manifest(manifest, config)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", required=True, choices=("audit_sources", "build"))
    args = parser.parse_args()
    result = run_audit_sources() if args.stage == "audit_sources" else run_build()
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
