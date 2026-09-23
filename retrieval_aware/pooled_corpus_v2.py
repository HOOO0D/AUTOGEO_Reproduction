"""Build the one-shot, split-isolated Researchy pooled DEV corpus V2."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

from .config import E7Config
from .corpus_builder import (
    CorpusDocument,
    DocumentProvenance,
    QueryCandidateSet,
    SourceFile,
)
from .document_source_audit import (
    _chunk_rows,
    build_document_source_audit,
    persist_document_source_audit,
)
from .utils import (
    canonical_json_sha256,
    normalize_document_text,
    normalized_text_sha256,
    read_json,
    sha256_file,
    write_e7_json_once,
)


@dataclass
class _MutableDocument:
    document_id: str
    text_hash: str
    text: str
    primary_provenance: DocumentProvenance
    duplicate_provenance: list[DocumentProvenance]


def _relative(path: Path, config: E7Config) -> str:
    return str(path.resolve().relative_to(config.project_root.resolve()))


def _source_files(config: E7Config) -> tuple[SourceFile, ...]:
    paths = sorted(
        (config.data_root / "Researchy-GEO" / "train").glob("datachunk_*.json")
    )
    paths += [
        config.rule_pool_query_dir / "datachunk_0.json",
        config.dev_query_dir / "datachunk_0.json",
    ]
    source_by_path = {
        path.resolve(): "researchy_train" for path in paths[:-2]
    }
    source_by_path[paths[-2].resolve()] = "rule_pool"
    source_by_path[paths[-1].resolve()] = "dev"
    return tuple(
        SourceFile(
            source_split=source_by_path[path.resolve()],
            path=_relative(path, config),
            sha256=sha256_file(path),
        )
        for path in paths
    )


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
            raise RuntimeError(f"normalized-text SHA-256 collision: {document_id}")
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


def _test50_hashes(config: E7Config) -> set[str]:
    return {
        normalized_text_sha256(text)
        for _, row in _chunk_rows(config.test_query_dir)
        for text in row["text_list"]
    }


def build_pool_v2_manifest(
    config: E7Config,
    audit: dict[str, object],
    audit_path: Path,
) -> dict[str, object]:
    config.validate()
    expected_sources = [
        "researchy_geo_train_original",
        "researchy_rulepool100_v1",
        "researchy_dev20_v1",
    ]
    if audit["selection_contract"]["eligible_source_set"] != expected_sources:
        raise ValueError("source audit does not match the frozen Pool V2 contract")

    source_directories = (
        ("researchy_train", config.data_root / "Researchy-GEO" / "train"),
        ("rule_pool", config.rule_pool_query_dir),
        ("dev", config.dev_query_dir),
    )
    forbidden_hashes = _test50_hashes(config)
    documents_by_id: dict[str, _MutableDocument] = {}
    query_candidate_sets: list[QueryCandidateSet] = []
    raw_document_slots = 0
    excluded_test50_overlap_slots = 0
    included_document_slots = 0

    for source_name, directory in source_directories:
        for question_id, row in _chunk_rows(directory):
            query = row.get("query")
            text_list = row.get("text_list")
            target_id = row.get("target_id")
            if not isinstance(query, str) or not query.strip():
                raise ValueError(f"missing query: {source_name}/{question_id}")
            if not isinstance(text_list, list) or not text_list:
                raise ValueError(f"missing documents: {source_name}/{question_id}")
            if not isinstance(target_id, int) or not 0 <= target_id < len(text_list):
                raise ValueError(f"invalid target_id: {source_name}/{question_id}")

            candidate_ids: list[str] = []
            for text_index, text in enumerate(text_list):
                if not isinstance(text, str) or not text.strip():
                    raise ValueError(
                        f"empty document: {source_name}/{question_id}/{text_index}"
                    )
                raw_document_slots += 1
                text_hash = normalized_text_sha256(text)
                if text_hash in forbidden_hashes:
                    if source_name != "researchy_train":
                        raise ValueError(
                            "fixed RULE_POOL100/DEV20 overlaps held-out TEST50"
                        )
                    excluded_test50_overlap_slots += 1
                    continue
                candidate_ids.append(
                    _append_document(
                        source_name=source_name,
                        question_id=question_id,
                        text_index=text_index,
                        target_id=target_id,
                        text=text,
                        documents=documents_by_id,
                    )
                )
                included_document_slots += 1

            if source_name == "dev":
                if len(candidate_ids) != 5:
                    raise ValueError("DEV candidates cannot be removed by isolation")
                query_candidate_sets.append(
                    QueryCandidateSet(
                        source_split="dev",
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
    source_files = _source_files(config)
    identity_payload = {
        "schema_version": "e7_researchy_pooled_corpus_v2",
        "corpus_split": "dev",
        "corpus_version": "pool_v2",
        "retrieval_backend": config.retrieval_backend,
        "corpus_name": "researchy_geo_pooled_local_v2",
        "source_splits": list(config.pool_v2_corpus_sources),
        "source_files": [asdict(source) for source in source_files],
        "source_audit_hash": audit["audit_hash"],
        "raw_document_slots": raw_document_slots,
        "excluded_test50_overlap_slots": excluded_test50_overlap_slots,
        "document_slots": included_document_slots,
        "documents": [document.to_dict() for document in documents],
        "query_candidate_sets": [query.to_dict() for query in queries],
    }
    payload = {
        **identity_payload,
        "corpus_id": f"e7corpus_dev_pool_v2_{canonical_json_sha256(identity_payload)}",
        "source_audit_path": _relative(audit_path, config),
        "deduplicated_document_count": len(documents),
        "duplicate_count": included_document_slots - len(documents),
        "excluded_test50_physical_hash_count": len(forbidden_hashes & {
            normalized_text_sha256(text)
            for _, row in _chunk_rows(config.data_root / "Researchy-GEO" / "train")
            for text in row["text_list"]
        }),
        "query_count": len(queries),
        "selection_or_eligibility_used": False,
        "synthetic_documents_used": False,
        "test50_source_provenance_used": False,
    }
    validate_pool_v2_manifest(payload, config, forbidden_hashes)
    return payload


def validate_pool_v2_manifest(
    manifest: dict[str, object],
    config: E7Config,
    forbidden_hashes: set[str],
) -> None:
    if manifest["schema_version"] != "e7_researchy_pooled_corpus_v2":
        raise ValueError("invalid Pool V2 schema")
    if tuple(manifest["source_splits"]) != config.pool_v2_corpus_sources:
        raise ValueError("Pool V2 source composition changed")
    if manifest["query_count"] != 20:
        raise ValueError("Pool V2 must retain exactly DEV20 queries")
    documents = manifest["documents"]
    if manifest["deduplicated_document_count"] != len(documents):
        raise ValueError("Pool V2 physical document count mismatch")
    if manifest["document_slots"] != sum(
        1 + len(document["duplicate_provenance"])
        for document in documents
    ):
        raise ValueError("Pool V2 provenance count mismatch")
    if manifest["duplicate_count"] != (
        manifest["document_slots"] - manifest["deduplicated_document_count"]
    ):
        raise ValueError("Pool V2 duplicate count mismatch")
    ids = {document["document_id"] for document in documents}
    hashes = {document["text_hash"] for document in documents}
    if len(ids) != len(documents) or hashes & forbidden_hashes:
        raise ValueError("Pool V2 contains duplicate IDs or held-out TEST50 content")
    for document in documents:
        provenances = [
            {"source_split": document["source_split"]},
            *document["duplicate_provenance"],
        ]
        if any(item["source_split"] not in config.pool_v2_corpus_sources for item in provenances):
            raise ValueError("Pool V2 provenance escapes the frozen source set")
    for query in manifest["query_candidate_sets"]:
        if query["source_split"] != "dev" or len(query["original_candidate_ids"]) != 5:
            raise ValueError("Pool V2 changed DEV candidate semantics")
        if any(document_id not in ids for document_id in query["original_candidate_ids"]):
            raise ValueError("Pool V2 DEV candidate is absent from corpus")


def persist_pool_v2_manifest(
    manifest: dict[str, object],
    config: E7Config,
) -> tuple[Path, str]:
    destination = (
        config.output_paths["corpus"] / "pool_v2" / "dev_corpus_manifest.json"
    )
    if destination.exists():
        if read_json(destination) != manifest:
            raise FileExistsError(f"refusing to replace changed Pool V2: {destination}")
        return destination, "reused_identical"
    write_e7_json_once(destination, manifest, config)
    destination.chmod(0o444)
    return destination, "created"


def build_and_persist_pool_v2(config: E7Config | None = None) -> dict[str, object]:
    config = config or E7Config()
    audit = build_document_source_audit(config)
    audit_path, audit_action = persist_document_source_audit(audit, config)
    manifest = build_pool_v2_manifest(config, audit, audit_path)
    manifest_path, manifest_action = persist_pool_v2_manifest(manifest, config)
    summary = {
        "schema_version": "e7_pool_v2_build_summary_v1",
        "source_audit_path": str(audit_path),
        "manifest_path": str(manifest_path),
        "manifest_sha256": sha256_file(manifest_path),
        "corpus_id": manifest["corpus_id"],
        "raw_document_slots": manifest["raw_document_slots"],
        "excluded_test50_overlap_slots": manifest["excluded_test50_overlap_slots"],
        "document_slots": manifest["document_slots"],
        "deduplicated_document_count": manifest["deduplicated_document_count"],
        "duplicate_count": manifest["duplicate_count"],
        "query_count": manifest["query_count"],
        "retrieval_or_eligibility_run": False,
        "embedding_model_loaded": False,
        "llm_or_ge_calls_made": False,
    }
    summary_path = config.output_paths["corpus"] / "pool_v2" / "build_summary.json"
    if summary_path.exists():
        if read_json(summary_path) != summary:
            raise FileExistsError(f"refusing to replace Pool V2 summary: {summary_path}")
    else:
        write_e7_json_once(summary_path, summary, config)
        summary_path.chmod(0o444)
    return {
        "source_audit_action": audit_action,
        "manifest_action": manifest_action,
        **summary,
    }


def main() -> None:
    print(json.dumps(build_and_persist_pool_v2(), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
