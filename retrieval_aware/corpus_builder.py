"""Build deterministic Researchy-GEO pooled local corpus manifests.

This module only reads frozen Researchy-GEO chunks.  It does not retrieve,
embed, rewrite, or evaluate documents.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from enum import Enum
from pathlib import Path
from typing import Sequence

from .config import E7Config
from .query_mapping import AutoGEOQuestion, load_autogeo_questions
from .utils import (
    canonical_json_sha256,
    ensure_output_layout,
    normalize_document_text,
    normalized_text_sha256,
    read_json,
    sha256_file,
    write_e7_json_once,
)


class CorpusSplit(str, Enum):
    DEV = "dev"
    TEST = "test"


@dataclass(frozen=True)
class DocumentProvenance:
    source_split: str
    source_question_id: str
    source_text_index: int
    is_original_target_id: bool


@dataclass(frozen=True)
class CorpusDocument:
    document_id: str
    text_hash: str
    text: str
    identity_method: str
    source_split: str
    source_question_id: str
    source_text_index: int
    is_original_target_id: bool
    duplicate_provenance: tuple[DocumentProvenance, ...]

    def to_dict(self) -> dict[str, object]:
        result = asdict(self)
        result["duplicate_provenance"] = [
            asdict(provenance) for provenance in self.duplicate_provenance
        ]
        return result


@dataclass(frozen=True)
class QueryCandidateSet:
    source_split: str
    question_id: str
    query: str
    original_candidate_ids: tuple[str, ...]
    original_target_id_document: str
    original_target_text_index: int

    def to_dict(self) -> dict[str, object]:
        result = asdict(self)
        result["original_candidate_ids"] = list(self.original_candidate_ids)
        return result


@dataclass(frozen=True)
class SourceFile:
    source_split: str
    path: str
    sha256: str


@dataclass(frozen=True)
class CorpusManifest:
    schema_version: str
    corpus_id: str
    corpus_split: CorpusSplit
    retrieval_backend: str
    corpus_name: str
    source_splits: tuple[str, ...]
    source_files: tuple[SourceFile, ...]
    document_slots: int
    deduplicated_document_count: int
    duplicate_count: int
    query_count: int
    documents: tuple[CorpusDocument, ...]
    query_candidate_sets: tuple[QueryCandidateSet, ...]

    def to_dict(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "corpus_id": self.corpus_id,
            "corpus_split": self.corpus_split.value,
            "retrieval_backend": self.retrieval_backend,
            "corpus_name": self.corpus_name,
            "source_splits": list(self.source_splits),
            "source_files": [asdict(source) for source in self.source_files],
            "document_slots": self.document_slots,
            "deduplicated_document_count": self.deduplicated_document_count,
            "duplicate_count": self.duplicate_count,
            "query_count": self.query_count,
            "documents": [document.to_dict() for document in self.documents],
            "query_candidate_sets": [
                query.to_dict() for query in self.query_candidate_sets
            ],
        }


@dataclass
class _MutableDocument:
    document_id: str
    text_hash: str
    text: str
    primary_provenance: DocumentProvenance
    duplicate_provenance: list[DocumentProvenance]


def _source_directories(config: E7Config) -> dict[str, Path]:
    return {
        "rule_pool": config.rule_pool_query_dir,
        "dev": config.dev_query_dir,
        "test": config.test_query_dir,
    }


def _sources_for_split(split: CorpusSplit, config: E7Config) -> tuple[str, ...]:
    return (
        config.dev_corpus_sources
        if split is CorpusSplit.DEV
        else config.test_corpus_sources
    )


def _assert_source_is_frozen(source_dir: Path, config: E7Config) -> None:
    resolved = source_dir.resolve()
    frozen_root = config.frozen_root.resolve()
    if resolved != frozen_root and frozen_root not in resolved.parents:
        raise ValueError(
            f"corpus source must be inside experiments/frozen/: {resolved}"
        )


def _source_files(
    source_name: str,
    source_dir: Path,
    config: E7Config,
) -> tuple[SourceFile, ...]:
    files = sorted(source_dir.glob("datachunk_*.json"))
    if not files:
        raise FileNotFoundError(f"no frozen chunks found in {source_dir}")
    records: list[SourceFile] = []
    for path in files:
        try:
            display_path = str(
                path.resolve().relative_to(config.project_root.resolve())
            )
        except ValueError:
            display_path = str(path.resolve())
        records.append(
            SourceFile(
                source_split=source_name,
                path=display_path,
                sha256=sha256_file(path),
            )
        )
    return tuple(records)


def _document_identity(text: str) -> tuple[str, str]:
    text_hash = normalized_text_sha256(text)
    return f"e7doc_sha256_{text_hash}", text_hash


def _append_question_documents(
    source_name: str,
    question: AutoGEOQuestion,
    documents_by_id: dict[str, _MutableDocument],
) -> tuple[str, ...]:
    candidate_ids: list[str] = []
    for text_index, text in enumerate(question.text_list):
        if not text.strip():
            raise ValueError(
                f"empty document at {source_name}/{question.autogeo_question_id}/"
                f"{text_index}"
            )
        document_id, text_hash = _document_identity(text)
        provenance = DocumentProvenance(
            source_split=source_name,
            source_question_id=question.autogeo_question_id,
            source_text_index=text_index,
            is_original_target_id=text_index == question.target_id,
        )
        existing = documents_by_id.get(document_id)
        if existing is None:
            documents_by_id[document_id] = _MutableDocument(
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
        candidate_ids.append(document_id)
    return tuple(candidate_ids)


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


def _manifest_identity_payload(
    split: CorpusSplit,
    config: E7Config,
    source_names: tuple[str, ...],
    source_files: Sequence[SourceFile],
    document_slots: int,
    documents: Sequence[CorpusDocument],
    queries: Sequence[QueryCandidateSet],
) -> dict[str, object]:
    return {
        "schema_version": "e7_researchy_pooled_corpus_v1",
        "corpus_split": split.value,
        "retrieval_backend": config.retrieval_backend,
        "corpus_name": config.retrieval_corpus,
        "source_splits": list(source_names),
        "source_files": [asdict(source) for source in source_files],
        "document_slots": document_slots,
        "documents": [document.to_dict() for document in documents],
        "query_candidate_sets": [query.to_dict() for query in queries],
    }


def build_corpus_manifest(
    split: CorpusSplit,
    config: E7Config,
) -> CorpusManifest:
    """Build one split-isolated corpus entirely from frozen local chunks."""
    config.validate()
    if config.retrieval_backend != "researchy_pooled_local":
        raise ValueError("local corpus manifests require researchy_pooled_local")

    source_names = _sources_for_split(split, config)
    evaluation_source = split.value
    source_directories = _source_directories(config)
    documents_by_id: dict[str, _MutableDocument] = {}
    source_file_records: list[SourceFile] = []
    query_candidate_sets: list[QueryCandidateSet] = []
    document_slots = 0

    for source_name in source_names:
        source_dir = source_directories[source_name]
        _assert_source_is_frozen(source_dir, config)
        source_file_records.extend(_source_files(source_name, source_dir, config))
        questions = load_autogeo_questions(source_dir)
        for question in questions:
            candidate_ids = _append_question_documents(
                source_name,
                question,
                documents_by_id,
            )
            document_slots += len(candidate_ids)
            if source_name == evaluation_source:
                query_candidate_sets.append(
                    QueryCandidateSet(
                        source_split=source_name,
                        question_id=question.autogeo_question_id,
                        query=question.query,
                        original_candidate_ids=candidate_ids,
                        original_target_id_document=candidate_ids[question.target_id],
                        original_target_text_index=question.target_id,
                    )
                )

    documents = tuple(
        _freeze_document(documents_by_id[document_id])
        for document_id in sorted(documents_by_id)
    )
    queries = tuple(
        sorted(query_candidate_sets, key=lambda item: item.question_id)
    )
    identity_payload = _manifest_identity_payload(
        split,
        config,
        source_names,
        source_file_records,
        document_slots,
        documents,
        queries,
    )
    manifest = CorpusManifest(
        schema_version="e7_researchy_pooled_corpus_v1",
        corpus_id=f"e7corpus_{split.value}_{canonical_json_sha256(identity_payload)}",
        corpus_split=split,
        retrieval_backend=config.retrieval_backend,
        corpus_name=config.retrieval_corpus,
        source_splits=source_names,
        source_files=tuple(source_file_records),
        document_slots=document_slots,
        deduplicated_document_count=len(documents),
        duplicate_count=document_slots - len(documents),
        query_count=len(queries),
        documents=documents,
        query_candidate_sets=queries,
    )
    validate_corpus_manifest(manifest, config)
    return manifest


def validate_corpus_manifest(manifest: CorpusManifest, config: E7Config) -> None:
    expected_sources = _sources_for_split(manifest.corpus_split, config)
    if manifest.source_splits != expected_sources:
        raise ValueError("corpus manifest violates configured split isolation")
    forbidden_source = "test" if manifest.corpus_split is CorpusSplit.DEV else "dev"
    if forbidden_source in manifest.source_splits:
        raise ValueError(
            f"{manifest.corpus_split.value} corpus includes {forbidden_source}"
        )
    if manifest.document_slots != sum(
        1 + len(document.duplicate_provenance) for document in manifest.documents
    ):
        raise ValueError("document slots do not equal preserved provenance count")
    if manifest.deduplicated_document_count != len(manifest.documents):
        raise ValueError("deduplicated document count is inconsistent")
    if manifest.duplicate_count != (
        manifest.document_slots - manifest.deduplicated_document_count
    ):
        raise ValueError("duplicate count is inconsistent")
    document_ids = {document.document_id for document in manifest.documents}
    if len(document_ids) != len(manifest.documents):
        raise ValueError("physical corpus contains duplicate document IDs")
    for document in manifest.documents:
        expected_hash = normalized_text_sha256(document.text)
        if document.text_hash != expected_hash:
            raise ValueError(f"text hash mismatch for {document.document_id}")
        if document.document_id != f"e7doc_sha256_{expected_hash}":
            raise ValueError(f"document identity mismatch for {document.document_id}")
        provenance = (
            DocumentProvenance(
                source_split=document.source_split,
                source_question_id=document.source_question_id,
                source_text_index=document.source_text_index,
                is_original_target_id=document.is_original_target_id,
            ),
            *document.duplicate_provenance,
        )
        if any(item.source_split not in manifest.source_splits for item in provenance):
            raise ValueError("document provenance escapes corpus split sources")
    if manifest.query_count != len(manifest.query_candidate_sets):
        raise ValueError("query count is inconsistent")
    for query in manifest.query_candidate_sets:
        if query.source_split != manifest.corpus_split.value:
            raise ValueError(
                "shared rule-pool questions cannot become evaluation queries"
            )
        if len(query.original_candidate_ids) != 5:
            raise ValueError("Researchy-GEO query must retain five original candidates")
        if any(
            candidate not in document_ids
            for candidate in query.original_candidate_ids
        ):
            raise ValueError("query candidate identity is absent from physical corpus")
        if query.original_target_id_document not in query.original_candidate_ids:
            raise ValueError("original target identity must be one of five candidates")
        if (
            query.original_candidate_ids[query.original_target_text_index]
            != query.original_target_id_document
        ):
            raise ValueError("original target index and document identity disagree")


def persist_corpus_manifest(
    manifest: CorpusManifest,
    config: E7Config,
) -> tuple[Path, str]:
    """Write once; permit later runs only when the manifest is byte-equivalent."""
    destination = (
        config.output_paths["corpus"]
        / f"{manifest.corpus_split.value}_corpus_manifest.json"
    )
    payload = manifest.to_dict()
    if destination.exists():
        if read_json(destination) != payload:
            raise FileExistsError(
                f"refusing to replace changed E7 corpus manifest: {destination}"
            )
        return destination, "reused_identical"
    write_e7_json_once(destination, payload, config)
    return destination, "created"


def build_and_persist_all(config: E7Config) -> dict[str, object]:
    ensure_output_layout(config)
    summaries: dict[str, object] = {}
    for split in (CorpusSplit.DEV, CorpusSplit.TEST):
        manifest = build_corpus_manifest(split, config)
        path, action = persist_corpus_manifest(manifest, config)
        summaries[split.value] = {
            "path": str(path),
            "action": action,
            "corpus_id": manifest.corpus_id,
            "document_slots": manifest.document_slots,
            "deduplicated_document_count": manifest.deduplicated_document_count,
            "duplicate_count": manifest.duplicate_count,
            "query_count": manifest.query_count,
            "source_splits": list(manifest.source_splits),
        }
    return {
        "schema_version": "e7_local_corpus_build_summary_v1",
        "retrieval_backend": config.retrieval_backend,
        "corpora": summaries,
        "external_api_calls_made": False,
        "embedding_model_loaded": False,
        "llm_or_ge_calls_made": False,
    }


def _build_parser() -> argparse.ArgumentParser:
    return argparse.ArgumentParser(
        description="Build split-isolated E7 Researchy-GEO local corpus manifests."
    )


def main() -> None:
    _build_parser().parse_args()
    summary = build_and_persist_all(E7Config())
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
