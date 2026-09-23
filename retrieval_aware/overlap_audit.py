"""Optional ClueWeb overlap diagnostic; unused by pooled-local E7."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path

from .config import E7Config
from .pool_builder import RetrievalDocument, RetrievalPool
from .target_selector import normalize_url
from .utils import normalized_text_sha256, write_e7_json_once


@dataclass(frozen=True)
class OriginalDocumentIdentity:
    original_index: int
    text: str | None = None
    clueweb22_id: str | None = None
    clueweb_url_hash: str | None = None
    url: str | None = None


@dataclass(frozen=True)
class OriginalDocumentOverlap:
    original_index: int
    matched: bool
    matched_global_rank: int | None
    matched_document_id: str | None
    identity_method: str | None


@dataclass(frozen=True)
class OriginalTop5OverlapAudit:
    schema_version: str
    e7_query_id: str
    pool_id: str
    original_document_count: int
    matched_document_count: int
    original5_recall_at_100: float
    matches: tuple[OriginalDocumentOverlap, ...]
    diagnostic_only: bool = True

    def to_dict(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "e7_query_id": self.e7_query_id,
            "pool_id": self.pool_id,
            "original_document_count": self.original_document_count,
            "matched_document_count": self.matched_document_count,
            "original5_recall_at_100": self.original5_recall_at_100,
            "matches": [asdict(match) for match in self.matches],
            "diagnostic_only": self.diagnostic_only,
        }


def _find_match(
    original: OriginalDocumentIdentity,
    pool_documents: tuple[RetrievalDocument, ...],
    used_document_ids: set[str],
) -> tuple[RetrievalDocument, str] | None:
    available = [
        document
        for document in pool_documents
        if document.document_id not in used_document_ids
    ]
    if original.clueweb22_id:
        for document in available:
            if document.document_id.casefold() == original.clueweb22_id.casefold():
                return document, "clueweb22_id"
    if original.clueweb_url_hash:
        for document in available:
            if (
                document.clueweb_url_hash.casefold()
                == original.clueweb_url_hash.casefold()
            ):
                return document, "clueweb_url_hash"
    if original.url:
        normalized_original_url = normalize_url(original.url)
        for document in available:
            if normalize_url(document.url) == normalized_original_url:
                return document, "normalized_url"
    if original.text:
        original_hash = normalized_text_sha256(original.text)
        for document in available:
            if normalized_text_sha256(document.text) == original_hash:
                return document, "normalized_text_sha256_fallback"
    return None


def audit_original_top5_overlap(
    e7_query_id: str,
    original_documents: tuple[OriginalDocumentIdentity, ...],
    pool: RetrievalPool,
) -> OriginalTop5OverlapAudit:
    """Measure overlap only; never make target eligibility depend on it."""
    if len(original_documents) != 5:
        raise ValueError("Researchy-GEO original context must contain five documents")
    matches: list[OriginalDocumentOverlap] = []
    used_document_ids: set[str] = set()
    for original in original_documents:
        match = _find_match(original, pool.documents, used_document_ids)
        if match is None:
            matches.append(
                OriginalDocumentOverlap(
                    original_index=original.original_index,
                    matched=False,
                    matched_global_rank=None,
                    matched_document_id=None,
                    identity_method=None,
                )
            )
        else:
            document, method = match
            used_document_ids.add(document.document_id)
            matches.append(
                OriginalDocumentOverlap(
                    original_index=original.original_index,
                    matched=True,
                    matched_global_rank=document.rank,
                    matched_document_id=document.document_id,
                    identity_method=method,
                )
            )
    matched_count = sum(match.matched for match in matches)
    return OriginalTop5OverlapAudit(
        schema_version="e7_original_top5_overlap_v1",
        e7_query_id=e7_query_id,
        pool_id=pool.pool_id,
        original_document_count=5,
        matched_document_count=matched_count,
        original5_recall_at_100=matched_count / 5,
        matches=tuple(matches),
    )


def persist_overlap_audit(
    audit: OriginalTop5OverlapAudit,
    config: E7Config,
) -> Path:
    destination = config.output_paths["audits"] / f"{audit.e7_query_id}.json"
    write_e7_json_once(destination, audit.to_dict(), config)
    return destination
