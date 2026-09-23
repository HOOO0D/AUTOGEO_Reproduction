"""Optional ClueWeb-backend clicked-target selector.

The formal pooled-local protocol instead uses the frozen Researchy-GEO
``target_id`` document and makes no click/qrel relevance claim.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from enum import Enum
from pathlib import Path
from urllib.parse import parse_qsl, urlencode, urlsplit, urlunsplit

from .config import E7Config
from .pool_builder import RetrievalDocument, RetrievalPool
from .query_mapping import ClickedDocument, ResearchyQuestionRow
from .utils import write_e7_json_once


class TargetSelectionStatus(str, Enum):
    ELIGIBLE = "eligible"
    INELIGIBLE = "ineligible"


class RankBucket(str, Enum):
    EASY = "easy"
    MEDIUM = "medium"
    HARD = "hard"


@dataclass(frozen=True)
class TargetMetadata:
    query_id: str
    document_id: str
    original_rank: int
    original_retrieval_score: float | None
    relevance_source: str
    relevance_identifier: str
    selection_band: str
    clueweb_url_hash: str | None = None
    url: str | None = None
    click_count: float | None = None
    rank_bucket: str | None = None
    selection_reason: str | None = None
    identity_match_method: str | None = None


@dataclass(frozen=True)
class TargetSelectionResult:
    schema_version: str
    e7_query_id: str
    pool_id: str
    status: TargetSelectionStatus
    target_document_id: str | None
    clueweb_url_hash: str | None
    url: str | None
    click_count: float | None
    global_rank: int | None
    rank_bucket: RankBucket | None
    selection_reason: str
    identity_match_method: str | None

    def to_dict(self) -> dict[str, object]:
        result = asdict(self)
        result["status"] = self.status.value
        result["rank_bucket"] = (
            self.rank_bucket.value if self.rank_bucket is not None else None
        )
        result["CluewebURLHash"] = result.pop("clueweb_url_hash")
        result["URL"] = result.pop("url")
        result["Click_Cnt"] = result.pop("click_count")
        return result

    def as_target_metadata(self, pool: RetrievalPool) -> TargetMetadata:
        if self.status is not TargetSelectionStatus.ELIGIBLE:
            raise ValueError("ineligible selection has no target metadata")
        target = next(
            document
            for document in pool.documents
            if document.document_id == self.target_document_id
        )
        return TargetMetadata(
            query_id=self.e7_query_id,
            document_id=target.document_id,
            original_rank=target.rank,
            original_retrieval_score=target.score,
            relevance_source="researchy_questions_docstream",
            relevance_identifier=self.clueweb_url_hash or self.url or "",
            selection_band=(
                "preferred"
                if self.selection_reason == "max_click_count_within_rank_6_20"
                else "fallback"
            ),
            clueweb_url_hash=self.clueweb_url_hash,
            url=self.url,
            click_count=self.click_count,
            rank_bucket=self.rank_bucket.value if self.rank_bucket else None,
            selection_reason=self.selection_reason,
            identity_match_method=self.identity_match_method,
        )


@dataclass(frozen=True)
class _ClickedPoolMatch:
    pool_document: RetrievalDocument
    clicked_document: ClickedDocument
    identity_method: str


def normalize_url(url: str) -> str:
    """Normalize stable URL identity without fetching or resolving redirects."""
    parsed = urlsplit(url.strip())
    scheme = parsed.scheme.casefold()
    hostname = (parsed.hostname or "").casefold()
    port = parsed.port
    if port is not None and not (
        (scheme == "http" and port == 80) or (scheme == "https" and port == 443)
    ):
        hostname = f"{hostname}:{port}"
    path = parsed.path or "/"
    if path != "/":
        path = path.rstrip("/")
    query = urlencode(sorted(parse_qsl(parsed.query, keep_blank_values=True)))
    return urlunsplit((scheme, hostname, path, query, ""))


def _match_clicked_document(
    pool_document: RetrievalDocument,
    clicked_documents: tuple[ClickedDocument, ...],
) -> tuple[ClickedDocument, str] | None:
    pool_id = pool_document.document_id.casefold()
    pool_hash = pool_document.clueweb_url_hash.casefold()
    pool_url = normalize_url(pool_document.url)

    for clicked in clicked_documents:
        if clicked.clueweb22_id and clicked.clueweb22_id.casefold() == pool_id:
            return clicked, "clueweb22_id"
    for clicked in clicked_documents:
        if clicked.clueweb_url_hash.casefold() == pool_hash:
            return clicked, "clueweb_url_hash"
    for clicked in clicked_documents:
        if normalize_url(clicked.url) == pool_url:
            return clicked, "normalized_url"
    return None


def classify_target_rank(rank: int, config: E7Config) -> str:
    if config.preferred_target_rank_min <= rank <= config.preferred_target_rank_max:
        return "preferred"
    if config.ge_top_k < rank <= config.max_target_rank:
        return "fallback"
    return "ineligible"


def rank_bucket(rank: int, config: E7Config) -> RankBucket:
    if config.preferred_target_rank_min <= rank <= config.preferred_target_rank_max:
        return RankBucket.EASY
    if config.preferred_target_rank_max < rank <= config.medium_target_rank_max:
        return RankBucket.MEDIUM
    if config.medium_target_rank_max < rank <= config.max_target_rank:
        return RankBucket.HARD
    raise ValueError(f"rank {rank} is not eligible for an E7 target")


def select_target(
    pool: RetrievalPool,
    researchy_row: ResearchyQuestionRow,
    config: E7Config,
) -> TargetSelectionResult:
    """Select max-Click_Cnt clicked D* from 6--20, else from 21--100."""
    if pool.researchy_question_id != researchy_row.researchy_question_id:
        raise ValueError("pool and Researchy Questions row do not match")

    matches: list[_ClickedPoolMatch] = []
    for pool_document in pool.documents:
        if not config.preferred_target_rank_min <= pool_document.rank <= config.max_target_rank:
            continue
        match = _match_clicked_document(
            pool_document,
            researchy_row.clicked_documents,
        )
        if match is not None:
            clicked, identity_method = match
            matches.append(
                _ClickedPoolMatch(
                    pool_document=pool_document,
                    clicked_document=clicked,
                    identity_method=identity_method,
                )
            )

    preferred = [
        match
        for match in matches
        if match.pool_document.rank <= config.preferred_target_rank_max
    ]
    candidate_set = preferred if preferred else matches
    if not candidate_set:
        return TargetSelectionResult(
            schema_version="e7_target_selection_v1",
            e7_query_id=pool.e7_query_id,
            pool_id=pool.pool_id,
            status=TargetSelectionStatus.INELIGIBLE,
            target_document_id=None,
            clueweb_url_hash=None,
            url=None,
            click_count=None,
            global_rank=None,
            rank_bucket=None,
            selection_reason="no_clicked_document_at_rank_6_to_100",
            identity_match_method=None,
        )

    selected = sorted(
        candidate_set,
        key=lambda match: (
            -match.clicked_document.click_count,
            match.pool_document.rank,
            match.pool_document.document_id,
        ),
    )[0]
    selected_rank = selected.pool_document.rank
    reason = (
        "max_click_count_within_rank_6_20"
        if preferred
        else "max_click_count_within_rank_21_100"
    )
    return TargetSelectionResult(
        schema_version="e7_target_selection_v1",
        e7_query_id=pool.e7_query_id,
        pool_id=pool.pool_id,
        status=TargetSelectionStatus.ELIGIBLE,
        target_document_id=selected.pool_document.document_id,
        clueweb_url_hash=selected.clicked_document.clueweb_url_hash,
        url=selected.clicked_document.url,
        click_count=selected.clicked_document.click_count,
        global_rank=selected_rank,
        rank_bucket=rank_bucket(selected_rank, config),
        selection_reason=reason,
        identity_match_method=selected.identity_method,
    )


def persist_target_selection(
    result: TargetSelectionResult,
    config: E7Config,
) -> Path:
    destination = config.output_paths["targets"] / f"{result.e7_query_id}.json"
    write_e7_json_once(destination, result.to_dict(), config)
    return destination
