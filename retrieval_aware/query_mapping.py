"""Optional ClueWeb-backend mapping from Researchy-GEO to Researchy Questions.

The formal pooled-local E7 protocol does not use this module.
"""

from __future__ import annotations

import json
import time
import unicodedata
import urllib.error
import urllib.request
from dataclasses import asdict, dataclass
from enum import Enum
from pathlib import Path
from typing import Iterable, Mapping, Sequence


class MappingStatus(str, Enum):
    MAPPED = "mapped"
    UNMAPPED = "unmapped"
    AMBIGUOUS = "ambiguous"


@dataclass(frozen=True)
class ClickedDocument:
    url: str
    clueweb_url_hash: str
    click_count: float
    url_language: str | None = None
    title: str | None = None
    snippet: str | None = None
    clueweb22_id: str | None = None


@dataclass(frozen=True)
class ResearchyQuestionRow:
    researchy_question_id: str
    question: str
    clicked_documents: tuple[ClickedDocument, ...]


@dataclass(frozen=True)
class AutoGEOQuestion:
    autogeo_question_id: str
    query: str
    text_list: tuple[str, ...]
    target_id: int


@dataclass(frozen=True)
class QueryMappingRecord:
    e7_query_id: str
    autogeo_question_id: str
    researchy_question_id: str | None
    original_query: str
    normalized_query: str
    mapping_status: MappingStatus
    mapping_method: str | None

    def to_dict(self) -> dict[str, object]:
        result = asdict(self)
        result["mapping_status"] = self.mapping_status.value
        return result


def normalize_query(query: str) -> str:
    """Normalize only Unicode, case, and whitespace; no fuzzy matching."""
    normalized = unicodedata.normalize("NFKC", query).casefold()
    return " ".join(normalized.split())


def parse_researchy_row(row: Mapping[str, object]) -> ResearchyQuestionRow:
    question_id = row.get("id")
    question = row.get("question")
    doc_stream = row.get("DocStream")
    if not isinstance(question_id, (str, int)) or not isinstance(question, str):
        raise ValueError("Researchy row requires id and question")
    if not isinstance(doc_stream, list):
        raise ValueError("Researchy row requires a DocStream list")

    clicked_documents: list[ClickedDocument] = []
    for position, raw_document in enumerate(doc_stream):
        if not isinstance(raw_document, Mapping):
            raise ValueError(f"DocStream[{position}] must be an object")
        url = raw_document.get("Url")
        url_hash = raw_document.get("CluewebURLHash")
        click_count = raw_document.get("Click_Cnt")
        if not isinstance(url, str) or not isinstance(url_hash, str):
            raise ValueError(f"DocStream[{position}] lacks Url/CluewebURLHash")
        if not isinstance(click_count, (int, float)):
            raise ValueError(f"DocStream[{position}] lacks numeric Click_Cnt")
        clicked_documents.append(
            ClickedDocument(
                url=url,
                clueweb_url_hash=url_hash,
                click_count=float(click_count),
                url_language=_optional_string(raw_document.get("UrlLanguage")),
                title=_optional_string(raw_document.get("Title")),
                snippet=_optional_string(raw_document.get("Snippet")),
                # The audited DocStream schema does not expose ClueWeb22-ID.
                clueweb22_id=None,
            )
        )
    return ResearchyQuestionRow(
        researchy_question_id=str(question_id),
        question=question,
        clicked_documents=tuple(clicked_documents),
    )


def _optional_string(value: object) -> str | None:
    return value if isinstance(value, str) and value else None


def load_autogeo_questions(
    frozen_query_dir: Path,
    limit: int | None = None,
) -> list[AutoGEOQuestion]:
    """Read current frozen chunks without modifying them."""
    questions: list[AutoGEOQuestion] = []
    seen_ids: set[str] = set()
    for chunk_path in sorted(frozen_query_dir.glob("datachunk_*.json")):
        payload = json.loads(chunk_path.read_text(encoding="utf-8"))
        if not isinstance(payload, dict):
            raise ValueError(f"chunk must contain an object: {chunk_path}")
        for question_id in sorted(payload):
            raw = payload[question_id]
            query = raw.get("query")
            text_list = raw.get("text_list")
            target_id = raw.get("target_id")
            if not isinstance(query, str) or not isinstance(text_list, list):
                raise ValueError(f"invalid Researchy-GEO row: {question_id}")
            if not isinstance(target_id, int) or not 0 <= target_id < len(text_list):
                raise ValueError(f"invalid target_id for {question_id}")
            if not all(isinstance(text, str) for text in text_list):
                raise ValueError(f"non-text document for {question_id}")
            question_id = str(question_id)
            if question_id in seen_ids:
                raise ValueError(f"duplicate AutoGEO question id: {question_id}")
            seen_ids.add(question_id)
            questions.append(
                AutoGEOQuestion(
                    autogeo_question_id=question_id,
                    query=query,
                    text_list=tuple(text_list),
                    target_id=target_id,
                )
            )
            if limit is not None and len(questions) >= limit:
                return questions
    if not questions:
        raise FileNotFoundError(f"no Researchy-GEO chunks found in {frozen_query_dir}")
    return questions


class ResearchyQuestionsJSONLSource:
    """Stream the official test JSONL and retain only requested query rows.

    Scanning the complete split is intentional: a server-side raw-string filter
    cannot detect two distinct strings that become equal after NFKC/case/space
    normalization.  The source dataset is never written to the repository.
    """

    def __init__(
        self,
        url: str,
        timeout_seconds: float = 60.0,
        max_retries: int = 3,
        retry_backoff_seconds: float = 2.0,
    ) -> None:
        self.url = url
        self.timeout_seconds = timeout_seconds
        self.max_retries = max_retries
        self.retry_backoff_seconds = retry_backoff_seconds

    def _load_once(self, normalized_queries: set[str]) -> list[ResearchyQuestionRow]:
        request = urllib.request.Request(
            self.url,
            headers={"Accept": "application/json"},
            method="GET",
        )
        matches: list[ResearchyQuestionRow] = []
        with urllib.request.urlopen(request, timeout=self.timeout_seconds) as response:
            for line_number, encoded_line in enumerate(response, start=1):
                try:
                    raw_row = json.loads(encoded_line.decode("utf-8"))
                except (UnicodeDecodeError, json.JSONDecodeError) as error:
                    raise ValueError(
                        f"invalid Researchy Questions JSONL at line {line_number}"
                    ) from error
                if not isinstance(raw_row, dict):
                    raise ValueError(
                        f"Researchy Questions line {line_number} is not an object"
                    )
                question = raw_row.get("question")
                if not isinstance(question, str):
                    raise ValueError(
                        f"Researchy Questions line {line_number} lacks question"
                    )
                if normalize_query(question) in normalized_queries:
                    matches.append(parse_researchy_row(raw_row))
        return matches

    def load_matching_rows(
        self,
        queries: Iterable[str],
    ) -> list[ResearchyQuestionRow]:
        normalized_queries = {normalize_query(query) for query in queries}
        if not normalized_queries:
            return []
        for attempt in range(1, self.max_retries + 1):
            try:
                return self._load_once(normalized_queries)
            except (urllib.error.URLError, TimeoutError):
                if attempt == self.max_retries:
                    raise
                time.sleep(self.retry_backoff_seconds * attempt)
        raise AssertionError("unreachable Researchy Questions retry state")


def build_query_mapping(
    autogeo_questions: Sequence[AutoGEOQuestion],
    researchy_rows: Sequence[ResearchyQuestionRow],
) -> list[QueryMappingRecord]:
    """Map by normalized exact query equality and retain every failure status."""
    index: dict[str, list[ResearchyQuestionRow]] = {}
    for row in researchy_rows:
        index.setdefault(normalize_query(row.question), []).append(row)

    records: list[QueryMappingRecord] = []
    for question in autogeo_questions:
        normalized = normalize_query(question.query)
        matches = index.get(normalized, [])
        if len(matches) == 1:
            researchy_id = matches[0].researchy_question_id
            status = MappingStatus.MAPPED
            method = "normalized_exact"
        elif not matches:
            researchy_id = None
            status = MappingStatus.UNMAPPED
            method = None
        else:
            researchy_id = None
            status = MappingStatus.AMBIGUOUS
            method = "normalized_exact_ambiguous"
        records.append(
            QueryMappingRecord(
                e7_query_id=f"e7_{question.autogeo_question_id}",
                autogeo_question_id=question.autogeo_question_id,
                researchy_question_id=researchy_id,
                original_query=question.query,
                normalized_query=normalized,
                mapping_status=status,
                mapping_method=method,
            )
        )
    validate_query_mapping(records)
    return records


def validate_query_mapping(records: Iterable[QueryMappingRecord]) -> None:
    seen_e7_ids: set[str] = set()
    seen_autogeo_ids: set[str] = set()
    for record in records:
        if record.e7_query_id in seen_e7_ids:
            raise ValueError(f"duplicate E7 query id: {record.e7_query_id}")
        if record.autogeo_question_id in seen_autogeo_ids:
            raise ValueError(
                f"duplicate AutoGEO question id: {record.autogeo_question_id}"
            )
        seen_e7_ids.add(record.e7_query_id)
        seen_autogeo_ids.add(record.autogeo_question_id)
        if not record.original_query or not record.normalized_query:
            raise ValueError("mapping records require original and normalized query")
        if record.mapping_status is MappingStatus.MAPPED:
            if not record.researchy_question_id:
                raise ValueError("mapped query requires Researchy question id")
            if record.mapping_method != "normalized_exact":
                raise ValueError("v1 permits only normalized exact query mapping")
        elif record.researchy_question_id is not None:
            raise ValueError("unmapped/ambiguous query cannot claim a Researchy id")
