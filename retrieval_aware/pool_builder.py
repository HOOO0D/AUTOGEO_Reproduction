"""Optional DeepResearchGym adapter and immutable Top-N pool schema.

The adapter is retained for future experiments but is not the formal E7 backend.
"""

from __future__ import annotations

import base64
import json
import os
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Mapping, Protocol, Sequence

from .config import E7Config
from .query_mapping import MappingStatus, QueryMappingRecord
from .utils import canonical_json_sha256, read_json, write_e7_json_once


class RetrievalAuthenticationError(RuntimeError):
    pass


class RetrievalSchemaError(RuntimeError):
    pass


@dataclass(frozen=True)
class RetrievalDocument:
    document_id: str
    rank: int
    score: float | None
    text: str
    url: str
    clueweb_url_hash: str
    language: str

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True)
class RetrievalPool:
    schema_version: str
    pool_id: str
    e7_query_id: str
    autogeo_question_id: str
    researchy_question_id: str
    query: str
    corpus: str
    retriever: str
    requested_top_n: int
    documents: tuple[RetrievalDocument, ...]

    def to_dict(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "pool_id": self.pool_id,
            "e7_query_id": self.e7_query_id,
            "autogeo_question_id": self.autogeo_question_id,
            "researchy_question_id": self.researchy_question_id,
            "query": self.query,
            "corpus": self.corpus,
            "retriever": self.retriever,
            "requested_top_n": self.requested_top_n,
            "documents": [document.to_dict() for document in self.documents],
        }


class RetrievalBackend(Protocol):
    name: str

    def search(self, query: str, top_k: int) -> Sequence[RetrievalDocument]: ...


class DeepResearchGymClueWeb22B:
    """Official GET /search adapter; it does not invent undocumented fields."""

    name = "deepresearchgym_clueweb22_b"

    def __init__(
        self,
        endpoint: str,
        api_key: str,
        timeout_seconds: float = 60.0,
    ) -> None:
        if not api_key:
            raise RetrievalAuthenticationError(
                "DeepResearchGym ClueWeb22 requires a non-empty X-API-Key"
            )
        self.endpoint = endpoint
        self.api_key = api_key
        self.timeout_seconds = timeout_seconds

    @classmethod
    def from_config(cls, config: E7Config) -> "DeepResearchGymClueWeb22B":
        api_key = os.environ.get(config.retrieval_api_key_env, "")
        return cls(
            endpoint=config.retrieval_api_url,
            api_key=api_key,
            timeout_seconds=config.request_timeout_seconds,
        )

    def search(self, query: str, top_k: int) -> tuple[RetrievalDocument, ...]:
        query_string = urllib.parse.urlencode({"query": query, "k": top_k})
        request = urllib.request.Request(
            f"{self.endpoint}?{query_string}",
            headers={"Accept": "application/json", "X-API-Key": self.api_key},
            method="GET",
        )
        try:
            with urllib.request.urlopen(request, timeout=self.timeout_seconds) as response:
                payload = json.loads(response.read().decode("utf-8"))
        except urllib.error.HTTPError as error:
            body = error.read().decode("utf-8", errors="replace")
            if error.code in (401, 403):
                raise RetrievalAuthenticationError(
                    f"DeepResearchGym authentication failed with HTTP {error.code}"
                ) from error
            raise RuntimeError(
                f"DeepResearchGym search failed with HTTP {error.code}: {body[:300]}"
            ) from error
        return decode_search_response(payload)


def decode_search_response(payload: object) -> tuple[RetrievalDocument, ...]:
    """Decode the official ``results: [base64(JSON), ...]`` response."""
    if not isinstance(payload, Mapping):
        raise RetrievalSchemaError("search response must be an object")
    encoded_results = payload.get("results")
    if not isinstance(encoded_results, list):
        raise RetrievalSchemaError("search response requires a results list")

    documents: list[RetrievalDocument] = []
    for rank, encoded in enumerate(encoded_results, start=1):
        if not isinstance(encoded, str):
            raise RetrievalSchemaError(f"result {rank} must be a Base64 string")
        try:
            decoded = base64.b64decode(encoded, validate=True).decode("utf-8")
            raw = json.loads(decoded)
        except Exception as error:
            raise RetrievalSchemaError(f"cannot decode result {rank}") from error
        if not isinstance(raw, Mapping):
            raise RetrievalSchemaError(f"decoded result {rank} must be an object")
        required = {
            "URL": raw.get("URL"),
            "URL-hash": raw.get("URL-hash"),
            "Language": raw.get("Language"),
            "ClueWeb22-ID": raw.get("ClueWeb22-ID"),
            "Clean-Text": raw.get("Clean-Text"),
        }
        if not all(isinstance(value, str) and value for value in required.values()):
            raise RetrievalSchemaError(
                f"decoded result {rank} lacks an official ClueWeb field"
            )
        documents.append(
            RetrievalDocument(
                document_id=required["ClueWeb22-ID"],
                rank=rank,
                score=None,
                text=required["Clean-Text"],
                url=required["URL"],
                clueweb_url_hash=required["URL-hash"],
                language=required["Language"],
            )
        )
    return tuple(documents)


def build_retrieval_pool(
    mapping: QueryMappingRecord,
    backend: RetrievalBackend,
    config: E7Config,
) -> RetrievalPool:
    """Call the fixed retriever once and construct a content-addressed pool."""
    if mapping.mapping_status is not MappingStatus.MAPPED:
        raise ValueError("only mapped Researchy queries can be retrieved")
    if mapping.researchy_question_id is None:
        raise ValueError("mapped query lacks Researchy question id")
    documents = tuple(backend.search(mapping.original_query, config.pool_size))
    if len(documents) != config.pool_size:
        raise RetrievalSchemaError(
            f"requested Top-{config.pool_size}, received {len(documents)} documents"
        )
    expected_ranks = tuple(range(1, config.pool_size + 1))
    if tuple(document.rank for document in documents) != expected_ranks:
        raise RetrievalSchemaError("retrieval ranks must be contiguous and one-based")
    document_ids = [document.document_id for document in documents]
    if len(document_ids) != len(set(document_ids)):
        raise RetrievalSchemaError("retrieval pool contains duplicate ClueWeb22 IDs")

    identity_payload = {
        "schema_version": "e7_retrieval_pool_v1",
        "e7_query_id": mapping.e7_query_id,
        "query": mapping.original_query,
        "corpus": config.clueweb_retrieval_corpus,
        "retriever": config.retrieval_model,
        "documents": [document.to_dict() for document in documents],
    }
    pool_id = f"pool_{canonical_json_sha256(identity_payload)}"
    return RetrievalPool(
        schema_version="e7_retrieval_pool_v1",
        pool_id=pool_id,
        e7_query_id=mapping.e7_query_id,
        autogeo_question_id=mapping.autogeo_question_id,
        researchy_question_id=mapping.researchy_question_id,
        query=mapping.original_query,
        corpus=config.clueweb_retrieval_corpus,
        retriever=config.retrieval_model,
        requested_top_n=config.pool_size,
        documents=documents,
    )


def persist_retrieval_pool(pool: RetrievalPool, config: E7Config) -> Path:
    """Persist once; later methods must load this exact pool file."""
    destination = config.output_paths["pools"] / f"{pool.e7_query_id}.json"
    write_e7_json_once(destination, pool.to_dict(), config)
    return destination


def load_retrieval_pool(path: Path) -> RetrievalPool:
    raw = read_json(path)
    if raw.get("schema_version") != "e7_retrieval_pool_v1":
        raise RetrievalSchemaError("unsupported retrieval pool schema")
    documents = tuple(RetrievalDocument(**document) for document in raw["documents"])
    pool = RetrievalPool(
        schema_version=raw["schema_version"],
        pool_id=raw["pool_id"],
        e7_query_id=raw["e7_query_id"],
        autogeo_question_id=raw["autogeo_question_id"],
        researchy_question_id=raw["researchy_question_id"],
        query=raw["query"],
        corpus=raw["corpus"],
        retriever=raw["retriever"],
        requested_top_n=raw["requested_top_n"],
        documents=documents,
    )
    identity_payload = {
        "schema_version": pool.schema_version,
        "e7_query_id": pool.e7_query_id,
        "query": pool.query,
        "corpus": pool.corpus,
        "retriever": pool.retriever,
        "documents": [document.to_dict() for document in pool.documents],
    }
    expected_pool_id = f"pool_{canonical_json_sha256(identity_payload)}"
    if pool.pool_id != expected_pool_id:
        raise RetrievalSchemaError("pool content hash does not match pool_id")
    return pool
