from __future__ import annotations

import base64
import io
import json
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path
from unittest.mock import patch

from retrieval_aware.config import E7Config
from retrieval_aware.overlap_audit import (
    OriginalDocumentIdentity,
    audit_original_top5_overlap,
)
from retrieval_aware.pool_builder import (
    RetrievalDocument,
    RetrievalPool,
    decode_search_response,
    persist_retrieval_pool,
)
from retrieval_aware.query_mapping import (
    AutoGEOQuestion,
    ClickedDocument,
    MappingStatus,
    ResearchyQuestionsJSONLSource,
    ResearchyQuestionRow,
    build_query_mapping,
    normalize_query,
)
from retrieval_aware.target_selector import (
    RankBucket,
    TargetSelectionStatus,
    select_target,
)
from retrieval_aware.utils import ensure_output_layout


def _document(rank: int, text: str | None = None) -> RetrievalDocument:
    return RetrievalDocument(
        document_id=f"clueweb22-en-doc-{rank:05d}",
        rank=rank,
        score=None,
        text=text or f"document text {rank}",
        url=f"https://example.com/doc/{rank}",
        clueweb_url_hash=f"HASH{rank:05d}",
        language="en",
    )


def _pool() -> RetrievalPool:
    return RetrievalPool(
        schema_version="e7_retrieval_pool_v1",
        pool_id="pool-test",
        e7_query_id="e7_q1",
        autogeo_question_id="q1",
        researchy_question_id="rq1",
        query="What makes a society?",
        corpus="clueweb22-b",
        retriever="OpenBMB/MiniCPM-Embedding-Light",
        requested_top_n=100,
        documents=tuple(_document(rank) for rank in range(1, 101)),
    )


class RetrievalStageTests(unittest.TestCase):
    def test_official_source_scans_full_split_for_normalized_duplicates(self) -> None:
        def row(question_id: str, question: str) -> dict[str, object]:
            return {
                "id": question_id,
                "question": question,
                "DocStream": [
                    {
                        "Url": "https://example.com",
                        "CluewebURLHash": "HASH",
                        "UrlLanguage": "en",
                        "Title": "title",
                        "Snippet": "snippet",
                        "Click_Cnt": 1.0,
                    }
                ],
            }

        payload = b"\n".join(
            json.dumps(value).encode("utf-8")
            for value in (
                row("rq1", " Same   Query "),
                row("rq2", "same query"),
                row("rq3", "different query"),
            )
        )
        source = ResearchyQuestionsJSONLSource("https://official.example/test.jsonl")
        with patch(
            "retrieval_aware.query_mapping.urllib.request.urlopen",
            return_value=io.BytesIO(payload),
        ):
            matches = source.load_matching_rows(["SAME QUERY"])
        self.assertEqual(
            [row.researchy_question_id for row in matches],
            ["rq1", "rq2"],
        )

    def test_mapping_is_normalized_exact_only(self) -> None:
        local = [
            AutoGEOQuestion("q1", "  WHAT makes\tA society? ", ("a",) * 5, 0),
            AutoGEOQuestion("q2", "not present", ("b",) * 5, 0),
        ]
        upstream = [
            ResearchyQuestionRow("rq1", "what makes a society?", ()),
        ]
        records = build_query_mapping(local, upstream)
        self.assertEqual(normalize_query(local[0].query), "what makes a society?")
        self.assertEqual(records[0].mapping_status, MappingStatus.MAPPED)
        self.assertEqual(records[0].researchy_question_id, "rq1")
        self.assertEqual(records[0].mapping_method, "normalized_exact")
        self.assertEqual(records[1].mapping_status, MappingStatus.UNMAPPED)
        self.assertIsNone(records[1].researchy_question_id)

    def test_duplicate_exact_query_is_ambiguous_not_guessed(self) -> None:
        local = [AutoGEOQuestion("q1", "same query", ("a",) * 5, 0)]
        upstream = [
            ResearchyQuestionRow("rq1", "same query", ()),
            ResearchyQuestionRow("rq2", "SAME QUERY", ()),
        ]
        record = build_query_mapping(local, upstream)[0]
        self.assertEqual(record.mapping_status, MappingStatus.AMBIGUOUS)
        self.assertIsNone(record.researchy_question_id)

    def test_official_base64_search_schema_decodes_without_score_guess(self) -> None:
        raw = {
            "URL": "https://example.com/page",
            "URL-hash": "ABC123",
            "Language": "en",
            "ClueWeb22-ID": "clueweb22-en0000-00-00000",
            "Clean-Text": "archived clean text",
        }
        encoded = base64.b64encode(json.dumps(raw).encode()).decode()
        documents = decode_search_response({"results": [encoded]})
        self.assertEqual(documents[0].rank, 1)
        self.assertEqual(documents[0].document_id, raw["ClueWeb22-ID"])
        self.assertIsNone(documents[0].score)

    def test_target_prefers_rank_6_to_20_before_higher_click_fallback(self) -> None:
        row = ResearchyQuestionRow(
            researchy_question_id="rq1",
            question="What makes a society?",
            clicked_documents=(
                ClickedDocument(
                    url="https://example.com/doc/10",
                    clueweb_url_hash="HASH00010",
                    click_count=0.2,
                ),
                ClickedDocument(
                    url="https://example.com/doc/25",
                    clueweb_url_hash="HASH00025",
                    click_count=0.9,
                ),
            ),
        )
        selected = select_target(_pool(), row, E7Config())
        self.assertEqual(selected.status, TargetSelectionStatus.ELIGIBLE)
        self.assertEqual(selected.global_rank, 10)
        self.assertEqual(selected.click_count, 0.2)
        self.assertEqual(selected.rank_bucket, RankBucket.EASY)

    def test_target_fallback_and_ineligible(self) -> None:
        fallback_row = ResearchyQuestionRow(
            "rq1",
            "query",
            (
                ClickedDocument(
                    url="https://example.com/doc/75",
                    clueweb_url_hash="HASH00075",
                    click_count=0.4,
                ),
                ClickedDocument(
                    url="https://example.com/doc/30",
                    clueweb_url_hash="HASH00030",
                    click_count=0.8,
                ),
            ),
        )
        selected = select_target(_pool(), fallback_row, E7Config())
        self.assertEqual(selected.global_rank, 30)
        self.assertEqual(selected.rank_bucket, RankBucket.MEDIUM)

        absent_row = ResearchyQuestionRow(
            "rq1",
            "query",
            (ClickedDocument("https://absent.example", "ABSENT", 1.0),),
        )
        absent = select_target(_pool(), absent_row, E7Config())
        self.assertEqual(absent.status, TargetSelectionStatus.INELIGIBLE)
        self.assertIsNone(absent.target_document_id)

    def test_overlap_is_diagnostic_and_uses_text_fallback(self) -> None:
        pool = _pool()
        original = tuple(
            OriginalDocumentIdentity(
                original_index=index,
                text="document text 7" if index == 0 else f"absent {index}",
            )
            for index in range(5)
        )
        audit = audit_original_top5_overlap("e7_q1", original, pool)
        self.assertEqual(audit.matched_document_count, 1)
        self.assertEqual(audit.original5_recall_at_100, 0.2)
        self.assertEqual(
            audit.matches[0].identity_method,
            "normalized_text_sha256_fallback",
        )
        self.assertTrue(audit.diagnostic_only)

    def test_pool_artifact_is_immutable(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            config = replace(E7Config(), output_root=Path(directory) / "e7")
            ensure_output_layout(config)
            persist_retrieval_pool(_pool(), config)
            with self.assertRaises(FileExistsError):
                persist_retrieval_pool(_pool(), config)


if __name__ == "__main__":
    unittest.main()
