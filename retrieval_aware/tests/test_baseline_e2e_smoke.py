from __future__ import annotations

import unittest
from unittest.mock import patch

from retrieval_aware.baseline_e2e_smoke import (
    build_method_top5_sources,
    calculate_target_e2e_geo,
    choose_smoke_target,
    rewrite_selected_target_with_stock_autogeo,
)
from retrieval_aware.frozen_local_retriever import (
    FrozenLocalCorpus,
    LocalDocument,
    RetrievalHit,
)


class BaselineE2ESmokeTests(unittest.TestCase):
    def test_smoke_sample_is_preselected_by_original_target_rank_median(self) -> None:
        records = [
            {
                "query_id": "q6",
                "eligible": True,
                "target_source": "original_target_id",
                "initial_r1_rank": 6,
            },
            {
                "query_id": "q15",
                "eligible": True,
                "target_source": "original_target_id",
                "initial_r1_rank": 15,
            },
            {
                "query_id": "q7",
                "eligible": True,
                "target_source": "original_target_id",
                "initial_r1_rank": 7,
            },
            {
                "query_id": "alternate",
                "eligible": True,
                "target_source": "alternate_original_candidate",
                "initial_r1_rank": 7,
            },
        ]
        self.assertEqual(choose_smoke_target(records)["query_id"], "q7")

    def test_stock_wrapper_passes_no_retrieval_feedback(self) -> None:
        calls = []

        def fake_rewrite(**kwargs):
            calls.append(kwargs)
            return "rewritten"

        result = rewrite_selected_target_with_stock_autogeo(
            "original",
            rules_path=__import__("pathlib").Path("rules.json"),
            rewrite_callable=fake_rewrite,
        )
        self.assertEqual(result, "rewritten")
        self.assertEqual(
            set(calls[0]),
            {"document", "dataset", "engine_llm", "rule_path"},
        )
        self.assertNotIn("query", calls[0])
        self.assertNotIn("rank", calls[0])
        self.assertNotIn("score", calls[0])

    def test_method_top5_overlays_only_target(self) -> None:
        documents = tuple(
            LocalDocument(f"d{i}", f"text-{i}", f"hash-{i}")
            for i in range(1, 6)
        )
        corpus = FrozenLocalCorpus(
            corpus_id="pool",
            corpus_split="dev",
            manifest_path=__import__("pathlib").Path("manifest.json"),
            manifest_sha256="hash",
            documents=documents,
        )
        hits = tuple(
            RetrievalHit(document_id=f"d{i}", rank=i, score=1 / i)
            for i in range(1, 6)
        )
        identities, sources = build_method_top5_sources(
            corpus,
            hits,
            target_document_id="d3",
            replacement_target_text="rewritten-target",
        )
        self.assertEqual(identities, ("d1", "d2", "d3", "d4", "d5"))
        self.assertEqual(
            sources,
            ("text-1", "text-2", "rewritten-target", "text-4", "text-5"),
        )

    def test_retrieval_miss_forces_zero_geo_without_parsing_response(self) -> None:
        with patch(
            "retrieval_aware.baseline_e2e_smoke.extract_citations_new",
            side_effect=AssertionError("must not parse a miss"),
        ):
            self.assertEqual(
                calculate_target_e2e_geo("answer", None),
                {"word": 0.0, "pos": 0.0, "wordpos": 0.0},
            )

    def test_dynamic_target_source_index_selects_stock_geo_values(self) -> None:
        with (
            patch(
                "retrieval_aware.baseline_e2e_smoke.extract_citations_new",
                return_value="citations",
            ),
            patch(
                "retrieval_aware.baseline_e2e_smoke.impression_word_count_simple",
                return_value=[0.1, 0.2, 0.3, 0.15, 0.25],
            ),
            patch(
                "retrieval_aware.baseline_e2e_smoke.impression_pos_count_simple",
                return_value=[0.2, 0.1, 0.4, 0.1, 0.2],
            ),
            patch(
                "retrieval_aware.baseline_e2e_smoke.impression_wordpos_count_simple",
                return_value=[0.05, 0.1, 0.5, 0.15, 0.2],
            ),
        ):
            self.assertEqual(
                calculate_target_e2e_geo("answer", 2),
                {"word": 0.3, "pos": 0.4, "wordpos": 0.5},
            )


if __name__ == "__main__":
    unittest.main()
