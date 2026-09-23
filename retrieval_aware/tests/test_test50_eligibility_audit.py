from __future__ import annotations

import unittest

from retrieval_aware.config import E7Config
from retrieval_aware.frozen_local_retriever import RetrievalHit
from retrieval_aware.test50_eligibility_audit import (
    TestQueryCandidates,
    audit_query,
    is_eligible_rank,
    rank_bucket,
    summarize_audits,
)


class Test50EligibilityAuditTests(unittest.TestCase):
    def setUp(self) -> None:
        self.config = E7Config()

    def test_frozen_rank_boundaries(self) -> None:
        expected = {
            5: (False, "top5"),
            6: (True, "easy"),
            20: (True, "easy"),
            21: (True, "medium"),
            50: (True, "medium"),
            51: (True, "hard"),
            100: (True, "hard"),
            101: (False, "out_of_pool"),
        }
        for rank, value in expected.items():
            with self.subTest(rank=rank):
                self.assertEqual(
                    (is_eligible_rank(rank, self.config), rank_bucket(rank, self.config)),
                    value,
                )

    def test_duplicate_slots_are_one_physical_candidate(self) -> None:
        query = TestQueryCandidates(
            query_id="q",
            query="query",
            original_candidate_ids=("d1", "d2", "d6", "d6", "d101"),
            original_target_text_index=3,
            original_target_document_id="d6",
        )
        ranking = tuple(
            RetrievalHit(f"d{rank}", rank, 1.0 / rank)
            for rank in range(1, 102)
        )
        result = audit_query(
            query,
            ranking,
            corpus_id="test-pool",
            retriever_id="r1_independent_dense",
            config=self.config,
            deterministic_repeat=ranking,
        )
        self.assertEqual(len(result.original_candidate_slots), 5)
        self.assertEqual(len(result.unique_original_candidates), 4)
        duplicate = next(
            row for row in result.unique_original_candidates
            if row.physical_document_id == "d6"
        )
        self.assertEqual(duplicate.source_slots, (2, 3))
        self.assertTrue(duplicate.contains_original_autogeo_target_id)
        self.assertTrue(result.eligible)
        summary = summarize_audits((result,))
        self.assertEqual(summary["original_candidate_slots_total"], 5)
        self.assertEqual(summary["unique_original_physical_candidates_total"], 4)
        self.assertEqual(summary["eligible_query_count"], 1)

    def test_full_ranking_repeat_must_match(self) -> None:
        query = TestQueryCandidates("q", "query", ("d1", "d2", "d3", "d4", "d5"), 0, "d1")
        ranking = tuple(RetrievalHit(f"d{rank}", rank, 1.0 / rank) for rank in range(1, 102))
        changed = list(ranking)
        changed[0], changed[1] = changed[1], changed[0]
        with self.assertRaisesRegex(AssertionError, "not deterministic"):
            audit_query(
                query,
                ranking,
                corpus_id="test-pool",
                retriever_id="r1_independent_dense",
                config=self.config,
                deterministic_repeat=changed,
            )


if __name__ == "__main__":
    unittest.main()
