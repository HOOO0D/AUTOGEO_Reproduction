from __future__ import annotations

import unittest

from retrieval_aware.config import E7Config
from retrieval_aware.frozen_local_retriever import RetrievalHit
from retrieval_aware.local_target_selection import DevQueryCandidates
from retrieval_aware.r1_eligibility_audit import (
    audit_query,
    rank_range,
    summarize_audits,
)


class R1EligibilityAuditTests(unittest.TestCase):
    def setUp(self) -> None:
        self.config = E7Config()
        self.query = DevQueryCandidates(
            query_id="q1",
            query="query",
            original_candidate_ids=("d1", "d2", "d3", "d4", "d5"),
            original_target_text_index=2,
            original_target_document_id="d3",
        )
        ranks = {"d1": 1, "d2": 6, "d3": 21, "d4": 51, "d5": 101}
        ordered_ids = [f"other-{index}" for index in range(1, 102)]
        for document_id, rank in ranks.items():
            ordered_ids[rank - 1] = document_id
        self.ranking = tuple(
            RetrievalHit(document_id, rank, 1.0 / rank)
            for rank, document_id in enumerate(ordered_ids, start=1)
        )

    def test_rank_ranges_use_unchanged_e7_thresholds(self) -> None:
        expected = {
            1: "rank_1_5",
            5: "rank_1_5",
            6: "rank_6_20",
            20: "rank_6_20",
            21: "rank_21_50",
            50: "rank_21_50",
            51: "rank_51_100",
            100: "rank_51_100",
            101: "rank_over_100",
        }
        for rank, label in expected.items():
            self.assertEqual(rank_range(rank, self.config), label)

    def test_audit_preserves_slots_and_does_not_select_target(self) -> None:
        record = audit_query(self.query, self.ranking, self.config)
        self.assertEqual(len(record.original_candidate_rankings), 5)
        self.assertEqual(
            [candidate.r1_rank for candidate in record.original_candidate_rankings],
            [1, 6, 21, 51, 101],
        )
        self.assertTrue(record.eligible)
        self.assertEqual(record.eligible_slot_count, 3)
        self.assertEqual(record.eligible_candidate_ids, ("d2", "d3", "d4"))
        serialized = record.to_dict()
        self.assertNotIn("target_document_id", serialized)
        self.assertNotIn("selection_reason", serialized)

    def test_summary_counts_all_five_slot_ranges(self) -> None:
        record = audit_query(self.query, self.ranking, self.config)
        summary = summarize_audits([record])
        self.assertEqual(summary["eligible_query_count"], 1)
        self.assertEqual(summary["total_original_candidate_slots"], 5)
        self.assertEqual(
            summary["candidate_slot_rank_distribution"],
            {
                "rank_1_5": 1,
                "rank_6_20": 1,
                "rank_21_50": 1,
                "rank_51_100": 1,
                "rank_over_100": 1,
            },
        )


if __name__ == "__main__":
    unittest.main()
