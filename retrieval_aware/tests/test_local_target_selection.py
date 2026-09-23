from __future__ import annotations

import unittest

from retrieval_aware.frozen_local_retriever import RetrievalHit
from retrieval_aware.local_target_selection import (
    DevQueryCandidates,
    LocalRankBucket,
    LocalTargetSource,
    select_dev_target,
    summarize_selections,
)


def _query(target_index: int = 2) -> DevQueryCandidates:
    candidates = ("d0", "d1", "d2", "d3", "d4")
    return DevQueryCandidates(
        query_id="q1",
        query="query",
        original_candidate_ids=candidates,
        original_target_text_index=target_index,
        original_target_document_id=candidates[target_index],
    )


def _hits(ranks: dict[str, int]) -> tuple[RetrievalHit, ...]:
    return tuple(
        RetrievalHit(document_id=document_id, rank=rank, score=1.0 / rank)
        for document_id, rank in ranks.items()
    )


class LocalTargetSelectionTests(unittest.TestCase):
    def test_original_target_id_has_priority_when_eligible(self) -> None:
        result = select_dev_target(
            _query(target_index=2),
            _hits({"d0": 6, "d1": 7, "d2": 80, "d3": 3, "d4": 101}),
        )
        self.assertTrue(result.eligible)
        self.assertEqual(result.target_document_id, "d2")
        self.assertEqual(result.target_source, LocalTargetSource.ORIGINAL_TARGET_ID)
        self.assertEqual(result.original_text_index, 2)
        self.assertEqual(result.rank_bucket, LocalRankBucket.HARD)

    def test_alternate_uses_bucket_then_rank_priority(self) -> None:
        result = select_dev_target(
            _query(target_index=2),
            _hits({"d0": 45, "d1": 19, "d2": 2, "d3": 8, "d4": 60}),
        )
        self.assertEqual(result.target_document_id, "d3")
        self.assertEqual(
            result.target_source,
            LocalTargetSource.ALTERNATE_ORIGINAL_CANDIDATE,
        )
        self.assertEqual(result.original_text_index, 3)
        self.assertEqual(result.initial_local_rank, 8)
        self.assertEqual(result.rank_bucket, LocalRankBucket.EASY)

    def test_pooled_competitor_cannot_be_selected(self) -> None:
        result = select_dev_target(
            _query(target_index=0),
            _hits(
                {
                    "pooled-competitor": 6,
                    "d0": 1,
                    "d1": 2,
                    "d2": 3,
                    "d3": 4,
                    "d4": 105,
                }
            ),
        )
        self.assertFalse(result.eligible)
        self.assertIsNone(result.target_document_id)
        self.assertEqual(
            result.selection_reason,
            "no_original_candidate_within_rank_6_100",
        )

    def test_duplicate_identity_preserves_five_original_slots(self) -> None:
        query = DevQueryCandidates(
            query_id="q1",
            query="query",
            original_candidate_ids=("d0", "shared", "shared", "d3", "d4"),
            original_target_text_index=2,
            original_target_document_id="shared",
        )
        result = select_dev_target(
            query,
            _hits({"d0": 1, "shared": 10, "d3": 3, "d4": 4}),
        )
        self.assertEqual(len(result.original_candidate_rankings), 5)
        self.assertEqual(result.original_text_index, 2)
        self.assertEqual(result.target_source, LocalTargetSource.ORIGINAL_TARGET_ID)
        self.assertEqual(result.initial_local_rank, 10)

    def test_summary_keeps_target_sources_and_buckets_separate(self) -> None:
        original = select_dev_target(
            _query(target_index=2),
            _hits({"d0": 1, "d1": 2, "d2": 25, "d3": 3, "d4": 4}),
        )
        alternate = select_dev_target(
            _query(target_index=2),
            _hits({"d0": 1, "d1": 55, "d2": 2, "d3": 3, "d4": 4}),
        )
        ineligible = select_dev_target(
            _query(target_index=2),
            _hits({"d0": 1, "d1": 2, "d2": 3, "d3": 4, "d4": 5}),
        )
        summary = summarize_selections((original, alternate, ineligible))
        self.assertEqual(summary["eligible_count"], 2)
        self.assertEqual(summary["ineligible_count"], 1)
        self.assertEqual(summary["original_target_id_eligible_count"], 1)
        self.assertEqual(summary["alternate_candidate_count"], 1)
        self.assertEqual(
            summary["rank_bucket_distribution"],
            {"easy": 0, "medium": 1, "hard": 1},
        )


if __name__ == "__main__":
    unittest.main()
