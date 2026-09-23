from __future__ import annotations

import unittest

from retrieval_aware.config import E7Config
from retrieval_aware.frozen_local_retriever import RetrievalHit
from retrieval_aware.pool_v2_eligibility_audit import (
    PoolV2QueryCandidates,
    audit_query,
    is_eligible_rank,
    rank_bucket,
    summarize_audits,
)


class PoolV2EligibilityAuditTests(unittest.TestCase):
    def setUp(self) -> None:
        self.config = E7Config()

    def test_frozen_eligibility_boundaries(self) -> None:
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
        for rank, (eligible, bucket) in expected.items():
            with self.subTest(rank=rank):
                self.assertEqual(is_eligible_rank(rank, self.config), eligible)
                self.assertEqual(rank_bucket(rank, self.config), bucket)

    def test_duplicate_slots_are_one_physical_candidate_with_full_provenance(self) -> None:
        query = PoolV2QueryCandidates(
            query_id="q1",
            query="query",
            original_candidate_ids=("d1", "d2", "d3", "d3", "d4"),
            original_target_text_index=3,
            original_target_document_id="d3",
        )
        ranking = tuple(
            RetrievalHit(document_id=f"d{index}", rank=index, score=1 / index)
            for index in range(1, 102)
        )
        record = audit_query(
            query,
            ranking,
            corpus_id="pool_v2",
            retriever_id="r1_independent_dense",
            config=self.config,
            deterministic_repeat=ranking,
        )
        self.assertEqual(len(record.original_candidate_slots), 5)
        self.assertEqual(len(record.unique_original_candidates), 4)
        duplicate = next(
            candidate
            for candidate in record.unique_original_candidates
            if candidate.physical_document_id == "d3"
        )
        self.assertEqual(duplicate.source_slots, (2, 3))
        self.assertTrue(duplicate.contains_original_autogeo_target_id)
        self.assertEqual(
            sum(
                candidate.contains_original_autogeo_target_id
                for candidate in record.unique_original_candidates
            ),
            1,
        )

    def test_full_ranking_repeat_must_be_identical(self) -> None:
        query = PoolV2QueryCandidates(
            query_id="q1",
            query="query",
            original_candidate_ids=("d1", "d2", "d3", "d4", "d5"),
            original_target_text_index=0,
            original_target_document_id="d1",
        )
        ranking = tuple(
            RetrievalHit(document_id=f"d{index}", rank=index, score=1 / index)
            for index in range(1, 102)
        )
        changed = list(ranking)
        changed[0], changed[1] = changed[1], changed[0]
        with self.assertRaisesRegex(AssertionError, "not deterministic"):
            audit_query(
                query,
                ranking,
                corpus_id="pool_v2",
                retriever_id="r1_independent_dense",
                config=self.config,
                deterministic_repeat=changed,
            )

    def test_summary_counts_query_scoped_unique_candidates(self) -> None:
        query = PoolV2QueryCandidates(
            query_id="q1",
            query="query",
            original_candidate_ids=("d1", "d2", "d3", "d3", "d4"),
            original_target_text_index=3,
            original_target_document_id="d3",
        )
        ranking = tuple(
            RetrievalHit(document_id=f"d{index}", rank=index, score=1 / index)
            for index in range(1, 102)
        )
        record = audit_query(
            query,
            ranking,
            corpus_id="pool_v2",
            retriever_id="r1_independent_dense",
            config=self.config,
        )
        summary = summarize_audits((record,))
        self.assertEqual(summary["original_candidate_slots_total"], 5)
        self.assertEqual(summary["unique_original_physical_candidates_total"], 4)
        self.assertEqual(summary["eligible_query_count"], 0)
        self.assertEqual(
            summary["original_autogeo_target_rank_distribution"]["top5"],
            1,
        )


if __name__ == "__main__":
    unittest.main()
