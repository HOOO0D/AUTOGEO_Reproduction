from __future__ import annotations

import unittest

from retrieval_aware.ra_e2e_smoke import (
    RankedOption,
    build_ra_prompt,
    select_retrieval_best,
)


def _option(source_id: str, rank: int, score: float) -> RankedOption:
    return RankedOption(
        source_id=source_id,
        text=f"text-{source_id}",
        rewrite_hash=f"hash-{source_id}",
        rank=rank,
        score=score,
        top5_document_ids=("d1", "d2", "d3", "d4", "d5"),
        parent_candidate_id="original",
    )


class RAAutoGEOE2ESmokeTests(unittest.TestCase):
    def test_regression_protection_keeps_original_against_worse_rewrites(self) -> None:
        original = _option("original", 7, 0.7)
        selected = select_retrieval_best(
            (original, _option("C1", 10, 0.9), _option("C2", 12, 0.95))
        )
        self.assertEqual(selected.source_id, "original")

    def test_selection_uses_rank_score_then_candidate_id(self) -> None:
        selected = select_retrieval_best(
            (
                _option("original", 7, 0.7),
                _option("C2", 5, 0.8),
                _option("C1", 5, 0.8),
                _option("C3", 5, 0.7),
            )
        )
        self.assertEqual(selected.source_id, "C1")

    def test_equal_rank_higher_score_can_replace_current_best(self) -> None:
        selected = select_retrieval_best(
            (_option("original", 7, 0.7), _option("C1", 7, 0.71))
        )
        self.assertEqual(selected.source_id, "C1")

    def test_round1_prompt_contains_only_original_and_target_feedback(self) -> None:
        prompt = build_ra_prompt(
            candidate_id="C1",
            query="why",
            original_document="original facts",
            filtered_rules=("rule one", "rule two"),
            current_best=_option("original", 7, 0.7),
            initial_rank=7,
            round_index=1,
        )
        for required in (
            "why",
            "original facts",
            "rule one",
            "Current R1 rank: 7",
            "Current R1 score: 0.700000000",
            "keyword stuffing",
            "Do not fabricate facts",
        ):
            self.assertIn(required, prompt)
        self.assertNotIn("Previous Best Candidate", prompt)

    def test_round2_prompt_keeps_original_and_previous_best(self) -> None:
        prompt = build_ra_prompt(
            candidate_id="C4",
            query="why",
            original_document="original facts",
            filtered_rules=("same rule",),
            current_best=_option("C2", 6, 0.8),
            initial_rank=7,
            round_index=2,
        )
        self.assertIn("original facts", prompt)
        self.assertIn("Previous Best Candidate", prompt)
        self.assertIn("text-C2", prompt)
        self.assertIn("sole factual source of truth", prompt)
        self.assertIn("Rank improvement versus original: 1", prompt)


if __name__ == "__main__":
    unittest.main()
