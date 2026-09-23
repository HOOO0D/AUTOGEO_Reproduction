from __future__ import annotations

import unittest

from retrieval_aware.retrieval_ablation_smoke import (
    NO_FEEDBACK_CANDIDATE_IDS,
    build_query_aware_no_feedback_prompt,
    freeze_no_feedback_selection,
    selected_no_feedback_result,
    sha256_text,
)


class RetrievalAblationSmokeTests(unittest.TestCase):
    def test_shared_prompt_has_query_and_rules_but_no_retrieval_feedback(self) -> None:
        prompt = build_query_aware_no_feedback_prompt(
            query="why was it successful",
            original_document="original facts",
            filtered_rules=("rule one", "rule two"),
        )
        self.assertIn("why was it successful", prompt)
        self.assertIn("original facts", prompt)
        self.assertIn("rule one", prompt)
        self.assertIn("directly addresses and aligns", prompt)
        lowered = prompt.casefold()
        for forbidden in (
            "r1",
            "retrieval rank",
            "retrieval score",
            "retrieval feedback",
            "previous retrieval",
            "iterative refinement",
        ):
            self.assertNotIn(forbidden, lowered)

    def test_same_input_produces_one_shared_prompt_hash(self) -> None:
        kwargs = {
            "query": "query",
            "original_document": "facts",
            "filtered_rules": ("rule",),
        }
        first = build_query_aware_no_feedback_prompt(**kwargs)
        second = build_query_aware_no_feedback_prompt(**kwargs)
        self.assertEqual(first, second)
        self.assertEqual(sha256_text(first), sha256_text(second))

    def test_no_feedback_selection_is_frozen_to_c1_without_scores(self) -> None:
        hashes = {candidate_id: "a" * 64 for candidate_id in NO_FEEDBACK_CANDIDATE_IDS}
        selection = freeze_no_feedback_selection(hashes)
        self.assertEqual(selection["selected_candidate_id"], "C1")
        self.assertTrue(selection["selected_before_r1_evaluation"])
        self.assertFalse(selection["r1_rank_or_score_available_at_selection"])
        self.assertNotIn("rank", selection)
        self.assertNotIn("score", selection)

    def test_better_posthoc_c2_cannot_replace_precommitted_c1(self) -> None:
        selection = freeze_no_feedback_selection(
            {candidate_id: "b" * 64 for candidate_id in NO_FEEDBACK_CANDIDATE_IDS}
        )
        diagnostics = [
            {"candidate_id": "C1", "rank": 8, "score": 0.7},
            {"candidate_id": "C2", "rank": 1, "score": 0.9},
            {"candidate_id": "C3", "rank": 2, "score": 0.8},
        ]
        selected = selected_no_feedback_result(selection, diagnostics)
        self.assertEqual(selected["candidate_id"], "C1")
        self.assertEqual(selected["rank"], 8)


if __name__ == "__main__":
    unittest.main()
