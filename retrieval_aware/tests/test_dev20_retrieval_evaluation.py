import unittest

from retrieval_aware.dev20_retrieval_evaluation import (
    _method_metrics,
    paired_comparison,
)


class Dev20RetrievalAggregationTests(unittest.TestCase):
    def _records(self):
        return [
            {
                "query_id": "q1",
                "methods": {
                    "left": {"final_rank": 8, "rank_gain": 0, "entered_top5": False},
                    "right": {"final_rank": 4, "rank_gain": 4, "entered_top5": True},
                },
            },
            {
                "query_id": "q2",
                "methods": {
                    "left": {"final_rank": 3, "rank_gain": 2, "entered_top5": True},
                    "right": {"final_rank": 3, "rank_gain": 2, "entered_top5": True},
                },
            },
            {
                "query_id": "q3",
                "methods": {
                    "left": {"final_rank": 2, "rank_gain": 3, "entered_top5": True},
                    "right": {"final_rank": 6, "rank_gain": -1, "entered_top5": False},
                },
            },
        ]

    def test_method_metrics_use_only_passed_eligible_records(self):
        metrics = _method_metrics(self._records(), "right")
        self.assertEqual(metrics["n"], 3)
        self.assertEqual(metrics["hit_at_5_count"], 2)
        self.assertAlmostEqual(metrics["mean_final_rank"], 13 / 3)
        self.assertEqual(metrics["median_final_rank"], 4)

    def test_paired_direction_and_hit5_discordance(self):
        result = paired_comparison(self._records(), "left", "right")
        self.assertEqual(result["treatment_rank_wins"], 1)
        self.assertEqual(result["rank_ties"], 1)
        self.assertEqual(result["treatment_rank_losses"], 1)
        self.assertEqual(result["treatment_hit5_gains"], 1)
        self.assertEqual(result["treatment_hit5_losses"], 1)
        self.assertEqual(result["per_query_rank_delta"], {"q1": 4, "q2": 0, "q3": -4})


if __name__ == "__main__":
    unittest.main()
