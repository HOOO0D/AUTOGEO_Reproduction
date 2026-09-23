import unittest

from retrieval_aware.dev20_e2e_evaluation import (
    derive_ra_top5,
    ge_content_key,
    paired_downstream,
    strict_mean,
)


GEU_KEYS = (
    "Precision", "Recall", "Clarity", "Depth", "Balance", "Breadth",
    "Support", "Insightfulness", "KPR", "KPC",
)


class Dev20E2EEvaluationTests(unittest.TestCase):
    def test_derive_ra_top5_inserts_target_at_dynamic_rank(self):
        original = ["a", "b", "c", "d", "e"]
        self.assertEqual(
            derive_ra_top5(original, "target", 3),
            ["a", "b", "target", "c", "d"],
        )
        self.assertEqual(derive_ra_top5(original, "target", 9), original)

    def test_derive_ra_top5_rejects_target_already_present(self):
        with self.assertRaises(ValueError):
            derive_ra_top5(["target", "b", "c", "d", "e"], "target", 2)

    def test_ge_content_key_is_shared_but_source_order_sensitive(self):
        key = ge_content_key("query", ["source a", "source b"], "contract")
        self.assertEqual(
            key,
            ge_content_key("query", ["source a", "source b"], "contract"),
        )
        self.assertNotEqual(
            key,
            ge_content_key("query", ["source b", "source a"], "contract"),
        )

    def test_strict_mean_preserves_missing_metrics(self):
        self.assertEqual(strict_mean([1.0, 3.0]), 2.0)
        self.assertIsNone(strict_mean([None, None]))
        self.assertIsNone(strict_mean([1.0, None]))

    def test_paired_downstream_excludes_null_pairs(self):
        rows = [
            {
                "methods": {
                    "left": {
                        "mean_target_e2e_geo": {
                            "word": 0.1, "pos": 0.2, "wordpos": 0.3,
                        },
                        "mean_geu": {key: None for key in GEU_KEYS},
                    },
                    "right": {
                        "mean_target_e2e_geo": {
                            "word": 0.3, "pos": 0.2, "wordpos": 0.1,
                        },
                        "mean_geu": {key: None for key in GEU_KEYS},
                    },
                }
            }
        ]
        result = paired_downstream(rows, "left", "right")
        self.assertEqual(result["metrics"]["geo_word"]["n"], 1)
        self.assertAlmostEqual(
            result["metrics"]["geo_word"]["mean_treatment_minus_baseline"],
            0.2,
        )
        self.assertEqual(result["metrics"]["geu_KPR"]["n"], 0)
        self.assertIsNone(
            result["metrics"]["geu_KPR"]["mean_treatment_minus_baseline"]
        )


if __name__ == "__main__":
    unittest.main()
