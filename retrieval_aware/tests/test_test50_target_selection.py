from __future__ import annotations

import unittest

from retrieval_aware.test50_target_selection import (
    TARGET_SOURCE_ALTERNATE,
    TARGET_SOURCE_ORIGINAL,
    select_query_target,
    summarize_targets,
)


def _candidate(document_id: str, slots: list[int], rank: int, *, target: bool = False):
    if rank <= 5:
        bucket = "top5"
    elif rank <= 20:
        bucket = "easy"
    elif rank <= 50:
        bucket = "medium"
    elif rank <= 100:
        bucket = "hard"
    else:
        bucket = "out_of_pool"
    return {
        "physical_document_id": document_id,
        "source_slots": slots,
        "contains_original_autogeo_target_id": target,
        "rank": rank,
        "score": 1.0 / rank,
        "eligible": 6 <= rank <= 100,
        "rank_bucket": bucket,
    }


def _row(target_slot: int, candidates: list[dict[str, object]], slot_ids=None):
    slot_ids = slot_ids or ["d0", "d1", "d2", "d3", "d4"]
    return {
        "query_id": "q",
        "query": "query",
        "eligible": any(candidate["eligible"] for candidate in candidates),
        "original_candidate_slots": [
            {
                "source_slot": index,
                "physical_document_id": document_id,
                "is_original_autogeo_target_id": index == target_slot,
            }
            for index, document_id in enumerate(slot_ids)
        ],
        "unique_original_candidates": candidates,
    }


def _select(row, hashes=None):
    return select_query_target(
        row,
        hashes or {f"d{index}": f"h{index}" for index in range(5)},
        retriever_id="r1_independent_dense",
        retriever_config_sha256="r1-hash",
        corpus_manifest_sha256="corpus-hash",
        eligibility_result_sha256="eligibility-hash",
    )


class Test50TargetSelectionTests(unittest.TestCase):
    def test_original_target_has_absolute_priority(self) -> None:
        candidates = [
            _candidate("d0", [0], 6),
            _candidate("d1", [1], 7),
            _candidate("d2", [2], 90, target=True),
            _candidate("d3", [3], 3),
            _candidate("d4", [4], 2),
        ]
        result = _select(_row(2, candidates))
        self.assertEqual(result["target_document_id"], "d2")
        self.assertEqual(result["target_source"], TARGET_SOURCE_ORIGINAL)
        self.assertEqual(result["rank_bucket"], "hard")

    def test_alternate_uses_bucket_rank_then_id(self) -> None:
        candidates = [
            _candidate("d0", [0], 40),
            _candidate("d1", [1], 9),
            _candidate("d2", [2], 2, target=True),
            _candidate("d3", [3], 9),
            _candidate("d4", [4], 70),
        ]
        result = _select(_row(2, candidates))
        self.assertEqual(result["target_document_id"], "d1")
        self.assertEqual(result["target_source"], TARGET_SOURCE_ALTERNATE)

    def test_duplicate_target_slots_preserve_all_indices(self) -> None:
        slot_ids = ["d0", "shared", "shared", "d3", "d4"]
        candidates = [
            _candidate("d0", [0], 1),
            _candidate("shared", [1, 2], 10, target=True),
            _candidate("d3", [3], 2),
            _candidate("d4", [4], 3),
        ]
        result = _select(
            _row(2, candidates, slot_ids),
            {"d0": "h0", "shared": "hs", "d3": "h3", "d4": "h4"},
        )
        self.assertEqual(result["source_text_indices"], [1, 2])
        self.assertEqual(result["original_autogeo_target_id"], 2)
        self.assertEqual(result["target_source"], TARGET_SOURCE_ORIGINAL)

    def test_ineligible_target_fields_are_null(self) -> None:
        candidates = [
            _candidate("d0", [0], 1),
            _candidate("d1", [1], 2),
            _candidate("d2", [2], 3, target=True),
            _candidate("d3", [3], 4),
            _candidate("d4", [4], 101),
        ]
        result = _select(_row(2, candidates))
        self.assertFalse(result["eligible"])
        self.assertEqual(result["status"], "ineligible")
        for field in (
            "target_document_id",
            "target_text_hash",
            "target_source",
            "source_text_indices",
            "initial_r1_rank",
            "initial_r1_score",
            "rank_bucket",
        ):
            self.assertIsNone(result[field])

    def test_zero_eligible_summary_is_defined(self) -> None:
        summary = summarize_targets(
            [{"eligible": False, "target_source": None, "rank_bucket": None, "initial_r1_rank": None}]
        )
        self.assertEqual(summary["eligible_count"], 0)
        self.assertEqual(
            summary["selected_target_rank_statistics"],
            {"min": None, "max": None, "mean": None, "median": None},
        )


if __name__ == "__main__":
    unittest.main()
