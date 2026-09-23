from __future__ import annotations

import json
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path

from retrieval_aware.config import E7Config
from retrieval_aware.corpus_builder import (
    CorpusSplit,
    build_corpus_manifest,
    persist_corpus_manifest,
)
from retrieval_aware.utils import ensure_output_layout, normalized_text_sha256


def _row(query: str, documents: list[str], target_id: int) -> dict[str, object]:
    return {"query": query, "text_list": documents, "target_id": target_id}


class CorpusBuilderTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary_directory = tempfile.TemporaryDirectory()
        root = Path(self.temporary_directory.name)
        frozen = root / "experiments" / "frozen"
        rule_pool = frozen / "researchy_rulepool100"
        dev = frozen / "researchy_dev20"
        test = frozen / "researchy_test50"
        for directory in (rule_pool, dev, test):
            directory.mkdir(parents=True)

        shared = "Shared document"
        (rule_pool / "datachunk_0.json").write_text(
            json.dumps(
                {
                    "rule-q": _row(
                        "rule query",
                        [shared, "rule 1", "rule 2", "rule 3", "rule 4"],
                        1,
                    )
                }
            ),
            encoding="utf-8",
        )
        (dev / "datachunk_0.json").write_text(
            json.dumps(
                {
                    "dev-q": _row(
                        "dev query",
                        [
                            " Shared   document ",
                            "dev 1",
                            "dev 2",
                            "dev 3",
                            "DEV SENTINEL",
                        ],
                        0,
                    )
                }
            ),
            encoding="utf-8",
        )
        (test / "datachunk_0.json").write_text(
            json.dumps(
                {
                    "test-q": _row(
                        "test query",
                        [
                            "Shared document",
                            "test 1",
                            "test 2",
                            "test 3",
                            "TEST SENTINEL",
                        ],
                        4,
                    )
                }
            ),
            encoding="utf-8",
        )
        self.config = replace(
            E7Config(),
            project_root=root,
            data_root=root / "data",
            frozen_root=frozen,
            rule_pool_query_dir=rule_pool,
            dev_query_dir=dev,
            test_query_dir=test,
            frozen_query_dir=dev,
            output_root=root / "outputs" / "e7_retrieval_aware",
        )

    def tearDown(self) -> None:
        self.temporary_directory.cleanup()

    def test_dedup_preserves_every_provenance(self) -> None:
        manifest = build_corpus_manifest(CorpusSplit.DEV, self.config)
        self.assertEqual(manifest.document_slots, 10)
        self.assertEqual(manifest.deduplicated_document_count, 9)
        self.assertEqual(manifest.duplicate_count, 1)

        shared_hash = normalized_text_sha256("Shared document")
        shared_id = f"e7doc_sha256_{shared_hash}"
        shared = next(
            document
            for document in manifest.documents
            if document.document_id == shared_id
        )
        self.assertEqual(shared.source_split, "rule_pool")
        self.assertEqual(len(shared.duplicate_provenance), 1)
        duplicate = shared.duplicate_provenance[0]
        self.assertEqual(duplicate.source_split, "dev")
        self.assertEqual(duplicate.source_question_id, "dev-q")
        self.assertEqual(duplicate.source_text_index, 0)
        self.assertTrue(duplicate.is_original_target_id)

    def test_split_isolation_and_query_candidate_identity(self) -> None:
        dev = build_corpus_manifest(CorpusSplit.DEV, self.config)
        test = build_corpus_manifest(CorpusSplit.TEST, self.config)

        self.assertEqual(dev.source_splits, ("rule_pool", "dev"))
        self.assertEqual(test.source_splits, ("rule_pool", "test"))
        self.assertNotIn("TEST SENTINEL", {item.text for item in dev.documents})
        self.assertNotIn("DEV SENTINEL", {item.text for item in test.documents})

        self.assertEqual(dev.query_count, 1)
        query = dev.query_candidate_sets[0]
        self.assertEqual(query.question_id, "dev-q")
        self.assertEqual(len(query.original_candidate_ids), 5)
        self.assertEqual(
            query.original_target_id_document,
            query.original_candidate_ids[query.original_target_text_index],
        )

    def test_manifest_hash_and_persistence_are_deterministic(self) -> None:
        first = build_corpus_manifest(CorpusSplit.DEV, self.config)
        second = build_corpus_manifest(CorpusSplit.DEV, self.config)
        self.assertEqual(first.corpus_id, second.corpus_id)
        self.assertEqual(first.to_dict(), second.to_dict())

        ensure_output_layout(self.config)
        path, action = persist_corpus_manifest(first, self.config)
        self.assertEqual(action, "created")
        same_path, second_action = persist_corpus_manifest(second, self.config)
        self.assertEqual(same_path, path)
        self.assertEqual(second_action, "reused_identical")


if __name__ == "__main__":
    unittest.main()
