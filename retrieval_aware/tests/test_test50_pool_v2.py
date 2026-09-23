from __future__ import annotations

import json
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path

from retrieval_aware.config import E7Config
from retrieval_aware.test50_pool_v2 import (
    TEST_POOL_V2_SOURCES,
    _persist_immutable_json,
    build_test_pool_v2_manifest,
    build_test_source_audit,
    persist_test_pool_v2_manifest,
    persist_test_source_audit,
    test_pool_v2_paths,
)
from retrieval_aware.utils import read_json


def _config(root: Path) -> E7Config:
    frozen = root / "experiments" / "frozen"
    return replace(
        E7Config(),
        project_root=root,
        data_root=root / "data",
        frozen_root=frozen,
        rule_pool_query_dir=frozen / "rule_pool",
        dev_query_dir=frozen / "dev20",
        test_query_dir=frozen / "test50",
        frozen_query_dir=frozen / "test50",
        output_root=root / "outputs" / "e7_retrieval_aware",
    )


def _write_chunk(directory: Path, rows: dict[str, object]) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "datachunk_0.json").write_text(
        json.dumps(rows), encoding="utf-8"
    )


class Test50PoolV2Tests(unittest.TestCase):
    def _fixture(self, root: Path):
        config = _config(root)
        directories = {
            "researchy_train": config.data_root / "Researchy-GEO" / "train",
            "rule_pool": config.rule_pool_query_dir,
            "test": config.test_query_dir,
        }
        _write_chunk(
            directories["researchy_train"],
            {
                "train-q": {
                    "query": "training query",
                    "text_list": ["Shared document", "Train only"],
                    "target_id": 0,
                }
            },
        )
        _write_chunk(
            directories["rule_pool"],
            {
                "rule-q": {
                    "query": "rule query",
                    "text_list": ["Shared  document", "Rule only"],
                    "target_id": 1,
                }
            },
        )
        _write_chunk(
            directories["test"],
            {
                "test-q": {
                    "query": "test query",
                    "text_list": [
                        "Shared document",
                        "Test one",
                        "Duplicate text",
                        "Duplicate   text",
                        "Test five",
                    ],
                    "target_id": 3,
                }
            },
        )
        return config, directories

    def test_dedup_provenance_order_and_namespace_isolation(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            config, directories = self._fixture(Path(temporary))
            audit = build_test_source_audit(
                config,
                source_directories=directories,
                expected_test_query_count=1,
            )
            self.assertEqual(tuple(audit["source_definition"]), TEST_POOL_V2_SOURCES)
            audit_result = persist_test_source_audit(audit, config)
            self.assertEqual(audit_result["action"], "created")
            self.assertEqual(
                persist_test_source_audit(audit, config)["action"],
                "reused_identical",
            )
            audit_path = test_pool_v2_paths(config)["audit"]
            manifest = build_test_pool_v2_manifest(
                config,
                audit,
                audit_path,
                source_directories=directories,
                expected_test_query_count=1,
            )
            self.assertEqual(manifest["source_splits"], list(TEST_POOL_V2_SOURCES))
            self.assertNotIn("dev", manifest["source_splits"])
            self.assertEqual(manifest["document_slots"], 9)
            self.assertEqual(manifest["deduplicated_document_count"], 6)
            self.assertEqual(manifest["duplicate_count"], 3)
            ids = [document["document_id"] for document in manifest["documents"]]
            self.assertEqual(ids, sorted(ids))
            shared = next(
                document
                for document in manifest["documents"]
                if document["text"] == "Shared document"
            )
            provenance = [
                shared["source_split"],
                *[item["source_split"] for item in shared["duplicate_provenance"]],
            ]
            self.assertEqual(provenance, ["researchy_train", "rule_pool", "test"])
            query = manifest["query_candidate_sets"][0]
            self.assertEqual(query["original_candidate_ids"][2], query["original_candidate_ids"][3])
            self.assertEqual(query["original_target_text_index"], 3)
            result = persist_test_pool_v2_manifest(manifest, config)
            self.assertEqual(result["action"], "created")
            self.assertEqual(
                persist_test_pool_v2_manifest(manifest, config)["action"],
                "reused_identical",
            )
            manifest_path = test_pool_v2_paths(config)["manifest"]
            self.assertFalse(manifest_path.stat().st_mode & 0o222)
            self.assertTrue(manifest_path.is_relative_to(config.output_root))
            self.assertNotIn("dev", manifest_path.parts[-2:])

    def test_changed_immutable_artifact_hard_fails(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            config = _config(Path(temporary))
            destination = config.output_root / "corpus" / "pool_v2" / "test" / "x.json"
            action, _ = _persist_immutable_json(destination, {"value": 1}, config)
            self.assertEqual(action, "created")
            with self.assertRaisesRegex(FileExistsError, "changed artifact"):
                _persist_immutable_json(destination, {"value": 2}, config)
            self.assertEqual(read_json(destination), {"value": 1})


if __name__ == "__main__":
    unittest.main()
