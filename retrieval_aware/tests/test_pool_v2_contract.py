from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from retrieval_aware.config import E7Config
from retrieval_aware.frozen_local_retriever import (
    FrozenLocalCorpus,
    LocalDocument,
    _cache_directory,
    load_local_corpus,
)
from retrieval_aware.pool_v2_embeddings import _partition_v2_documents


class PoolV2ContractTests(unittest.TestCase):
    def test_source_and_cache_namespaces_are_fixed(self) -> None:
        config = E7Config()
        self.assertEqual(
            config.pool_v2_corpus_sources,
            ("researchy_train", "rule_pool", "dev"),
        )
        self.assertEqual(
            config.pool_v2_cache_namespace,
            "r1_independent/pool_v2",
        )

    def test_nested_v2_cache_namespace_does_not_overlap_v1(self) -> None:
        config = E7Config()
        corpus = FrozenLocalCorpus(
            corpus_id="pool_v2",
            corpus_split="dev",
            manifest_path=Path("manifest.json"),
            manifest_sha256="hash",
            documents=(),
        )
        v1 = _cache_directory(corpus, config)
        v2 = _cache_directory(
            corpus,
            config,
            config.pool_v2_cache_namespace,
        )
        self.assertNotEqual(v1, v2)
        self.assertEqual(
            v2,
            config.output_paths["embeddings"]
            / "r1_independent"
            / "pool_v2"
            / "dev_corpus_cache",
        )

    def test_unsafe_cache_namespace_is_rejected(self) -> None:
        corpus = FrozenLocalCorpus(
            corpus_id="pool_v2",
            corpus_split="dev",
            manifest_path=Path("manifest.json"),
            manifest_sha256="hash",
            documents=(),
        )
        with self.assertRaisesRegex(ValueError, "safe relative path"):
            _cache_directory(corpus, E7Config(), "../escape")

    def test_v2_manifest_schema_loads_without_changing_document_order(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "manifest.json"
            path.write_text(
                json.dumps(
                    {
                        "schema_version": "e7_researchy_pooled_corpus_v2",
                        "corpus_id": "pool_v2",
                        "corpus_split": "dev",
                        "documents": [
                            {"document_id": "d2", "text": "two", "text_hash": "h2"},
                            {"document_id": "d1", "text": "one", "text_hash": "h1"},
                        ],
                    }
                ),
                encoding="utf-8",
            )
            self.assertEqual(load_local_corpus(path).document_ids, ("d2", "d1"))

    def test_v2_partition_reuses_v1_and_marks_only_new_documents(self) -> None:
        v1 = FrozenLocalCorpus(
            corpus_id="v1",
            corpus_split="dev",
            manifest_path=Path("v1.json"),
            manifest_sha256="v1",
            documents=(
                LocalDocument("d1", "one", "h1"),
                LocalDocument("d2", "two", "h2"),
            ),
        )
        v2 = FrozenLocalCorpus(
            corpus_id="v2",
            corpus_split="dev",
            manifest_path=Path("v2.json"),
            manifest_sha256="v2",
            documents=(
                LocalDocument("d2", "two", "h2"),
                LocalDocument("d3", "three", "h3"),
                LocalDocument("d1", "one", "h1"),
            ),
        )
        reused, new_indices = _partition_v2_documents(v2, v1)
        self.assertEqual(reused, {0: 1, 2: 0})
        self.assertEqual(new_indices, (1,))


if __name__ == "__main__":
    unittest.main()
