from __future__ import annotations

import unittest
from pathlib import Path

from retrieval_aware.frozen_local_retriever import FrozenLocalCorpus, LocalDocument
from retrieval_aware.test50_pool_v2_embeddings import (
    FROZEN_R1_CONFIG_SHA256,
    TEST_POOL_V2_CACHE_NAMESPACE,
    plan_dev_embedding_reuse,
)


class Test50PoolV2EmbeddingTests(unittest.TestCase):
    def _corpus(self, second_hash: str = "h2") -> FrozenLocalCorpus:
        return FrozenLocalCorpus(
            corpus_id="test_pool_v2",
            corpus_split="test",
            manifest_path=Path("test.json"),
            manifest_sha256="manifest-hash",
            documents=(
                LocalDocument("d1", "one", "h1"),
                LocalDocument("d2", "two", second_hash),
                LocalDocument("d3", "three", "h3"),
            ),
        )

    def test_reuse_requires_matching_identity_hash_and_r1(self) -> None:
        plan = plan_dev_embedding_reuse(
            self._corpus(),
            ("d2", "d1", "dev-only"),
            ("h2", "h1", "hd"),
            test_r1_config_sha256=FROZEN_R1_CONFIG_SHA256,
            dev_r1_config_sha256=FROZEN_R1_CONFIG_SHA256,
        )
        self.assertEqual(plan.reused_test_to_dev, {0: 1, 1: 0})
        self.assertEqual(plan.new_test_indices, (2,))
        self.assertEqual(
            plan.embedding_sources,
            ("reused_dev", "reused_dev", "newly_encoded"),
        )

    def test_r1_config_mismatch_hard_fails(self) -> None:
        with self.assertRaisesRegex(ValueError, "R1 config"):
            plan_dev_embedding_reuse(
                self._corpus(),
                ("d1",),
                ("h1",),
                test_r1_config_sha256="changed",
                dev_r1_config_sha256=FROZEN_R1_CONFIG_SHA256,
            )
        with self.assertRaisesRegex(ValueError, "DEV cache R1"):
            plan_dev_embedding_reuse(
                self._corpus(),
                ("d1",),
                ("h1",),
                test_r1_config_sha256=FROZEN_R1_CONFIG_SHA256,
                dev_r1_config_sha256="changed",
            )

    def test_same_identity_with_changed_hash_hard_fails(self) -> None:
        with self.assertRaisesRegex(ValueError, "different normalized text hash"):
            plan_dev_embedding_reuse(
                self._corpus(second_hash="changed"),
                ("d2",),
                ("h2",),
                test_r1_config_sha256=FROZEN_R1_CONFIG_SHA256,
                dev_r1_config_sha256=FROZEN_R1_CONFIG_SHA256,
            )

    def test_cache_namespace_is_test_isolated(self) -> None:
        self.assertEqual(TEST_POOL_V2_CACHE_NAMESPACE, "r1_independent/pool_v2/test")
        self.assertNotEqual(TEST_POOL_V2_CACHE_NAMESPACE, "r1_independent/pool_v2")


if __name__ == "__main__":
    unittest.main()
