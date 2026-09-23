from __future__ import annotations

import unittest

from retrieval_aware.config import E7Config
from retrieval_aware.retriever_registry import cache_namespace_for


class RetrieverRegistryTests(unittest.TestCase):
    def test_cache_namespaces_keep_r0_and_r1_disjoint(self) -> None:
        config = E7Config()
        self.assertIsNone(cache_namespace_for(config.r0_retriever_id, config))
        self.assertEqual(
            cache_namespace_for(config.r1_retriever_id, config),
            "r1_independent",
        )

    def test_unknown_retriever_is_rejected_without_fallback(self) -> None:
        with self.assertRaisesRegex(ValueError, "unknown frozen retriever_id"):
            cache_namespace_for("r2_not_allowed", E7Config())


if __name__ == "__main__":
    unittest.main()
