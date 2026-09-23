from __future__ import annotations

import unittest

from retrieval_aware.config import E7Config
from retrieval_aware.independent_retriever import (
    BGEBaseEnV15Encoder,
    r1_retriever_specification,
)


class IndependentRetrieverContractTests(unittest.TestCase):
    def test_r1_contract_is_frozen_and_independent_from_r0(self) -> None:
        config = E7Config()
        specification = r1_retriever_specification(config)
        self.assertEqual(specification.retriever_id, "r1_independent_dense")
        self.assertEqual(specification.model_name, "BAAI/bge-base-en-v1.5")
        self.assertNotEqual(specification.model_name, config.retrieval_model)
        self.assertEqual(specification.embedding_dimension, 768)
        self.assertEqual(specification.max_length, 512)
        self.assertEqual(specification.query_encoding_config.pooling, "cls")
        self.assertEqual(specification.document_encoding_config.pooling, "cls")
        self.assertEqual(specification.normalization, "l2")
        self.assertEqual(specification.similarity, "dot_product")

    def test_query_prompt_and_unprompted_document_are_distinct(self) -> None:
        config = E7Config()
        query = BGEBaseEnV15Encoder.prepare_query(
            "example query", config.r1_query_instruction
        )
        document = BGEBaseEnV15Encoder.prepare_document("example query")
        self.assertEqual(
            query,
            "Represent this sentence for searching relevant passages: example query",
        )
        self.assertEqual(document, "example query")
        self.assertNotEqual(query, document)

    def test_execution_device_is_not_part_of_frozen_retriever_spec(self) -> None:
        specification = r1_retriever_specification(E7Config())
        self.assertNotIn("execution_device", specification.to_dict())


if __name__ == "__main__":
    unittest.main()
