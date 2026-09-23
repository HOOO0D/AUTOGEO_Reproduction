from __future__ import annotations

import json
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path
from typing import Sequence

import numpy as np

from retrieval_aware.config import E7Config
from retrieval_aware.frozen_local_retriever import (
    EncoderSpecification,
    FrozenLocalRetriever,
    build_embedding_cache,
    load_local_corpus,
    rerank_with_rewritten_target,
    retrieve,
)
from retrieval_aware.retriever_contract import (
    EncodingConfiguration,
    FrozenRetrieverSpecification,
)


class _FakeEncoder:
    def __init__(self, specification: EncoderSpecification) -> None:
        self.specification = specification
        self.retriever_specification = FrozenRetrieverSpecification(
            retriever_id="r0_local_aligned",
            model_name=specification.model_name,
            model_revision=specification.model_revision,
            query_encoding_config=EncodingConfiguration(
                instruction=specification.query_instruction,
                pooling="official_encode_query",
                max_length=specification.max_length,
            ),
            document_encoding_config=EncodingConfiguration(
                instruction=specification.document_instruction,
                pooling="official_encode_corpus",
                max_length=specification.max_length,
            ),
            normalization=specification.normalization,
            similarity=specification.similarity_function,
            max_length=specification.max_length,
            embedding_dimension=specification.embedding_dimension,
            dtype=specification.dtype,
        )
        self.query_calls: list[tuple[str, ...]] = []
        self.document_calls: list[tuple[str, ...]] = []

    @staticmethod
    def _vector(text: str) -> np.ndarray:
        vectors = {
            "query": (1.0, 0.0, 0.0),
            "alpha": (1.0, 0.0, 0.0),
            "beta": (1.0, 0.0, 0.0),
            "gamma": (0.0, 1.0, 0.0),
            "rewritten": (-1.0, 0.0, 0.0),
        }
        return np.asarray(vectors[text], dtype=np.float32)

    def encode_queries(self, texts: Sequence[str]) -> np.ndarray:
        self.query_calls.append(tuple(texts))
        return np.stack([self._vector(text) for text in texts])

    def encode_documents(self, texts: Sequence[str]) -> np.ndarray:
        self.document_calls.append(tuple(texts))
        return np.stack([self._vector(text) for text in texts])


class FrozenLocalRetrieverTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary_directory = tempfile.TemporaryDirectory()
        root = Path(self.temporary_directory.name)
        frozen = root / "experiments" / "frozen"
        frozen.mkdir(parents=True)
        output = root / "outputs" / "e7_retrieval_aware"
        corpus_directory = output / "corpus"
        corpus_directory.mkdir(parents=True)
        self.config = replace(
            E7Config(),
            project_root=root,
            data_root=root / "data",
            frozen_root=frozen,
            rule_pool_query_dir=frozen / "rule_pool",
            dev_query_dir=frozen / "dev",
            test_query_dir=frozen / "test",
            frozen_query_dir=frozen / "dev",
            output_root=output,
            embedding_dimension=3,
        )
        manifest_path = corpus_directory / "dev_corpus_manifest.json"
        manifest_path.write_text(
            json.dumps(
                {
                    "schema_version": "e7_researchy_pooled_corpus_v1",
                    "corpus_id": "test-corpus",
                    "corpus_split": "dev",
                    "documents": [
                        {
                            "document_id": "doc-b",
                            "text": "beta",
                            "text_hash": "hash-beta",
                        },
                        {
                            "document_id": "doc-a",
                            "text": "alpha",
                            "text_hash": "hash-alpha",
                        },
                        {
                            "document_id": "doc-c",
                            "text": "gamma",
                            "text_hash": "hash-gamma",
                        },
                    ],
                }
            ),
            encoding="utf-8",
        )
        self.corpus = load_local_corpus(manifest_path)
        self.encoder = _FakeEncoder(EncoderSpecification.from_config(self.config))
        self.cache, action = build_embedding_cache(
            self.corpus,
            self.encoder,
            self.config,
        )
        self.assertEqual(action, "created")
        self.retriever = FrozenLocalRetriever(self.encoder, self.cache)

    def tearDown(self) -> None:
        self.temporary_directory.cleanup()

    def test_full_retrieval_is_deterministic_with_document_id_tie_break(self) -> None:
        first = retrieve("query", self.corpus, 3, retriever=self.retriever)
        second = retrieve("query", self.corpus, 3, retriever=self.retriever)
        self.assertEqual(first, second)
        self.assertEqual(
            [hit.document_id for hit in first],
            ["doc-a", "doc-b", "doc-c"],
        )
        self.assertEqual([hit.rank for hit in first], [1, 2, 3])

    def test_counterfactual_reencodes_only_target_and_does_not_mutate(self) -> None:
        corpus_before = self.corpus
        embeddings_before = np.asarray(self.cache.embeddings).copy()
        calls_before = len(self.encoder.document_calls)

        result = rerank_with_rewritten_target(
            "query",
            self.corpus,
            target_id="doc-b",
            rewritten_text="rewritten",
            retriever=self.retriever,
            top_k=3,
        )

        self.assertEqual(len(self.encoder.document_calls), calls_before + 1)
        self.assertEqual(self.encoder.document_calls[-1], ("rewritten",))
        self.assertEqual(result.reencoded_document_ids, ("doc-b",))
        self.assertTrue(result.competitor_invariant_verified)
        self.assertEqual(result.original_target_rank, 2)
        self.assertEqual(result.new_target_rank, 3)
        self.assertIs(self.corpus, corpus_before)
        self.assertTrue(np.array_equal(self.cache.embeddings, embeddings_before))
        self.assertFalse(self.cache.embeddings.flags.writeable)

    def test_cache_order_matches_manifest_and_is_immutable(self) -> None:
        self.assertEqual(self.cache.document_ids, self.corpus.document_ids)
        self.assertEqual(self.cache.text_hashes, self.corpus.text_hashes)
        _, action = build_embedding_cache(
            self.corpus,
            self.encoder,
            self.config,
        )
        self.assertEqual(action, "reused")
        self.assertEqual(len(self.encoder.document_calls), 1)


if __name__ == "__main__":
    unittest.main()
