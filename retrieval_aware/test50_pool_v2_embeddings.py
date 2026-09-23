"""Build the immutable frozen-R1 cache for TEST50 Pool V2.

Unchanged physical documents may reuse DEV vectors only when document ID,
normalized-text hash, and the complete frozen R1 contract all match.  The DEV
cache is opened read-only and verified unchanged after construction.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Sequence

import numpy as np

from .config import E7Config
from .frozen_local_retriever import (
    EncoderSpecification,
    FrozenLocalCorpus,
    load_embedding_cache,
    load_local_corpus,
    persist_precomputed_embedding_cache,
)
from .independent_retriever import BGEBaseEnV15Encoder, r1_retriever_specification
from .test50_pool_v2 import test_pool_v2_paths
from .utils import canonical_json_sha256, read_json, sha256_file


FROZEN_R1_CONFIG_SHA256 = (
    "6ac758e1ee097a0df3c4df14e8d3d1534a5374a6e933b4ea355158fb48477ba2"
)
TEST_POOL_V2_CACHE_NAMESPACE = "r1_independent/pool_v2/test"


@dataclass(frozen=True)
class EmbeddingReusePlan:
    reused_test_to_dev: dict[int, int]
    new_test_indices: tuple[int, ...]
    embedding_sources: tuple[str, ...]


def r1_encoder_specification(config: E7Config) -> EncoderSpecification:
    return EncoderSpecification(
        model_name=config.r1_model_name,
        model_revision=config.r1_model_revision,
        query_instruction=config.r1_query_instruction,
        document_instruction=config.r1_document_instruction,
        max_length=config.r1_max_length,
        embedding_dimension=config.r1_embedding_dimension,
        normalization=config.r1_normalization,
        similarity_function=config.r1_similarity_function,
        dtype=config.r1_embedding_dtype,
    )


def frozen_r1_config_sha256(config: E7Config) -> str:
    config.validate()
    return canonical_json_sha256(r1_retriever_specification(config).to_dict())


def assert_frozen_r1_contract(config: E7Config) -> str:
    digest = frozen_r1_config_sha256(config)
    if digest != FROZEN_R1_CONFIG_SHA256:
        raise ValueError(
            f"R1 config hash mismatch: {digest} != {FROZEN_R1_CONFIG_SHA256}"
        )
    return digest


def plan_dev_embedding_reuse(
    test_corpus: FrozenLocalCorpus,
    dev_document_ids: Sequence[str],
    dev_text_hashes: Sequence[str],
    *,
    test_r1_config_sha256: str,
    dev_r1_config_sha256: str,
) -> EmbeddingReusePlan:
    """Create a deterministic reuse plan without loading an embedding model."""
    if test_r1_config_sha256 != FROZEN_R1_CONFIG_SHA256:
        raise ValueError("TEST cache does not use the frozen DEV R1 config")
    if dev_r1_config_sha256 != FROZEN_R1_CONFIG_SHA256:
        raise ValueError("DEV cache R1 config hash mismatch")
    if len(dev_document_ids) != len(dev_text_hashes):
        raise ValueError("DEV cache identity/hash ordering lengths differ")
    if len(set(dev_document_ids)) != len(dev_document_ids):
        raise ValueError("DEV cache contains duplicate document identities")
    dev_by_id = {
        document_id: (index, dev_text_hashes[index])
        for index, document_id in enumerate(dev_document_ids)
    }
    reused: dict[int, int] = {}
    new_indices: list[int] = []
    sources: list[str] = []
    for test_index, document in enumerate(test_corpus.documents):
        prior = dev_by_id.get(document.document_id)
        if prior is None:
            new_indices.append(test_index)
            sources.append("newly_encoded")
            continue
        dev_index, dev_hash = prior
        if dev_hash != document.text_hash:
            raise ValueError("same document ID has different normalized text hash")
        reused[test_index] = dev_index
        sources.append("reused_dev")
    return EmbeddingReusePlan(
        reused_test_to_dev=reused,
        new_test_indices=tuple(new_indices),
        embedding_sources=tuple(sources),
    )


def _cache_directory(config: E7Config) -> Path:
    return (
        config.output_paths["embeddings"]
        / TEST_POOL_V2_CACHE_NAMESPACE
        / "test_corpus_cache"
    )


def _dev_paths(config: E7Config) -> dict[str, Path]:
    cache = (
        config.output_paths["embeddings"]
        / config.pool_v2_cache_namespace
        / "dev_corpus_cache"
    )
    return {
        "manifest": config.output_paths["corpus"] / "pool_v2" / "dev_corpus_manifest.json",
        "embeddings": cache / "embeddings.npy",
        "metadata": cache / "metadata.json",
    }


def _validate_reused_metadata(
    metadata: dict[str, object],
    corpus: FrozenLocalCorpus,
    plan: EmbeddingReusePlan,
    config_hash: str,
) -> None:
    construction = metadata.get("construction_metadata")
    if not isinstance(construction, dict):
        raise ValueError("TEST cache lacks construction metadata")
    if construction.get("r1_config_sha256") != config_hash:
        raise ValueError("TEST cache construction used another R1 contract")
    rows = construction.get("document_embedding_sources")
    if not isinstance(rows, list) or len(rows) != len(corpus.documents):
        raise ValueError("TEST cache embedding provenance is incomplete")
    expected = [
        {
            "document_id": document.document_id,
            "text_hash": document.text_hash,
            "embedding_source": plan.embedding_sources[index],
        }
        for index, document in enumerate(corpus.documents)
    ]
    if rows != expected:
        raise ValueError("TEST cache embedding provenance changed")


def build_test_pool_v2_r1_cache(config: E7Config | None = None) -> dict[str, object]:
    config = config or E7Config()
    config_hash = assert_frozen_r1_contract(config)
    test_manifest_path = test_pool_v2_paths(config)["manifest"]
    test_corpus = load_local_corpus(test_manifest_path)
    if test_corpus.corpus_split != "test" or "pool_v2" not in test_corpus.corpus_id:
        raise ValueError("TEST R1 cache requires the immutable TEST Pool V2 manifest")

    dev_paths = _dev_paths(config)
    dev_hashes_before = {name: sha256_file(path) for name, path in dev_paths.items()}
    specification = r1_encoder_specification(config)
    retriever_specification = r1_retriever_specification(config)
    dev_corpus = load_local_corpus(dev_paths["manifest"])
    dev_cache = load_embedding_cache(
        dev_corpus,
        specification,
        config,
        cache_namespace=config.pool_v2_cache_namespace,
        retriever_specification=retriever_specification,
    )
    dev_config_hash = canonical_json_sha256(dev_cache.retriever.to_dict())
    plan = plan_dev_embedding_reuse(
        test_corpus,
        dev_cache.document_ids,
        dev_cache.text_hashes,
        test_r1_config_sha256=config_hash,
        dev_r1_config_sha256=dev_config_hash,
    )

    destination = _cache_directory(config)
    if destination.exists():
        cache = load_embedding_cache(
            test_corpus,
            specification,
            config,
            cache_namespace=TEST_POOL_V2_CACHE_NAMESPACE,
            retriever_specification=retriever_specification,
        )
        metadata = read_json(destination / "metadata.json")
        _validate_reused_metadata(metadata, test_corpus, plan, config_hash)
        action = "reused_identical"
    else:
        encoder = BGEBaseEnV15Encoder(config, execution_device="cuda")
        if canonical_json_sha256(encoder.retriever_specification.to_dict()) != config_hash:
            raise AssertionError("loaded R1 encoder differs from frozen contract")
        matrix = np.empty(
            (len(test_corpus.documents), specification.embedding_dimension),
            dtype=np.float32,
        )
        for test_index, dev_index in plan.reused_test_to_dev.items():
            matrix[test_index] = dev_cache.embeddings[dev_index]
        chunk_size = 256
        for start in range(0, len(plan.new_test_indices), chunk_size):
            indices = plan.new_test_indices[start : start + chunk_size]
            matrix[list(indices)] = encoder.encode_documents(
                [test_corpus.documents[index].text for index in indices]
            )
            print(
                json.dumps(
                    {
                        "event": "test_pool_v2_r1_encoding_progress",
                        "new_documents_encoded": min(start + len(indices), len(plan.new_test_indices)),
                        "new_document_total": len(plan.new_test_indices),
                    }
                ),
                flush=True,
            )
        construction = {
            "method": "reuse_matching_dev_vectors_encode_test_only_documents",
            "r1_config_sha256": config_hash,
            "dev_cache_id": dev_cache.cache_id,
            "dev_embeddings_sha256": dev_cache.embeddings_sha256,
            "reused_dev_document_count": len(plan.reused_test_to_dev),
            "newly_encoded_document_count": len(plan.new_test_indices),
            "document_embedding_sources": [
                {
                    "document_id": document.document_id,
                    "text_hash": document.text_hash,
                    "embedding_source": plan.embedding_sources[index],
                }
                for index, document in enumerate(test_corpus.documents)
            ],
            "execution_device": "cuda",
            "encode_chunk_size": chunk_size,
            "query_embeddings_computed": False,
        }
        cache, created_action = persist_precomputed_embedding_cache(
            test_corpus,
            encoder,
            matrix,
            config,
            cache_namespace=TEST_POOL_V2_CACHE_NAMESPACE,
            construction_metadata=construction,
        )
        if created_action != "created":
            raise AssertionError("fresh TEST cache was not created atomically")
        metadata = read_json(destination / "metadata.json")
        _validate_reused_metadata(metadata, test_corpus, plan, config_hash)
        action = "created"

    dev_hashes_after = {name: sha256_file(path) for name, path in dev_paths.items()}
    if dev_hashes_after != dev_hashes_before:
        raise AssertionError("TEST cache construction modified a frozen DEV artifact")
    for artifact in (destination / "embeddings.npy", destination / "metadata.json"):
        if artifact.stat().st_mode & 0o222:
            raise PermissionError(f"TEST cache artifact is writable: {artifact}")
    return {
        "schema_version": "e7_test50_pool_v2_r1_embedding_build_v1",
        "action": action,
        "retriever_id": retriever_specification.retriever_id,
        "retriever_config_sha256": config_hash,
        "cache_id": cache.cache_id,
        "cache_directory": str(cache.cache_directory),
        "corpus_id": cache.corpus_id,
        "document_count": len(cache.document_ids),
        "reused_dev_document_count": len(plan.reused_test_to_dev),
        "newly_encoded_document_count": len(plan.new_test_indices),
        "embedding_dimension": cache.embeddings.shape[1],
        "embeddings_sha256": cache.embeddings_sha256,
        "dev_cache_unchanged": True,
        "query_retrieval_run": False,
        "eligibility_audit_run": False,
        "llm_or_external_api_calls_made": False,
    }


def main() -> None:
    print(json.dumps(build_test_pool_v2_r1_cache(), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
