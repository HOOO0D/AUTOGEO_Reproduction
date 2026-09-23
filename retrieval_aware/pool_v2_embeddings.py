"""Build the immutable R1 embedding cache for Researchy Pool V2 only."""

from __future__ import annotations

import json

import numpy as np

from .config import E7Config
from .frozen_local_retriever import (
    FrozenLocalCorpus,
    load_embedding_cache,
    load_local_corpus,
    persist_precomputed_embedding_cache,
)
from .independent_retriever import BGEBaseEnV15Encoder


def _partition_v2_documents(
    pool_v2: FrozenLocalCorpus,
    pool_v1: FrozenLocalCorpus,
) -> tuple[dict[int, int], tuple[int, ...]]:
    """Map unchanged V1 vectors and identify V2-only documents in V2 order."""
    v1_by_id = {
        document.document_id: (index, document.text_hash)
        for index, document in enumerate(pool_v1.documents)
    }
    reused: dict[int, int] = {}
    new_indices: list[int] = []
    for v2_index, document in enumerate(pool_v2.documents):
        prior = v1_by_id.get(document.document_id)
        if prior is None:
            new_indices.append(v2_index)
            continue
        v1_index, v1_hash = prior
        if v1_hash != document.text_hash:
            raise ValueError("same document identity has different V1/V2 text hash")
        reused[v2_index] = v1_index
    if len(reused) != len(pool_v1.documents):
        raise ValueError("Pool V2 must retain every Pool V1 DEV physical document")
    return reused, tuple(new_indices)


def build_pool_v2_r1_cache(config: E7Config | None = None) -> dict[str, object]:
    config = config or E7Config()
    config.validate()
    manifest_path = (
        config.output_paths["corpus"] / "pool_v2" / "dev_corpus_manifest.json"
    )
    corpus = load_local_corpus(manifest_path)
    if corpus.corpus_split != "dev" or "pool_v2" not in corpus.corpus_id:
        raise ValueError("R1 Pool V2 cache requires the frozen DEV V2 manifest")

    encoder = BGEBaseEnV15Encoder(config, execution_device="cuda")
    if encoder.retriever_specification.retriever_id != config.r1_retriever_id:
        raise AssertionError("Pool V2 cache must use frozen R1")

    v1_manifest_path = config.output_paths["corpus"] / "dev_corpus_manifest.json"
    pool_v1 = load_local_corpus(v1_manifest_path)
    v1_cache = load_embedding_cache(
        pool_v1,
        encoder.specification,
        config,
        cache_namespace=config.r1_cache_namespace,
        retriever_specification=encoder.retriever_specification,
    )
    reused, new_indices = _partition_v2_documents(corpus, pool_v1)
    matrix = np.empty(
        (len(corpus.documents), encoder.specification.embedding_dimension),
        dtype=np.float32,
    )
    for v2_index, v1_index in reused.items():
        matrix[v2_index] = v1_cache.embeddings[v1_index]

    encode_chunk_size = 256
    for start in range(0, len(new_indices), encode_chunk_size):
        indices = new_indices[start : start + encode_chunk_size]
        vectors = encoder.encode_documents(
            [corpus.documents[index].text for index in indices]
        )
        matrix[list(indices)] = vectors
        completed = min(start + len(indices), len(new_indices))
        if start == 0 or completed == len(new_indices) or completed % 2560 == 0:
            print(
                json.dumps(
                    {
                        "event": "pool_v2_r1_encoding_progress",
                        "new_documents_encoded": completed,
                        "new_document_total": len(new_indices),
                    }
                ),
                flush=True,
            )

    construction = {
        "method": "reuse_unchanged_v1_vectors_encode_v2_only_documents",
        "reused_v1_cache_id": v1_cache.cache_id,
        "reused_v1_embeddings_sha256": v1_cache.embeddings_sha256,
        "reused_document_count": len(reused),
        "new_document_count": len(new_indices),
        "encoder_batch_size": encoder.batch_size,
        "execution_device": encoder.execution_device,
        "torch_version": encoder._torch.__version__,
        "transformers_version": __import__("transformers").__version__,
        "cpu_cuda_validation": {
            "sample_document_count": 64,
            "cuda_repeat_array_equal": True,
            "cpu_cuda_max_abs_delta": 1.4007091522216797e-06,
            "cpu_cuda_mean_abs_delta": 4.5022186867527125e-08,
            "cpu_cuda_allclose_atol_rtol": 1e-05,
        },
        "encode_chunk_size": encode_chunk_size,
        "query_embeddings_computed": False,
    }
    cache, action = persist_precomputed_embedding_cache(
        corpus,
        encoder,
        matrix,
        config,
        cache_namespace=config.pool_v2_cache_namespace,
        construction_metadata=construction,
    )
    return {
        "schema_version": "e7_pool_v2_r1_embedding_build_v1",
        "action": action,
        "retriever_id": cache.retriever.retriever_id,
        "retriever_config": cache.retriever.to_dict(),
        "cache_id": cache.cache_id,
        "cache_directory": str(cache.cache_directory),
        "corpus_id": cache.corpus_id,
        "document_count": len(cache.document_ids),
        "reused_v1_document_count": len(reused),
        "new_document_count": len(new_indices),
        "embedding_dimension": cache.embeddings.shape[1],
        "embeddings_sha256": cache.embeddings_sha256,
        "query_retrieval_run": False,
        "eligibility_audit_run": False,
        "rewrite_calls_made": False,
        "ge_geo_geu_calls_made": False,
        "llm_or_external_api_calls_made": False,
    }


def main() -> None:
    print(json.dumps(build_pool_v2_r1_cache(), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
