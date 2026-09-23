"""Frozen exact dense retriever for the Researchy-GEO pooled local corpus."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import tempfile
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Protocol, Sequence

import numpy as np

from .config import E7Config
from .retriever_contract import (
    EncodingConfiguration,
    FrozenRetrieverSpecification,
)
from .utils import (
    assert_e7_output_path,
    canonical_json_sha256,
    read_json,
    sha256_file,
)


@dataclass(frozen=True)
class LocalDocument:
    document_id: str
    text: str
    text_hash: str


@dataclass(frozen=True)
class FrozenLocalCorpus:
    corpus_id: str
    corpus_split: str
    manifest_path: Path
    manifest_sha256: str
    documents: tuple[LocalDocument, ...]

    @property
    def document_ids(self) -> tuple[str, ...]:
        return tuple(document.document_id for document in self.documents)

    @property
    def text_hashes(self) -> tuple[str, ...]:
        return tuple(document.text_hash for document in self.documents)

    def index_of(self, document_id: str) -> int:
        try:
            return self.document_ids.index(document_id)
        except ValueError as error:
            raise KeyError(f"document is absent from corpus: {document_id}") from error


@dataclass(frozen=True)
class EncoderSpecification:
    model_name: str
    model_revision: str
    query_instruction: str
    document_instruction: str | None
    max_length: int
    embedding_dimension: int
    normalization: str
    similarity_function: str
    dtype: str

    @classmethod
    def from_config(cls, config: E7Config) -> "EncoderSpecification":
        return cls(
            model_name=config.retrieval_model,
            model_revision=config.retrieval_model_revision,
            query_instruction=config.retrieval_query_instruction,
            document_instruction=config.retrieval_document_instruction,
            max_length=config.retrieval_max_input_tokens,
            embedding_dimension=config.embedding_dimension,
            normalization=config.embedding_normalization,
            similarity_function=config.similarity_function,
            dtype=config.embedding_dtype,
        )


class DenseEncoder(Protocol):
    specification: EncoderSpecification
    retriever_specification: FrozenRetrieverSpecification

    def encode_queries(self, texts: Sequence[str]) -> np.ndarray: ...

    def encode_documents(self, texts: Sequence[str]) -> np.ndarray: ...


@dataclass(frozen=True)
class FrozenEmbeddingCache:
    cache_id: str
    cache_directory: Path
    corpus_id: str
    corpus_split: str
    corpus_manifest_sha256: str
    encoder: EncoderSpecification
    retriever: FrozenRetrieverSpecification
    document_ids: tuple[str, ...]
    text_hashes: tuple[str, ...]
    embeddings: np.ndarray
    embeddings_sha256: str


@dataclass(frozen=True)
class RetrievalHit:
    document_id: str
    rank: int
    score: float

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True)
class CounterfactualRerankResult:
    target_id: str
    original_target_rank: int
    original_target_score: float
    new_target_rank: int
    new_target_score: float
    new_top_k: tuple[RetrievalHit, ...]
    reencoded_document_ids: tuple[str, ...]
    competitor_invariant_verified: bool

    def to_dict(self) -> dict[str, object]:
        result = asdict(self)
        result["new_top_k"] = [hit.to_dict() for hit in self.new_top_k]
        result["reencoded_document_ids"] = list(self.reencoded_document_ids)
        return result


def load_local_corpus(manifest_path: Path) -> FrozenLocalCorpus:
    raw = read_json(manifest_path)
    if raw.get("schema_version") not in {
        "e7_researchy_pooled_corpus_v1",
        "e7_researchy_pooled_corpus_v2",
    }:
        raise ValueError(f"unsupported local corpus manifest: {manifest_path}")
    corpus_split = raw.get("corpus_split")
    corpus_id = raw.get("corpus_id")
    raw_documents = raw.get("documents")
    if corpus_split not in {"dev", "test"} or not isinstance(corpus_id, str):
        raise ValueError("local corpus manifest lacks corpus identity")
    if not isinstance(raw_documents, list) or not raw_documents:
        raise ValueError("local corpus manifest has no documents")

    documents: list[LocalDocument] = []
    for raw_document in raw_documents:
        if not isinstance(raw_document, dict):
            raise ValueError("corpus document must be an object")
        document_id = raw_document.get("document_id")
        text = raw_document.get("text")
        text_hash = raw_document.get("text_hash")
        required_values = (document_id, text, text_hash)
        if not all(isinstance(value, str) and value for value in required_values):
            raise ValueError("corpus document lacks identity, text, or text hash")
        documents.append(LocalDocument(document_id, text, text_hash))
    document_ids = [document.document_id for document in documents]
    if len(document_ids) != len(set(document_ids)):
        raise ValueError("local corpus manifest contains duplicate physical IDs")
    return FrozenLocalCorpus(
        corpus_id=corpus_id,
        corpus_split=corpus_split,
        manifest_path=manifest_path.resolve(),
        manifest_sha256=sha256_file(manifest_path),
        documents=tuple(documents),
    )


def resolve_pinned_model_snapshot(config: E7Config) -> Path:
    root = config.output_paths["embeddings"] / "hf_home"
    candidates = sorted(
        path.parent
        for path in root.glob(
            f"**/snapshots/{config.retrieval_model_revision}/model.safetensors"
        )
    )
    complete = [
        path
        for path in candidates
        if (path / "config.json").is_file()
        and (path / "tokenizer.model").is_file()
        and (path / "modeling_minicpm.py").is_file()
    ]
    if len(complete) != 1:
        raise FileNotFoundError(
            "expected exactly one complete pinned MiniCPM snapshot under E7 embeddings"
        )
    return complete[0].resolve()


class MiniCPMEmbeddingEncoder:
    """Pinned official MiniCPM dense encoder used before and after rewriting."""

    def __init__(self, config: E7Config) -> None:
        config.validate()
        self.specification = EncoderSpecification.from_config(config)
        self.retriever_specification = FrozenRetrieverSpecification(
            retriever_id=config.r0_retriever_id,
            model_name=config.retrieval_model,
            model_revision=config.retrieval_model_revision,
            query_encoding_config=EncodingConfiguration(
                instruction=config.retrieval_query_instruction,
                pooling="official_encode_query",
                max_length=config.retrieval_max_input_tokens,
            ),
            document_encoding_config=EncodingConfiguration(
                instruction=config.retrieval_document_instruction,
                pooling="official_encode_corpus",
                max_length=config.retrieval_max_input_tokens,
            ),
            normalization=config.embedding_normalization,
            similarity=config.similarity_function,
            max_length=config.retrieval_max_input_tokens,
            embedding_dimension=config.embedding_dimension,
            dtype=config.embedding_dtype,
        )
        self.batch_size = config.embedding_batch_size
        self.snapshot_path = resolve_pinned_model_snapshot(config)

        hf_home = config.output_paths["embeddings"] / "hf_home"
        os.environ.setdefault("HF_HOME", str(hf_home))
        os.environ.setdefault("TRANSFORMERS_CACHE", str(hf_home / "transformers"))
        os.environ.setdefault("HF_MODULES_CACHE", str(hf_home / "modules"))
        os.environ.setdefault("HF_HUB_OFFLINE", "1")
        os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

        import torch
        from transformers import AutoModel

        self._torch = torch
        self._model = AutoModel.from_pretrained(
            str(self.snapshot_path),
            trust_remote_code=True,
            torch_dtype=torch.float32,
            local_files_only=True,
        )
        self._model.eval()

    def _validate_embeddings(
        self,
        embeddings: object,
        expected_rows: int,
    ) -> np.ndarray:
        matrix = np.asarray(embeddings, dtype=np.float32)
        expected_shape = (
            expected_rows,
            self.specification.embedding_dimension,
        )
        if matrix.shape != expected_shape:
            raise ValueError(
                f"encoder returned {matrix.shape}, expected {expected_shape}"
            )
        if not np.isfinite(matrix).all():
            raise ValueError("encoder returned non-finite dense embeddings")
        norms = np.linalg.norm(matrix, axis=1)
        if not np.allclose(norms, 1.0, atol=1e-5, rtol=1e-5):
            raise ValueError("MiniCPM dense embeddings are not L2-normalized")
        matrix.setflags(write=False)
        return matrix

    def encode_queries(self, texts: Sequence[str]) -> np.ndarray:
        dense, sparse = self._model.encode_query(
            list(texts),
            batch_size=self.batch_size,
            show_progress_bar=False,
            return_dense_vectors=True,
            return_sparse_vectors=False,
            max_length=self.specification.max_length,
            dense_dim=self.specification.embedding_dimension,
            query_instruction=self.specification.query_instruction,
        )
        if sparse is not None:
            raise ValueError("formal dense retriever must not return sparse vectors")
        return self._validate_embeddings(dense, len(texts))

    def encode_documents(self, texts: Sequence[str]) -> np.ndarray:
        if self.specification.document_instruction is not None:
            raise ValueError("document encoder must not apply a prompt")
        dense, sparse = self._model.encode_corpus(
            list(texts),
            batch_size=self.batch_size,
            show_progress_bar=len(texts) > 1,
            return_dense_vectors=True,
            return_sparse_vectors=False,
            max_length=self.specification.max_length,
            dense_dim=self.specification.embedding_dimension,
        )
        if sparse is not None:
            raise ValueError("formal dense retriever must not return sparse vectors")
        return self._validate_embeddings(dense, len(texts))


def _cache_directory(
    corpus: FrozenLocalCorpus,
    config: E7Config,
    cache_namespace: str | None = None,
) -> Path:
    base = config.output_paths["embeddings"]
    if cache_namespace is not None:
        namespace = Path(cache_namespace)
        if (
            namespace.is_absolute()
            or not namespace.parts
            or any(part in {"", ".", ".."} for part in namespace.parts)
        ):
            raise ValueError("cache namespace must be a safe relative path")
        base = base / namespace
    return base / f"{corpus.corpus_split}_corpus_cache"


def _cache_metadata(
    corpus: FrozenLocalCorpus,
    encoder: DenseEncoder,
    embeddings_sha256: str,
    shape: tuple[int, int],
    config: E7Config,
    retriever_specification: FrozenRetrieverSpecification,
    construction_metadata: dict[str, object] | None = None,
) -> dict[str, object]:
    metadata: dict[str, object] = {
        "schema_version": "e7_frozen_local_embeddings_v1",
        "corpus_id": corpus.corpus_id,
        "corpus_split": corpus.corpus_split,
        "corpus_manifest_sha256": corpus.manifest_sha256,
        "model_name": encoder.specification.model_name,
        "model_revision": encoder.specification.model_revision,
        "encoder_config": asdict(encoder.specification),
        "retriever_specification": retriever_specification.to_dict(),
        "batch_size": getattr(encoder, "batch_size", config.embedding_batch_size),
        "document_ids": list(corpus.document_ids),
        "text_hashes": list(corpus.text_hashes),
        "embedding_dimension": encoder.specification.embedding_dimension,
        "normalization": encoder.specification.normalization,
        "similarity_function": encoder.specification.similarity_function,
        "embedding_dtype": encoder.specification.dtype,
        "embedding_shape": list(shape),
        "embeddings_file": "embeddings.npy",
        "embeddings_sha256": embeddings_sha256,
    }
    if construction_metadata is not None:
        metadata["construction_metadata"] = construction_metadata
    identity = dict(metadata)
    metadata["cache_id"] = f"e7emb_{canonical_json_sha256(identity)}"
    return metadata


def build_embedding_cache(
    corpus: FrozenLocalCorpus,
    encoder: DenseEncoder,
    config: E7Config,
    *,
    cache_namespace: str | None = None,
) -> tuple[FrozenEmbeddingCache, str]:
    """Encode every physical document once and atomically freeze the cache."""
    destination = assert_e7_output_path(
        _cache_directory(corpus, config, cache_namespace), config
    )
    if destination.exists():
        return (
            load_embedding_cache(
                corpus,
                encoder.specification,
                config,
                cache_namespace=cache_namespace,
                retriever_specification=encoder.retriever_specification,
            ),
            "reused",
        )

    matrix = encoder.encode_documents(
        [document.text for document in corpus.documents]
    )
    if matrix.dtype != np.float32:
        raise ValueError("embedding cache must be float32")
    if matrix.shape != (
        len(corpus.documents),
        encoder.specification.embedding_dimension,
    ):
        raise ValueError("embedding matrix shape does not match corpus ordering")

    return persist_precomputed_embedding_cache(
        corpus,
        encoder,
        matrix,
        config,
        cache_namespace=cache_namespace,
    )


def persist_precomputed_embedding_cache(
    corpus: FrozenLocalCorpus,
    encoder: DenseEncoder,
    matrix: np.ndarray,
    config: E7Config,
    *,
    cache_namespace: str | None = None,
    construction_metadata: dict[str, object] | None = None,
) -> tuple[FrozenEmbeddingCache, str]:
    """Atomically freeze an already-ordered matrix under an isolated namespace."""
    destination = assert_e7_output_path(
        _cache_directory(corpus, config, cache_namespace), config
    )
    if destination.exists():
        return (
            load_embedding_cache(
                corpus,
                encoder.specification,
                config,
                cache_namespace=cache_namespace,
                retriever_specification=encoder.retriever_specification,
            ),
            "reused",
        )
    matrix = np.asarray(matrix)
    if matrix.dtype != np.float32:
        raise ValueError("embedding cache must be float32")
    if matrix.shape != (
        len(corpus.documents),
        encoder.specification.embedding_dimension,
    ):
        raise ValueError("embedding matrix shape does not match corpus ordering")
    if not np.isfinite(matrix).all():
        raise ValueError("embedding cache contains non-finite values")
    if not np.allclose(
        np.linalg.norm(matrix, axis=1),
        1.0,
        atol=1e-5,
        rtol=1e-5,
    ):
        raise ValueError("embedding cache violates L2 normalization")

    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(
        prefix=f".{destination.name}.",
        dir=destination.parent,
    ) as temporary_directory:
        temporary = Path(temporary_directory)
        embedding_path = temporary / "embeddings.npy"
        with embedding_path.open("wb") as handle:
            np.save(handle, matrix, allow_pickle=False)
        embeddings_sha256 = sha256_file(embedding_path)
        metadata = _cache_metadata(
            corpus,
            encoder,
            embeddings_sha256,
            matrix.shape,
            config,
            encoder.retriever_specification,
            construction_metadata,
        )
        metadata_path = temporary / "metadata.json"
        metadata_path.write_text(
            json.dumps(metadata, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        embedding_path.chmod(0o444)
        metadata_path.chmod(0o444)
        temporary.rename(destination)
    return (
        load_embedding_cache(
            corpus,
            encoder.specification,
            config,
            cache_namespace=cache_namespace,
            retriever_specification=encoder.retriever_specification,
        ),
        "created",
    )


def load_embedding_cache(
    corpus: FrozenLocalCorpus,
    specification: EncoderSpecification,
    config: E7Config,
    *,
    cache_namespace: str | None = None,
    retriever_specification: FrozenRetrieverSpecification | None = None,
) -> FrozenEmbeddingCache:
    directory = assert_e7_output_path(
        _cache_directory(corpus, config, cache_namespace), config
    )
    metadata_path = directory / "metadata.json"
    embedding_path = directory / "embeddings.npy"
    if not metadata_path.is_file() or not embedding_path.is_file():
        raise FileNotFoundError(f"incomplete frozen embedding cache: {directory}")
    metadata = read_json(metadata_path)
    if metadata.get("schema_version") != "e7_frozen_local_embeddings_v1":
        raise ValueError("unsupported embedding cache schema")
    if metadata.get("corpus_id") != corpus.corpus_id:
        raise ValueError("embedding cache corpus ID mismatch")
    if metadata.get("corpus_manifest_sha256") != corpus.manifest_sha256:
        raise ValueError("embedding cache manifest hash mismatch")
    if metadata.get("encoder_config") != asdict(specification):
        raise ValueError("embedding cache encoder configuration mismatch")
    metadata_retriever = metadata.get("retriever_specification")
    if retriever_specification is not None and metadata_retriever is not None:
        if metadata_retriever != retriever_specification.to_dict():
            raise ValueError("embedding cache retriever specification mismatch")
    if tuple(metadata.get("document_ids", ())) != corpus.document_ids:
        raise ValueError("embedding cache document ordering mismatch")
    if tuple(metadata.get("text_hashes", ())) != corpus.text_hashes:
        raise ValueError("embedding cache text hashes mismatch")
    embeddings_sha256 = sha256_file(embedding_path)
    if metadata.get("embeddings_sha256") != embeddings_sha256:
        raise ValueError("immutable embedding file hash mismatch")

    embeddings = np.load(embedding_path, mmap_mode="r", allow_pickle=False)
    expected_shape = (len(corpus.documents), specification.embedding_dimension)
    if embeddings.shape != expected_shape or embeddings.dtype != np.float32:
        raise ValueError("embedding cache matrix shape/dtype mismatch")
    if not np.isfinite(embeddings).all():
        raise ValueError("embedding cache contains non-finite values")
    if not np.allclose(
        np.linalg.norm(embeddings, axis=1),
        1.0,
        atol=1e-5,
        rtol=1e-5,
    ):
        raise ValueError("embedding cache violates L2 normalization")
    embeddings.setflags(write=False)
    effective_retriever = retriever_specification
    if effective_retriever is None:
        effective_retriever = FrozenRetrieverSpecification(
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
    return FrozenEmbeddingCache(
        cache_id=metadata["cache_id"],
        cache_directory=directory,
        corpus_id=corpus.corpus_id,
        corpus_split=corpus.corpus_split,
        corpus_manifest_sha256=corpus.manifest_sha256,
        encoder=specification,
        retriever=effective_retriever,
        document_ids=corpus.document_ids,
        text_hashes=corpus.text_hashes,
        embeddings=embeddings,
        embeddings_sha256=embeddings_sha256,
    )


class FrozenLocalRetriever:
    """One encoder, preprocessing contract, similarity, and ranking path."""

    def __init__(
        self,
        encoder: DenseEncoder,
        cache: FrozenEmbeddingCache,
    ) -> None:
        if encoder.specification != cache.encoder:
            raise ValueError("retriever encoder does not match frozen cache")
        if encoder.retriever_specification != cache.retriever:
            raise ValueError("retriever identity/configuration does not match cache")
        self.encoder = encoder
        self.cache = cache

    @property
    def retriever_id(self) -> str:
        return self.cache.retriever.retriever_id

    def _validate_corpus(self, corpus: FrozenLocalCorpus) -> None:
        if corpus.corpus_id != self.cache.corpus_id:
            raise ValueError("retriever cache and corpus ID differ")
        if corpus.manifest_sha256 != self.cache.corpus_manifest_sha256:
            raise ValueError("retriever cache and corpus manifest differ")
        if corpus.document_ids != self.cache.document_ids:
            raise ValueError("retriever cache ordering differs from corpus")
        if corpus.text_hashes != self.cache.text_hashes:
            raise ValueError("retriever cache hashes differ from corpus")

    @staticmethod
    def _rank(
        document_ids: Sequence[str],
        scores: np.ndarray,
        top_k: int,
    ) -> tuple[RetrievalHit, ...]:
        if not 1 <= top_k <= len(document_ids):
            raise ValueError("top_k must be within the full local corpus")
        order = sorted(
            range(len(document_ids)),
            key=lambda index: (-float(scores[index]), document_ids[index]),
        )
        return tuple(
            RetrievalHit(
                document_id=document_ids[index],
                rank=rank,
                score=float(scores[index]),
            )
            for rank, index in enumerate(order[:top_k], start=1)
        )

    def _query_scores(self, query: str, corpus: FrozenLocalCorpus) -> np.ndarray:
        self._validate_corpus(corpus)
        query_embedding = self.encoder.encode_queries([query])[0]
        scores = np.asarray(self.cache.embeddings @ query_embedding, dtype=np.float32)
        if scores.shape != (len(corpus.documents),):
            raise ValueError("exact dense score vector has unexpected shape")
        return scores

    def retrieve(
        self,
        query: str,
        corpus: FrozenLocalCorpus,
        top_k: int,
    ) -> tuple[RetrievalHit, ...]:
        """Rank the full physical corpus exactly, then truncate to Top-k."""
        scores = self._query_scores(query, corpus)
        return self._rank(corpus.document_ids, scores, top_k)

    def rerank_with_rewritten_target(
        self,
        query: str,
        corpus: FrozenLocalCorpus,
        target_id: str,
        rewritten_text: str,
        top_k: int,
    ) -> CounterfactualRerankResult:
        """Re-encode only target and reuse all N-1 cached competitor vectors."""
        self._validate_corpus(corpus)
        if not 1 <= top_k <= len(corpus.documents):
            raise ValueError("top_k must be within the full local corpus")
        if not rewritten_text.strip():
            raise ValueError("rewritten target text cannot be empty")
        target_index = corpus.index_of(target_id)
        competitor_indices = tuple(
            index for index in range(len(corpus.documents)) if index != target_index
        )
        competitor_identity_before = tuple(
            (
                corpus.documents[index].document_id,
                corpus.documents[index].text,
                corpus.documents[index].text_hash,
            )
            for index in competitor_indices
        )
        competitor_embeddings_before = np.asarray(
            self.cache.embeddings[list(competitor_indices)]
        ).copy()
        cache_digest_before = hashlib.sha256(
            np.asarray(self.cache.embeddings).tobytes()
        ).digest()

        query_embedding = self.encoder.encode_queries([query])[0]
        rewritten_target_embedding = self.encoder.encode_documents(
            [rewritten_text]
        )[0]
        original_scores = np.asarray(
            self.cache.embeddings @ query_embedding,
            dtype=np.float32,
        )
        new_scores = original_scores.copy()
        new_scores[target_index] = np.float32(
            rewritten_target_embedding @ query_embedding
        )

        full_size = len(corpus.documents)
        original_ranking = self._rank(corpus.document_ids, original_scores, full_size)
        new_ranking = self._rank(corpus.document_ids, new_scores, full_size)
        original_target = next(
            hit for hit in original_ranking if hit.document_id == target_id
        )
        new_target = next(hit for hit in new_ranking if hit.document_id == target_id)

        competitor_identity_after = tuple(
            (
                corpus.documents[index].document_id,
                corpus.documents[index].text,
                corpus.documents[index].text_hash,
            )
            for index in competitor_indices
        )
        competitor_embeddings_after = np.asarray(
            self.cache.embeddings[list(competitor_indices)]
        )
        cache_digest_after = hashlib.sha256(
            np.asarray(self.cache.embeddings).tobytes()
        ).digest()
        assert competitor_identity_before == competitor_identity_after
        assert np.array_equal(
            competitor_embeddings_before,
            competitor_embeddings_after,
        )
        assert cache_digest_before == cache_digest_after
        assert not self.cache.embeddings.flags.writeable

        return CounterfactualRerankResult(
            target_id=target_id,
            original_target_rank=original_target.rank,
            original_target_score=original_target.score,
            new_target_rank=new_target.rank,
            new_target_score=new_target.score,
            new_top_k=new_ranking[:top_k],
            reencoded_document_ids=(target_id,),
            competitor_invariant_verified=True,
        )


def retrieve(
    query: str,
    corpus: FrozenLocalCorpus,
    top_k: int,
    *,
    retriever: FrozenLocalRetriever,
) -> tuple[RetrievalHit, ...]:
    """Functional API for full-corpus exact ranking followed by Top-k."""
    return retriever.retrieve(query, corpus, top_k)


def rerank_with_rewritten_target(
    query: str,
    corpus: FrozenLocalCorpus,
    target_id: str,
    rewritten_text: str,
    *,
    retriever: FrozenLocalRetriever,
    top_k: int,
) -> CounterfactualRerankResult:
    """Functional API that re-encodes only the counterfactual target text."""
    return retriever.rerank_with_rewritten_target(
        query,
        corpus,
        target_id,
        rewritten_text,
        top_k,
    )


def _manifest_path(split: str, config: E7Config) -> Path:
    return config.output_paths["corpus"] / f"{split}_corpus_manifest.json"


def build_all_embedding_caches(config: E7Config) -> dict[str, object]:
    encoder = MiniCPMEmbeddingEncoder(config)
    summaries: dict[str, object] = {}
    for split in ("dev", "test"):
        corpus = load_local_corpus(_manifest_path(split, config))
        cache, action = build_embedding_cache(corpus, encoder, config)
        summaries[split] = {
            "action": action,
            "cache_id": cache.cache_id,
            "cache_directory": str(cache.cache_directory),
            "document_count": len(cache.document_ids),
            "embedding_dimension": cache.embeddings.shape[1],
            "embeddings_sha256": cache.embeddings_sha256,
        }
    return {
        "schema_version": "e7_frozen_local_retriever_build_v1",
        "encoder": asdict(encoder.specification),
        "caches": summaries,
        "ann_used": False,
        "external_api_used_for_encoding": False,
        "llm_or_ge_calls_made": False,
    }


def _build_parser() -> argparse.ArgumentParser:
    return argparse.ArgumentParser(
        description="Build immutable exact-dense E7 local corpus embeddings."
    )


def main() -> None:
    _build_parser().parse_args()
    print(
        json.dumps(
            build_all_embedding_caches(E7Config()),
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
