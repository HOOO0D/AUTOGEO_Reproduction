"""Independent frozen R1 retriever and its one-query DEV smoke test."""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np

from .config import E7Config
from .frozen_local_retriever import (
    EncoderSpecification,
    FrozenLocalRetriever,
    build_embedding_cache,
    load_local_corpus,
)
from .retriever_contract import (
    EncodingConfiguration,
    FrozenRetrieverSpecification,
)
from .utils import read_json, write_e7_json_once


@dataclass(frozen=True)
class SmokeQuery:
    query_id: str
    query: str
    original_candidate_ids: tuple[str, ...]
    original_target_text_index: int


def _load_first_dev_smoke_query(manifest_path: Path) -> SmokeQuery:
    raw = read_json(manifest_path)
    if raw.get("corpus_split") != "dev":
        raise ValueError("R1 smoke is restricted to the DEV corpus")
    rows = raw.get("query_candidate_sets")
    if not isinstance(rows, list) or len(rows) != 20:
        raise ValueError("R1 smoke expects the frozen DEV20 candidate sets")
    row = rows[0]
    candidate_ids = row.get("original_candidate_ids")
    target_index = row.get("original_target_text_index")
    if not isinstance(candidate_ids, list) or len(candidate_ids) != 5:
        raise ValueError("R1 smoke query must retain five original candidates")
    if not isinstance(target_index, int) or not 0 <= target_index < 5:
        raise ValueError("R1 smoke query has an invalid original target index")
    return SmokeQuery(
        query_id=str(row["question_id"]),
        query=str(row["query"]),
        original_candidate_ids=tuple(str(value) for value in candidate_ids),
        original_target_text_index=target_index,
    )


def r1_retriever_specification(config: E7Config) -> FrozenRetrieverSpecification:
    """Return the single predeclared R1 contract; no model selection occurs."""
    return FrozenRetrieverSpecification(
        retriever_id=config.r1_retriever_id,
        model_name=config.r1_model_name,
        model_revision=config.r1_model_revision,
        query_encoding_config=EncodingConfiguration(
            instruction=config.r1_query_instruction,
            pooling=config.r1_pooling,
            max_length=config.r1_max_length,
        ),
        document_encoding_config=EncodingConfiguration(
            instruction=config.r1_document_instruction,
            pooling=config.r1_pooling,
            max_length=config.r1_max_length,
        ),
        normalization=config.r1_normalization,
        similarity=config.r1_similarity_function,
        max_length=config.r1_max_length,
        embedding_dimension=config.r1_embedding_dimension,
        dtype=config.r1_embedding_dtype,
    )


def resolve_r1_snapshot(config: E7Config) -> Path:
    root = (
        config.output_paths["embeddings"]
        / config.r1_cache_namespace
        / "hf_home"
        / f"models--{config.r1_model_name.replace('/', '--')}"
        / "snapshots"
        / config.r1_model_revision
    )
    required = (
        "config.json",
        "model.safetensors",
        "tokenizer_config.json",
        "tokenizer.json",
        "vocab.txt",
    )
    if not root.is_dir() or any(not (root / name).is_file() for name in required):
        raise FileNotFoundError(
            "pinned BGE R1 snapshot is incomplete; automatic fallback is forbidden"
        )
    return root.resolve()


class BGEBaseEnV15Encoder:
    """Pinned BGE encoder following the official query-to-passage recipe."""

    def __init__(self, config: E7Config, *, execution_device: str = "cpu") -> None:
        config.validate()
        self.retriever_specification = r1_retriever_specification(config)
        self.specification = EncoderSpecification(
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
        self.batch_size = config.r1_embedding_batch_size
        self.snapshot_path = resolve_r1_snapshot(config)
        if execution_device not in {"cpu", "cuda"}:
            raise ValueError("R1 execution device must be cpu or cuda")
        self.execution_device = execution_device

        hf_home = (
            config.output_paths["embeddings"]
            / config.r1_cache_namespace
            / "hf_home"
        )
        os.environ.setdefault("HF_HOME", str(hf_home))
        os.environ.setdefault("HF_HUB_OFFLINE", "1")
        os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

        import torch
        from transformers import AutoModel, AutoTokenizer

        self._torch = torch
        self._functional = torch.nn.functional
        if execution_device == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested for R1 but is unavailable")
        if execution_device == "cuda":
            torch.use_deterministic_algorithms(True)
            torch.backends.cudnn.benchmark = False
            torch.backends.cudnn.deterministic = True
        self._tokenizer = AutoTokenizer.from_pretrained(
            str(self.snapshot_path),
            local_files_only=True,
        )
        self._model = AutoModel.from_pretrained(
            str(self.snapshot_path),
            local_files_only=True,
            torch_dtype=torch.float32,
        )
        self._model.to(execution_device)
        self._model.eval()
        if self._model.config.model_type != "bert":
            raise ValueError("R1 checkpoint is not the expected BERT encoder")
        if self._model.config.hidden_size != config.r1_embedding_dimension:
            raise ValueError("R1 checkpoint hidden size differs from frozen config")
        if self._model.config.max_position_embeddings != config.r1_max_length:
            raise ValueError("R1 checkpoint max length differs from frozen config")

    @staticmethod
    def prepare_query(text: str, instruction: str) -> str:
        return f"{instruction}{text}"

    @staticmethod
    def prepare_document(text: str) -> str:
        return text

    def _encode(self, texts: Sequence[str]) -> np.ndarray:
        rows: list[np.ndarray] = []
        for start in range(0, len(texts), self.batch_size):
            batch = list(texts[start : start + self.batch_size])
            tokens = self._tokenizer(
                batch,
                padding=True,
                truncation=True,
                max_length=self.specification.max_length,
                return_tensors="pt",
            )
            tokens = {
                key: value.to(self.execution_device)
                for key, value in tokens.items()
            }
            with self._torch.no_grad():
                # Official BGE transformers usage selects the first token (CLS).
                vectors = self._model(**tokens)[0][:, 0]
                vectors = self._functional.normalize(vectors, p=2, dim=1)
            rows.append(vectors.cpu().numpy().astype(np.float32, copy=False))
        matrix = np.concatenate(rows, axis=0)
        expected = (len(texts), self.specification.embedding_dimension)
        if matrix.shape != expected or not np.isfinite(matrix).all():
            raise ValueError(f"R1 returned invalid embedding matrix: {matrix.shape}")
        if not np.allclose(np.linalg.norm(matrix, axis=1), 1.0, atol=1e-5):
            raise ValueError("R1 embeddings are not L2-normalized")
        matrix.setflags(write=False)
        return matrix

    def encode_queries(self, texts: Sequence[str]) -> np.ndarray:
        prompt = self.specification.query_instruction
        if not prompt:
            raise ValueError("R1 query instruction is required")
        return self._encode([self.prepare_query(text, prompt) for text in texts])

    def encode_documents(self, texts: Sequence[str]) -> np.ndarray:
        if self.specification.document_instruction is not None:
            raise ValueError("R1 documents must remain unprompted")
        return self._encode([self.prepare_document(text) for text in texts])


def run_r1_one_query_smoke(config: E7Config | None = None) -> dict[str, object]:
    """Encode DEV corpus and rank exactly one query; never select a target."""
    config = config or E7Config()
    config.validate()
    manifest_path = config.output_paths["corpus"] / "dev_corpus_manifest.json"
    corpus = load_local_corpus(manifest_path)
    query = _load_first_dev_smoke_query(manifest_path)

    encoder = BGEBaseEnV15Encoder(config)
    cache, cache_action = build_embedding_cache(
        corpus,
        encoder,
        config,
        cache_namespace=config.r1_cache_namespace,
    )
    retriever = FrozenLocalRetriever(encoder, cache)
    full_count = len(corpus.documents)
    first = retriever.retrieve(query.query, corpus, top_k=full_count)
    second = retriever.retrieve(query.query, corpus, top_k=full_count)
    if first != second:
        raise AssertionError("R1 repeated full rankings are not deterministic")

    hits = {hit.document_id: hit for hit in first}
    original_candidates = []
    for text_index, document_id in enumerate(query.original_candidate_ids):
        hit = hits[document_id]
        original_candidates.append(
            {
                "document_id": document_id,
                "original_text_index": text_index,
                "is_original_autogeo_target": (
                    text_index == query.original_target_text_index
                ),
                "local_rank": hit.rank,
                "local_score": hit.score,
            }
        )

    payload: dict[str, object] = {
        "schema_version": "e7_r1_one_query_smoke_v1",
        "retriever": encoder.retriever_specification.to_dict(),
        "split": "dev",
        "query_count": 1,
        "query_id": query.query_id,
        "query": query.query,
        "corpus_id": corpus.corpus_id,
        "full_corpus_document_count": full_count,
        "cache_id": cache.cache_id,
        "cache_directory": str(cache.cache_directory),
        "ranking_deterministic": True,
        "original_candidate_rankings": original_candidates,
        "eligibility_audit_performed": False,
        "target_selected": False,
        "test50_processed": False,
        "rewrite_calls_made": False,
        "llm_calls_made": False,
        "ge_geo_geu_calls_made": False,
    }
    destination = config.output_paths["logs"] / "r1_one_query_smoke.json"
    if destination.exists():
        if read_json(destination) != payload:
            raise FileExistsError(f"refusing to replace R1 smoke artifact: {destination}")
        action = "reused_identical"
    else:
        write_e7_json_once(destination, payload, config)
        destination.chmod(0o444)
        action = "created"
    return {
        "output_path": str(destination),
        "output_action": action,
        "cache_action": cache_action,
        **payload,
    }


def main() -> None:
    print(json.dumps(run_r1_one_query_smoke(), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
