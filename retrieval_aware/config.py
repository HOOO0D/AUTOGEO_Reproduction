"""Central configuration for E7 Retrieval-Aware AutoGEO."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any


def _project_root() -> Path:
    return Path(__file__).resolve().parents[1]


@dataclass(frozen=True)
class E7Config:
    """Single source of truth for E7 retrieval and evaluation parameters.

    The formal backend is the frozen Researchy-GEO pooled local corpus.  The
    ClueWeb settings remain available for an explicitly selected future backend.
    Constructing this config never initiates network or model work.
    """

    project_root: Path = field(default_factory=_project_root)
    data_root: Path = field(default_factory=lambda: _project_root() / "data")
    frozen_root: Path = field(
        default_factory=lambda: _project_root() / "experiments" / "frozen"
    )
    rule_pool_query_dir: Path = field(
        default_factory=lambda: (
            _project_root() / "experiments" / "frozen" / "researchy_rulepool100"
        )
    )
    dev_query_dir: Path = field(
        default_factory=lambda: (
            _project_root() / "experiments" / "frozen" / "researchy_dev20"
        )
    )
    test_query_dir: Path = field(
        default_factory=lambda: (
            _project_root() / "experiments" / "frozen" / "researchy_test50"
        )
    )
    frozen_query_dir: Path = field(
        default_factory=lambda: (
            _project_root()
            / "experiments"
            / "frozen"
            / "researchy_dev20"
        )
    )
    output_root: Path = field(
        default_factory=lambda: _project_root() / "outputs" / "e7_retrieval_aware"
    )

    pool_size: int = 100
    ge_top_k: int = 5
    preferred_target_rank_min: int = 6
    preferred_target_rank_max: int = 20
    medium_target_rank_max: int = 50
    max_target_rank: int = 100
    num_candidates_per_round: int = 3
    max_rounds: int = 2

    retrieval_backend: str = "researchy_pooled_local"
    retrieval_corpus: str = "researchy_geo_pooled_local"
    dev_corpus_sources: tuple[str, ...] = ("rule_pool", "dev")
    test_corpus_sources: tuple[str, ...] = ("rule_pool", "test")

    clueweb_retrieval_backend: str = "deepresearchgym"
    clueweb_retrieval_corpus: str = "clueweb22-b"
    retrieval_model: str = "openbmb/MiniCPM-Embedding-Light"
    r0_retriever_id: str = "r0_local_aligned"
    retrieval_model_revision: str = (
        "ce6cb0e22f4838f44731910be439651eebc76838"
    )
    embedding_dimension: int = 1024
    retrieval_max_input_tokens: int = 8192
    retrieval_query_instruction: str = "Query:"
    retrieval_document_instruction: str | None = None
    embedding_normalization: str = "l2"
    similarity_function: str = "dot_product"
    embedding_dtype: str = "float32"
    embedding_batch_size: int = 1

    r1_retriever_id: str = "r1_independent_dense"
    r1_model_name: str = "BAAI/bge-base-en-v1.5"
    r1_model_revision: str = "a5beb1e3e68b9ab74eb54cfd186867f64f240e1a"
    r1_query_instruction: str = (
        "Represent this sentence for searching relevant passages: "
    )
    r1_document_instruction: str | None = None
    r1_pooling: str = "cls"
    r1_max_length: int = 512
    r1_embedding_dimension: int = 768
    r1_normalization: str = "l2"
    r1_similarity_function: str = "dot_product"
    r1_embedding_dtype: str = "float32"
    r1_embedding_batch_size: int = 4
    r1_cache_namespace: str = "r1_independent"
    pool_v2_corpus_sources: tuple[str, ...] = (
        "researchy_train",
        "rule_pool",
        "dev",
    )
    pool_v2_cache_namespace: str = "r1_independent/pool_v2"
    diskann_l_multiplier: int = 5
    retrieval_api_url: str = "https://clueweb22.us/search"
    retrieval_api_key_env: str = "CLUEWEB_API_KEY"
    request_timeout_seconds: float = 60.0

    researchy_dataset_id: str = "corbyrosset/researchy_questions"
    researchy_dataset_config: str = "default"
    researchy_dataset_split: str = "test"
    researchy_test_jsonl_url: str = (
        "https://huggingface.co/datasets/corbyrosset/researchy_questions/"
        "resolve/main/researchy_questions.test.jsonl"
    )
    researchy_download_max_retries: int = 3
    researchy_retry_backoff_seconds: float = 2.0
    smoke_query_limit: int = 3

    output_subdirectories: tuple[str, ...] = (
        "pools",
        "targets",
        "embeddings",
        "rewrites",
        "calibration",
        "evaluation",
        "logs",
        "mappings",
        "audits",
        "corpus",
    )

    @property
    def diskann_l(self) -> int:
        return self.pool_size * self.diskann_l_multiplier

    @property
    def output_paths(self) -> dict[str, Path]:
        return {
            name: self.output_root / name for name in self.output_subdirectories
        }

    def validate(self) -> None:
        if self.pool_size < 1:
            raise ValueError("pool_size must be positive")
        if not 1 <= self.ge_top_k < self.pool_size:
            raise ValueError("ge_top_k must be within [1, pool_size)")
        if self.preferred_target_rank_min <= self.ge_top_k:
            raise ValueError("preferred targets must start below the GE top-k cutoff")
        if not (
            self.preferred_target_rank_min
            <= self.preferred_target_rank_max
            <= self.medium_target_rank_max
            <= self.max_target_rank
            <= self.pool_size
        ):
            raise ValueError("target rank bounds must be ordered within the pool")
        if self.embedding_dimension < 1 or self.retrieval_max_input_tokens < 1:
            raise ValueError("retriever dimensions and token limit must be positive")
        if self.embedding_batch_size < 1:
            raise ValueError("embedding_batch_size must be positive")
        if self.r0_retriever_id != "r0_local_aligned":
            raise ValueError("R0 retriever identity is frozen")
        if self.retrieval_query_instruction != "Query:":
            raise ValueError("MiniCPM query instruction must remain Query:")
        if self.retrieval_document_instruction is not None:
            raise ValueError("MiniCPM corpus documents must not receive a prompt")
        if self.embedding_normalization != "l2":
            raise ValueError("formal E7 embeddings must be L2-normalized")
        if self.similarity_function != "dot_product":
            raise ValueError("formal E7 similarity must be exact dot product")
        if self.embedding_dtype != "float32":
            raise ValueError("formal E7 embedding cache must use float32")
        if self.r1_retriever_id != "r1_independent_dense":
            raise ValueError("R1 retriever identity is frozen")
        if self.r1_model_name != "BAAI/bge-base-en-v1.5":
            raise ValueError("R1 model is frozen to BAAI/bge-base-en-v1.5")
        if self.r1_model_revision != "a5beb1e3e68b9ab74eb54cfd186867f64f240e1a":
            raise ValueError("R1 model revision is frozen")
        if self.r1_query_instruction != (
            "Represent this sentence for searching relevant passages: "
        ):
            raise ValueError("R1 query instruction must remain the official prompt")
        if self.r1_document_instruction is not None:
            raise ValueError("R1 documents must not receive a prompt")
        if self.r1_pooling != "cls" or self.r1_max_length != 512:
            raise ValueError("R1 must use CLS pooling at max length 512")
        if self.r1_embedding_dimension != 768:
            raise ValueError("R1 embedding dimension must remain 768")
        if self.r1_normalization != "l2":
            raise ValueError("R1 embeddings must be L2-normalized")
        if self.r1_similarity_function != "dot_product":
            raise ValueError("R1 exact similarity must remain dot product")
        if self.r1_embedding_dtype != "float32":
            raise ValueError("R1 cache must remain float32")
        if self.r1_embedding_batch_size < 1:
            raise ValueError("R1 batch size must be positive")
        if self.r1_cache_namespace != "r1_independent":
            raise ValueError("R1 cache namespace is frozen")
        if self.pool_v2_corpus_sources != (
            "researchy_train",
            "rule_pool",
            "dev",
        ):
            raise ValueError("Pool V2 source composition is frozen")
        if self.pool_v2_cache_namespace != "r1_independent/pool_v2":
            raise ValueError("Pool V2 cache namespace is frozen")
        if self.diskann_l_multiplier < 1:
            raise ValueError("diskann_l_multiplier must be positive")
        if self.num_candidates_per_round < 1 or self.max_rounds < 1:
            raise ValueError("RA candidate count and round count must be positive")
        if self.retrieval_backend not in {
            "researchy_pooled_local",
            self.clueweb_retrieval_backend,
        }:
            raise ValueError("unsupported E7 retrieval backend")
        if self.dev_corpus_sources != ("rule_pool", "dev"):
            raise ValueError("DEV corpus sources must be rule_pool plus dev")
        if self.test_corpus_sources != ("rule_pool", "test"):
            raise ValueError("TEST corpus sources must be rule_pool plus test")
        if self.request_timeout_seconds <= 0:
            raise ValueError("request timeout must be positive")
        if self.researchy_download_max_retries < 1:
            raise ValueError("Researchy download retries must be positive")
        if self.researchy_retry_backoff_seconds < 0:
            raise ValueError("Researchy retry backoff cannot be negative")
        if not 1 <= self.smoke_query_limit <= 3:
            raise ValueError("smoke_query_limit must remain within 1--3")
        if not self.retrieval_model.strip() or not self.retrieval_corpus.strip():
            raise ValueError("retrieval model and corpus must be named")

    def to_dict(self) -> dict[str, Any]:
        result = asdict(self)
        for key in (
            "project_root",
            "data_root",
            "frozen_root",
            "rule_pool_query_dir",
            "dev_query_dir",
            "test_query_dir",
            "frozen_query_dir",
            "output_root",
        ):
            result[key] = str(result[key])
        result["diskann_l"] = self.diskann_l
        return result
