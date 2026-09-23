"""Model-independent frozen retriever identity and encoding contract."""

from __future__ import annotations

from dataclasses import asdict, dataclass


@dataclass(frozen=True)
class EncodingConfiguration:
    """Everything that can make query/document vectors differ."""

    instruction: str | None
    pooling: str
    max_length: int
    truncation: bool = True
    padding: str = "longest"

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True)
class FrozenRetrieverSpecification:
    """Serializable identity shared by every E7 retrieval consumer."""

    retriever_id: str
    model_name: str
    model_revision: str
    query_encoding_config: EncodingConfiguration
    document_encoding_config: EncodingConfiguration
    normalization: str
    similarity: str
    max_length: int
    embedding_dimension: int
    dtype: str

    def __post_init__(self) -> None:
        if not self.retriever_id.strip():
            raise ValueError("retriever_id cannot be empty")
        if self.max_length < 1 or self.embedding_dimension < 1:
            raise ValueError("retriever length and dimension must be positive")
        if self.query_encoding_config.max_length != self.max_length:
            raise ValueError("query max length disagrees with retriever max length")
        if self.document_encoding_config.max_length != self.max_length:
            raise ValueError("document max length disagrees with retriever max length")

    def to_dict(self) -> dict[str, object]:
        return asdict(self)
