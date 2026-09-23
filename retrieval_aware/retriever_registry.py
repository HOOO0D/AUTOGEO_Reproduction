"""Central lookup for frozen E7 retrievers; callers depend on IDs, not models."""

from __future__ import annotations

from typing import Callable

from .config import E7Config
from .frozen_local_retriever import (
    DenseEncoder,
    FrozenLocalCorpus,
    FrozenLocalRetriever,
    MiniCPMEmbeddingEncoder,
    load_embedding_cache,
)
from .independent_retriever import BGEBaseEnV15Encoder


def create_frozen_encoder(
    retriever_id: str,
    config: E7Config,
) -> DenseEncoder:
    factories: dict[str, Callable[[E7Config], DenseEncoder]] = {
        config.r0_retriever_id: MiniCPMEmbeddingEncoder,
        config.r1_retriever_id: BGEBaseEnV15Encoder,
    }
    try:
        factory = factories[retriever_id]
    except KeyError as error:
        raise ValueError(f"unknown frozen retriever_id: {retriever_id}") from error
    return factory(config)


def cache_namespace_for(retriever_id: str, config: E7Config) -> str | None:
    if retriever_id == config.r0_retriever_id:
        return None
    if retriever_id == config.r1_retriever_id:
        return config.r1_cache_namespace
    raise ValueError(f"unknown frozen retriever_id: {retriever_id}")


def load_frozen_retriever(
    retriever_id: str,
    corpus: FrozenLocalCorpus,
    config: E7Config,
) -> FrozenLocalRetriever:
    encoder = create_frozen_encoder(retriever_id, config)
    cache = load_embedding_cache(
        corpus,
        encoder.specification,
        config,
        cache_namespace=cache_namespace_for(retriever_id, config),
        retriever_specification=encoder.retriever_specification,
    )
    return FrozenLocalRetriever(encoder, cache)
