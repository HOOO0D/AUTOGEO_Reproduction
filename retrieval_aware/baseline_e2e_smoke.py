"""One-query DEV smoke for Original-E2E and stock AutoGEO_API-E2E only."""

from __future__ import annotations

import argparse
import hashlib
import json
import statistics
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Mapping, Sequence

from autogeo.evaluation.generative_engine import generate_answer_gemini
from autogeo.evaluation.metrics.geo_score import (
    extract_citations_new,
    impression_pos_count_simple,
    impression_word_count_simple,
    impression_wordpos_count_simple,
)
from autogeo.evaluation.metrics.geu_score import process_single_question
from autogeo.rewriters import rewrite_document as stock_rewrite_document
from autogeo.rewriters.core import _load_rules_from_file

from .config import E7Config
from .frozen_local_retriever import (
    FrozenLocalCorpus,
    FrozenLocalRetriever,
    RetrievalHit,
    load_embedding_cache,
    load_local_corpus,
)
from .independent_retriever import BGEBaseEnV15Encoder
from .protocol import ProtocolId
from .utils import (
    canonical_json_sha256,
    normalized_text_sha256,
    read_json,
    sha256_file,
    write_e7_json_once,
)


EXPECTED_TARGET_MANIFEST_SHA256 = (
    "4e3c118e03461f92f852b69f9f1ca0f699f324ad1a98b5381cb151ed420b17ca"
)
EXPECTED_POOL_V2_MANIFEST_SHA256 = (
    "423e4cd932222fca8e16ba47c230393f5c9a67a00b8c10f05a0292dbe27f55d8"
)
EXPECTED_POOL_V2_EMBEDDINGS_SHA256 = (
    "a2d53f643cf4a3e34e9082c7bb32aacf54dd9396951830500fa743469caacfef"
)
EXPECTED_POOL_V2_METADATA_SHA256 = (
    "3964d6fac25079eff963493273aad7245c91bcb0ee242bb3d43102111576fd63"
)
EXPECTED_RETRIEVER_CONFIG_SHA256 = (
    "6ac758e1ee097a0df3c4df14e8d3d1534a5374a6e933b4ea355158fb48477ba2"
)
EXPECTED_RULE_FILE_SHA256 = (
    "7aab609ca6be500e16d882c326c8ced3dd446dc9c1b9f4746b83339d3ab9df94"
)
EXPECTED_SMOKE_QUERY_ID = "90444"

AUTOGEO_DATASET = "Researchy-GEO"
GE_MODEL = "gemini-2.5-flash-lite"
AUTOGEO_REWRITE_MODEL = "gemini-2.5-pro"
GEU_METRICS = (
    "citation_quality",
    "quality_dimensions",
    "keypoint_coverage",
)


@dataclass(frozen=True)
class SmokeInputs:
    target_record: dict[str, object]
    corpus: FrozenLocalCorpus
    target_manifest_path: Path
    corpus_manifest_path: Path
    rules_path: Path
    rules_file_sha256: str
    filtered_rules_sha256: str
    original_rules_reference_sha256: str


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def choose_smoke_target(records: Sequence[Mapping[str, object]]) -> Mapping[str, object]:
    """Pre-register the original-target sample closest to that group's median rank."""
    candidates = [
        record
        for record in records
        if record.get("eligible") is True
        and record.get("target_source") == "original_target_id"
    ]
    if not candidates:
        raise ValueError("frozen manifest has no eligible original_target_id sample")
    ranks = [record.get("initial_r1_rank") for record in candidates]
    if not all(isinstance(rank, int) for rank in ranks):
        raise ValueError("original-target smoke candidates have invalid ranks")
    median_rank = statistics.median(ranks)
    return min(
        candidates,
        key=lambda record: (
            abs(record["initial_r1_rank"] - median_rank),
            str(record["query_id"]),
        ),
    )


def _load_rules(config: E7Config) -> tuple[Path, str, str, str]:
    rules_path = (
        config.data_root
        / AUTOGEO_DATASET
        / "rule_sets"
        / GE_MODEL
        / "merged_rules.json"
    ).resolve()
    reference_path = (
        config.project_root / "experiments" / "artifacts" / "E0" / "autogeo_rules.json"
    ).resolve()
    rules_file_sha256 = sha256_file(rules_path)
    reference_sha256 = sha256_file(reference_path)
    if rules_file_sha256 != EXPECTED_RULE_FILE_SHA256:
        raise ValueError("stock AutoGEO rules file SHA256 changed")
    if reference_sha256 != rules_file_sha256:
        raise ValueError("active rules differ from the original E0 AutoGEO rules")
    rules_payload = read_json(rules_path)
    filtered_rules = rules_payload.get("filtered_rules")
    if not isinstance(filtered_rules, list) or not filtered_rules:
        raise ValueError("stock AutoGEO rules lack filtered_rules")
    if not all(isinstance(rule, str) and rule for rule in filtered_rules):
        raise ValueError("filtered_rules must be non-empty strings")
    loaded_rules, loaded_path = _load_rules_from_file(
        AUTOGEO_DATASET,
        GE_MODEL,
        None,
    )
    if loaded_rules != filtered_rules or loaded_path is None:
        raise ValueError("stock AutoGEO automatic rule loading changed")
    if Path(loaded_path).resolve() != rules_path:
        raise ValueError("stock AutoGEO resolved an unexpected rules artifact")
    return (
        rules_path,
        rules_file_sha256,
        canonical_json_sha256(filtered_rules),
        reference_sha256,
    )


def load_smoke_inputs(config: E7Config | None = None) -> SmokeInputs:
    config = config or E7Config()
    config.validate()
    target_manifest_path = (
        config.output_paths["targets"] / "dev_targets_r1_pool_v2.frozen.json"
    )
    corpus_manifest_path = (
        config.output_paths["corpus"] / "pool_v2" / "dev_corpus_manifest.json"
    )
    if sha256_file(target_manifest_path) != EXPECTED_TARGET_MANIFEST_SHA256:
        raise ValueError("frozen DEV target manifest SHA256 changed")
    if sha256_file(corpus_manifest_path) != EXPECTED_POOL_V2_MANIFEST_SHA256:
        raise ValueError("frozen Pool V2 manifest SHA256 changed")

    target_manifest = read_json(target_manifest_path)
    if target_manifest.get("schema_version") != "e7_dev_targets_r1_pool_v2_frozen_v1":
        raise ValueError("unsupported frozen DEV target manifest")
    if target_manifest.get("status") != "frozen":
        raise ValueError("DEV target manifest is not frozen")
    if target_manifest.get("retriever_config_sha256") != (
        EXPECTED_RETRIEVER_CONFIG_SHA256
    ):
        raise ValueError("frozen R1 config SHA256 changed")
    if target_manifest.get("corpus_manifest_sha256") != (
        EXPECTED_POOL_V2_MANIFEST_SHA256
    ):
        raise ValueError("target manifest and Pool V2 manifest disagree")
    records = target_manifest.get("records")
    if not isinstance(records, list) or len(records) != 20:
        raise ValueError("frozen target manifest must contain DEV20")
    chosen = dict(choose_smoke_target(records))
    if chosen.get("query_id") != EXPECTED_SMOKE_QUERY_ID:
        raise ValueError("deterministic smoke sample changed")
    if chosen.get("target_source") != "original_target_id":
        raise ValueError("smoke sample must minimize wrapper variables")

    corpus = load_local_corpus(corpus_manifest_path)
    if len(corpus.documents) != 44_264 or corpus.corpus_split != "dev":
        raise ValueError("frozen DEV Pool V2 identity changed")
    target_id = chosen.get("target_document_id")
    if not isinstance(target_id, str):
        raise ValueError("smoke target lacks physical identity")
    target_document = corpus.documents[corpus.index_of(target_id)]
    if target_document.text_hash != chosen.get("target_text_hash"):
        raise ValueError("frozen target hash differs from Pool V2")
    if normalized_text_sha256(target_document.text) != target_document.text_hash:
        raise ValueError("frozen target text no longer matches its physical identity")

    rules_path, rules_sha, filtered_sha, reference_sha = _load_rules(config)
    return SmokeInputs(
        target_record=chosen,
        corpus=corpus,
        target_manifest_path=target_manifest_path.resolve(),
        corpus_manifest_path=corpus_manifest_path.resolve(),
        rules_path=rules_path,
        rules_file_sha256=rules_sha,
        filtered_rules_sha256=filtered_sha,
        original_rules_reference_sha256=reference_sha,
    )


def _cache_paths(config: E7Config) -> tuple[Path, Path]:
    directory = (
        config.output_paths["embeddings"]
        / config.pool_v2_cache_namespace
        / "dev_corpus_cache"
    )
    return directory / "embeddings.npy", directory / "metadata.json"


def _load_retriever(
    inputs: SmokeInputs,
    config: E7Config,
) -> FrozenLocalRetriever:
    embedding_path, metadata_path = _cache_paths(config)
    if sha256_file(embedding_path) != EXPECTED_POOL_V2_EMBEDDINGS_SHA256:
        raise ValueError("frozen Pool V2 embedding cache SHA256 changed")
    if sha256_file(metadata_path) != EXPECTED_POOL_V2_METADATA_SHA256:
        raise ValueError("frozen Pool V2 cache metadata SHA256 changed")
    encoder = BGEBaseEnV15Encoder(config)
    cache = load_embedding_cache(
        inputs.corpus,
        encoder.specification,
        config,
        cache_namespace=config.pool_v2_cache_namespace,
        retriever_specification=encoder.retriever_specification,
    )
    retriever_config_hash = canonical_json_sha256(cache.retriever.to_dict())
    if retriever_config_hash != EXPECTED_RETRIEVER_CONFIG_SHA256:
        raise ValueError("loaded R1 differs from the frozen target manifest")
    return FrozenLocalRetriever(encoder, cache)


def _persist_or_reuse(
    destination: Path,
    payload: dict[str, object],
    config: E7Config,
) -> tuple[str, str]:
    if destination.exists():
        if read_json(destination) != payload:
            raise FileExistsError(f"refusing to replace changed smoke artifact: {destination}")
        return "reused_identical", sha256_file(destination)
    write_e7_json_once(destination, payload, config)
    destination.chmod(0o444)
    return "created", sha256_file(destination)


def _validate_reused_identity(
    payload: Mapping[str, object],
    *,
    schema_version: str,
    inputs: SmokeInputs,
) -> None:
    target = inputs.target_record
    required = {
        "schema_version": schema_version,
        "query_id": target["query_id"],
        "target_document_id": target["target_document_id"],
        "target_text_hash": target["target_text_hash"],
        "target_manifest_sha256": EXPECTED_TARGET_MANIFEST_SHA256,
        "corpus_manifest_sha256": EXPECTED_POOL_V2_MANIFEST_SHA256,
        "retriever_config_sha256": EXPECTED_RETRIEVER_CONFIG_SHA256,
    }
    for key, value in required.items():
        if payload.get(key) != value:
            raise ValueError(f"reused smoke artifact has wrong {key}")


def rewrite_selected_target_with_stock_autogeo(
    selected_target_text: str,
    *,
    rules_path: Path,
    rewrite_callable: Callable[..., str] = stock_rewrite_document,
) -> str:
    """E7 adapter: pass only selected text and stock rules to unchanged AutoGEO."""
    rewritten = rewrite_callable(
        document=selected_target_text,
        dataset=AUTOGEO_DATASET,
        engine_llm=GE_MODEL,
        rule_path=str(rules_path),
    )
    if not isinstance(rewritten, str) or not rewritten.strip():
        raise ValueError("stock AutoGEO returned an empty rewrite")
    return rewritten


def run_rewrite_stage(config: E7Config | None = None) -> dict[str, object]:
    config = config or E7Config()
    inputs = load_smoke_inputs(config)
    target = inputs.target_record
    destination = (
        config.output_paths["rewrites"]
        / ProtocolId.AUTOGEO_API_E2E.value
        / f"{target['query_id']}.json"
    )
    if destination.exists():
        payload = read_json(destination)
        _validate_reused_identity(
            payload,
            schema_version="e7_autogeo_api_e2e_smoke_rewrite_v1",
            inputs=inputs,
        )
        if payload.get("filtered_rules_sha256") != inputs.filtered_rules_sha256:
            raise ValueError("reused rewrite uses different filtered_rules")
        return {
            "stage": "rewrite",
            "action": "reused_identical",
            "output_path": str(destination),
            "output_sha256": sha256_file(destination),
            "rewrite_text_sha256": payload["rewrite_text_sha256"],
        }

    target_document = inputs.corpus.documents[
        inputs.corpus.index_of(target["target_document_id"])
    ]
    rewritten_text = rewrite_selected_target_with_stock_autogeo(
        target_document.text,
        rules_path=inputs.rules_path,
    )
    rewrite_hash = normalized_text_sha256(rewritten_text)
    if rewrite_hash == target_document.text_hash:
        raise AssertionError("AutoGEO rewrite did not change the target text/hash")
    payload: dict[str, object] = {
        "schema_version": "e7_autogeo_api_e2e_smoke_rewrite_v1",
        "protocol_id": ProtocolId.AUTOGEO_API_E2E.value,
        "query_id": target["query_id"],
        "query": target["query"],
        "target_document_id": target["target_document_id"],
        "target_source": target["target_source"],
        "target_text_hash": target["target_text_hash"],
        "original_text_sha256": sha256_text(target_document.text),
        "rewritten_text": rewritten_text,
        "rewrite_text_sha256": rewrite_hash,
        "rewrite_raw_text_sha256": sha256_text(rewritten_text),
        "rewrite_backend": "stock_autogeo_api",
        "rewrite_model": AUTOGEO_REWRITE_MODEL,
        "rewrite_mode": "one_shot_no_retrieval_feedback",
        "dataset": AUTOGEO_DATASET,
        "engine_llm_for_rules": GE_MODEL,
        "rules_path": str(inputs.rules_path),
        "rules_file_sha256": inputs.rules_file_sha256,
        "filtered_rules_sha256": inputs.filtered_rules_sha256,
        "original_e0_rules_file_sha256": inputs.original_rules_reference_sha256,
        "target_manifest_sha256": EXPECTED_TARGET_MANIFEST_SHA256,
        "corpus_manifest_sha256": EXPECTED_POOL_V2_MANIFEST_SHA256,
        "retriever_config_sha256": EXPECTED_RETRIEVER_CONFIG_SHA256,
        "retrieval_rank_feedback_provided": False,
        "retrieval_score_feedback_provided": False,
        "iterative_selection_used": False,
        "ra_logic_used": False,
    }
    action, output_sha = _persist_or_reuse(destination, payload, config)
    return {
        "stage": "rewrite",
        "action": action,
        "output_path": str(destination),
        "output_sha256": output_sha,
        "rewrite_text_sha256": rewrite_hash,
    }


def build_method_top5_sources(
    corpus: FrozenLocalCorpus,
    hits: Sequence[RetrievalHit],
    *,
    target_document_id: str,
    replacement_target_text: str | None,
) -> tuple[tuple[str, ...], tuple[str, ...]]:
    """Materialize actual Top-5, overlaying only D* when it is present."""
    document_by_id = {document.document_id: document for document in corpus.documents}
    identities: list[str] = []
    sources: list[str] = []
    for hit in hits:
        document = document_by_id[hit.document_id]
        identities.append(hit.document_id)
        if hit.document_id == target_document_id and replacement_target_text is not None:
            sources.append(replacement_target_text)
        else:
            sources.append(document.text)
    return tuple(identities), tuple(sources)


def calculate_target_e2e_geo(
    response: str,
    target_source_index: int | None,
    *,
    ge_top_k: int = 5,
) -> dict[str, float]:
    """Apply the retrieval gate before reusing the stock GEO metrics."""
    if target_source_index is None:
        return {"word": 0.0, "pos": 0.0, "wordpos": 0.0}
    if not 0 <= target_source_index < ge_top_k:
        raise ValueError("dynamic target source index is outside Top-5")
    citations = extract_citations_new(response)
    return {
        "word": impression_word_count_simple(citations, n=ge_top_k)[
            target_source_index
        ],
        "pos": impression_pos_count_simple(citations, n=ge_top_k)[
            target_source_index
        ],
        "wordpos": impression_wordpos_count_simple(citations, n=ge_top_k)[
            target_source_index
        ],
    }


def _load_keypoints(query_id: str, config: E7Config) -> list[dict[str, object]] | None:
    path = config.data_root / AUTOGEO_DATASET / "key_point" / f"{query_id}_aggregated.json"
    if not path.is_file():
        return None
    payload = read_json(path)
    keypoints = payload.get("key_points")
    if not isinstance(keypoints, list) or not keypoints:
        return None
    return keypoints


def calculate_existing_geu(
    query_id: str,
    query: str,
    response: str,
    sources: Sequence[str],
    config: E7Config,
) -> tuple[dict[str, object], bool]:
    """Call the existing single-question GEU evaluator even on retrieval misses."""
    item: dict[str, object] = {
        "query": query,
        "ori_response": response,
        "text_list": list(sources),
    }
    keypoints = _load_keypoints(query_id, config)
    if keypoints is not None:
        item["keypoint_list"] = keypoints
    returned_id, scores = process_single_question(
        query_id,
        item,
        list(GEU_METRICS),
    )
    if returned_id != query_id or not isinstance(scores, dict) or not scores:
        raise ValueError("existing GEU evaluator returned no result")
    return scores, keypoints is not None


def _common_result_fields(inputs: SmokeInputs) -> dict[str, object]:
    target = inputs.target_record
    return {
        "query_id": target["query_id"],
        "query": target["query"],
        "target_document_id": target["target_document_id"],
        "target_source": target["target_source"],
        "target_text_hash": target["target_text_hash"],
        "target_manifest_sha256": EXPECTED_TARGET_MANIFEST_SHA256,
        "corpus_id": inputs.corpus.corpus_id,
        "corpus_size": len(inputs.corpus.documents),
        "corpus_manifest_sha256": EXPECTED_POOL_V2_MANIFEST_SHA256,
        "retriever_id": "r1_independent_dense",
        "retriever_config_sha256": EXPECTED_RETRIEVER_CONFIG_SHA256,
        "embedding_cache_sha256": EXPECTED_POOL_V2_EMBEDDINGS_SHA256,
        "ge_model": GE_MODEL,
        "ge_top_k": 5,
    }


def run_original_stage(config: E7Config | None = None) -> dict[str, object]:
    config = config or E7Config()
    inputs = load_smoke_inputs(config)
    target = inputs.target_record
    destination = (
        config.output_paths["evaluation"]
        / "smoke"
        / f"{target['query_id']}_original_e2e.json"
    )
    if destination.exists():
        payload = read_json(destination)
        _validate_reused_identity(
            payload,
            schema_version="e7_original_e2e_smoke_v1",
            inputs=inputs,
        )
        return {
            "stage": ProtocolId.ORIGINAL_E2E.value,
            "action": "reused_identical",
            "output_path": str(destination),
            "output_sha256": sha256_file(destination),
            "result": payload,
        }

    retriever = _load_retriever(inputs, config)
    full_ranking = retriever.retrieve(
        target["query"],
        inputs.corpus,
        top_k=len(inputs.corpus.documents),
    )
    target_hit = next(
        hit for hit in full_ranking if hit.document_id == target["target_document_id"]
    )
    if target_hit.rank != target["initial_r1_rank"]:
        raise AssertionError("Original-E2E rank differs from frozen initial_r1_rank")
    if target_hit.rank <= config.ge_top_k:
        raise AssertionError("frozen E7 target must begin outside GE Top-5")
    top5_ids, top5_sources = build_method_top5_sources(
        inputs.corpus,
        full_ranking[: config.ge_top_k],
        target_document_id=target["target_document_id"],
        replacement_target_text=None,
    )
    response = generate_answer_gemini(
        target["query"],
        list(top5_sources),
        model_name=GE_MODEL,
    )
    geu, keypoints_available = calculate_existing_geu(
        target["query_id"],
        target["query"],
        response,
        top5_sources,
        config,
    )
    embedding_path, metadata_path = _cache_paths(config)
    if sha256_file(embedding_path) != EXPECTED_POOL_V2_EMBEDDINGS_SHA256:
        raise AssertionError("Original-E2E modified the frozen embedding cache")
    if sha256_file(metadata_path) != EXPECTED_POOL_V2_METADATA_SHA256:
        raise AssertionError("Original-E2E modified cache metadata")
    payload: dict[str, object] = {
        "schema_version": "e7_original_e2e_smoke_v1",
        "protocol_id": ProtocolId.ORIGINAL_E2E.value,
        **_common_result_fields(inputs),
        "initial_r1_rank": target["initial_r1_rank"],
        "final_target_rank": target_hit.rank,
        "final_target_score": target_hit.score,
        "entered_top5": False,
        "top5_document_ids": list(top5_ids),
        "target_source_index_in_top5": None,
        "target_e2e_geo": {"word": 0.0, "pos": 0.0, "wordpos": 0.0},
        "ge_response": response,
        "geu": geu,
        "geu_metrics_requested": list(GEU_METRICS),
        "keypoints_available": keypoints_available,
        "ge_executed_despite_target_miss": True,
        "target_text_modified": False,
        "rewrite_executed": False,
        "ra_logic_used": False,
    }
    action, output_sha = _persist_or_reuse(destination, payload, config)
    return {
        "stage": ProtocolId.ORIGINAL_E2E.value,
        "action": action,
        "output_path": str(destination),
        "output_sha256": output_sha,
        "result": payload,
    }


def run_autogeo_stage(config: E7Config | None = None) -> dict[str, object]:
    config = config or E7Config()
    inputs = load_smoke_inputs(config)
    target = inputs.target_record
    rewrite_path = (
        config.output_paths["rewrites"]
        / ProtocolId.AUTOGEO_API_E2E.value
        / f"{target['query_id']}.json"
    )
    if not rewrite_path.is_file():
        raise FileNotFoundError("run the one-shot AutoGEO rewrite stage first")
    rewrite = read_json(rewrite_path)
    _validate_reused_identity(
        rewrite,
        schema_version="e7_autogeo_api_e2e_smoke_rewrite_v1",
        inputs=inputs,
    )
    if rewrite.get("filtered_rules_sha256") != inputs.filtered_rules_sha256:
        raise ValueError("rewrite filtered_rules differ from stock AutoGEO")
    rewritten_text = rewrite.get("rewritten_text")
    if not isinstance(rewritten_text, str) or not rewritten_text.strip():
        raise ValueError("rewrite artifact lacks rewritten text")
    if normalized_text_sha256(rewritten_text) != rewrite.get("rewrite_text_sha256"):
        raise ValueError("rewrite artifact text hash mismatch")

    destination = (
        config.output_paths["evaluation"]
        / "smoke"
        / f"{target['query_id']}_autogeo_api_e2e.json"
    )
    if destination.exists():
        payload = read_json(destination)
        _validate_reused_identity(
            payload,
            schema_version="e7_autogeo_api_e2e_smoke_v1",
            inputs=inputs,
        )
        if payload.get("rewrite_text_sha256") != rewrite["rewrite_text_sha256"]:
            raise ValueError("reused AutoGEO evaluation has a different rewrite")
        return {
            "stage": ProtocolId.AUTOGEO_API_E2E.value,
            "action": "reused_identical",
            "output_path": str(destination),
            "output_sha256": sha256_file(destination),
            "result": payload,
        }

    target_index = inputs.corpus.index_of(target["target_document_id"])
    target_document = inputs.corpus.documents[target_index]
    corpus_identity_before = inputs.corpus.document_ids
    corpus_hashes_before = inputs.corpus.text_hashes
    retriever = _load_retriever(inputs, config)
    result = retriever.rerank_with_rewritten_target(
        target["query"],
        inputs.corpus,
        target["target_document_id"],
        rewritten_text,
        top_k=config.ge_top_k,
    )
    if result.original_target_rank != target["initial_r1_rank"]:
        raise AssertionError("AutoGEO sibling has a different original target rank")
    if result.target_id != target["target_document_id"]:
        raise AssertionError("Original and AutoGEO must use the same target identity")
    if result.reencoded_document_ids != (target["target_document_id"],):
        raise AssertionError("counterfactual re-index encoded more than target D*")
    if not result.competitor_invariant_verified:
        raise AssertionError("N-1 competitor invariant was not verified")
    if inputs.corpus.document_ids != corpus_identity_before:
        raise AssertionError("AutoGEO changed Pool V2 physical identities")
    if inputs.corpus.text_hashes != corpus_hashes_before:
        raise AssertionError("AutoGEO changed frozen Pool V2 text hashes")
    if inputs.corpus.documents[target_index] != target_document:
        raise AssertionError("counterfactual rewrite mutated the frozen target document")

    top5_ids, top5_sources = build_method_top5_sources(
        inputs.corpus,
        result.new_top_k,
        target_document_id=target["target_document_id"],
        replacement_target_text=rewritten_text,
    )
    target_source_index = (
        top5_ids.index(target["target_document_id"])
        if target["target_document_id"] in top5_ids
        else None
    )
    entered_top5 = result.new_target_rank <= config.ge_top_k
    if entered_top5 != (target_source_index is not None):
        raise AssertionError("new rank and method-specific Top-5 disagree")
    if target_source_index is not None and target_source_index != result.new_target_rank - 1:
        raise AssertionError("dynamic target source index is not its new R1 position")
    response = generate_answer_gemini(
        target["query"],
        list(top5_sources),
        model_name=GE_MODEL,
    )
    target_geo = calculate_target_e2e_geo(
        response,
        target_source_index,
        ge_top_k=config.ge_top_k,
    )
    geu, keypoints_available = calculate_existing_geu(
        target["query_id"],
        target["query"],
        response,
        top5_sources,
        config,
    )
    embedding_path, metadata_path = _cache_paths(config)
    if sha256_file(embedding_path) != EXPECTED_POOL_V2_EMBEDDINGS_SHA256:
        raise AssertionError("AutoGEO counterfactual modified embedding cache")
    if sha256_file(metadata_path) != EXPECTED_POOL_V2_METADATA_SHA256:
        raise AssertionError("AutoGEO counterfactual modified cache metadata")
    payload: dict[str, object] = {
        "schema_version": "e7_autogeo_api_e2e_smoke_v1",
        "protocol_id": ProtocolId.AUTOGEO_API_E2E.value,
        **_common_result_fields(inputs),
        "initial_r1_rank": target["initial_r1_rank"],
        "initial_r1_score": target["initial_r1_score"],
        "rewrite_text_sha256": rewrite["rewrite_text_sha256"],
        "rewrite_artifact_sha256": sha256_file(rewrite_path),
        "filtered_rules_sha256": inputs.filtered_rules_sha256,
        "rules_file_sha256": inputs.rules_file_sha256,
        "new_target_rank": result.new_target_rank,
        "new_target_score": result.new_target_score,
        "rank_delta": target["initial_r1_rank"] - result.new_target_rank,
        "rank_delta_definition": "initial_rank_minus_new_rank_positive_is_promotion",
        "entered_top5": entered_top5,
        "top5_document_ids": list(top5_ids),
        "target_source_index_in_top5": target_source_index,
        "target_e2e_geo": target_geo,
        "ge_response": response,
        "geu": geu,
        "geu_metrics_requested": list(GEU_METRICS),
        "keypoints_available": keypoints_available,
        "ge_executed_despite_target_miss": not entered_top5,
        "reencoded_document_ids": list(result.reencoded_document_ids),
        "competitor_invariant_verified": result.competitor_invariant_verified,
        "target_text_modified": True,
        "retrieval_rank_feedback_provided_to_rewriter": False,
        "retrieval_score_feedback_provided_to_rewriter": False,
        "iterative_selection_used": False,
        "ra_logic_used": False,
    }
    action, output_sha = _persist_or_reuse(destination, payload, config)
    return {
        "stage": ProtocolId.AUTOGEO_API_E2E.value,
        "action": action,
        "output_path": str(destination),
        "output_sha256": output_sha,
        "result": payload,
    }


def run_summary_stage(config: E7Config | None = None) -> dict[str, object]:
    config = config or E7Config()
    inputs = load_smoke_inputs(config)
    query_id = inputs.target_record["query_id"]
    rewrite_path = (
        config.output_paths["rewrites"]
        / ProtocolId.AUTOGEO_API_E2E.value
        / f"{query_id}.json"
    )
    original_path = (
        config.output_paths["evaluation"]
        / "smoke"
        / f"{query_id}_original_e2e.json"
    )
    autogeo_path = (
        config.output_paths["evaluation"]
        / "smoke"
        / f"{query_id}_autogeo_api_e2e.json"
    )
    for path in (rewrite_path, original_path, autogeo_path):
        if not path.is_file():
            raise FileNotFoundError(f"smoke stage artifact is missing: {path}")
    rewrite = read_json(rewrite_path)
    original = read_json(original_path)
    autogeo = read_json(autogeo_path)
    if original["target_document_id"] != autogeo["target_document_id"]:
        raise AssertionError("Original and AutoGEO smoke results use different targets")
    payload: dict[str, object] = {
        "schema_version": "e7_baseline_e2e_smoke_summary_v1",
        "query_id": query_id,
        "query": inputs.target_record["query"],
        "target_document_id": inputs.target_record["target_document_id"],
        "target_source": inputs.target_record["target_source"],
        "target_text_hash": inputs.target_record["target_text_hash"],
        "sample_selection": {
            "eligible_scope": "original_target_id_only",
            "criterion": "closest_initial_rank_to_group_median_then_query_id",
            "preselected_before_rewrite": True,
        },
        "original_e2e": {
            "initial_rank": original["initial_r1_rank"],
            "final_rank": original["final_target_rank"],
            "entered_top5": original["entered_top5"],
            "top5_document_ids": original["top5_document_ids"],
            "target_source_index_in_top5": original[
                "target_source_index_in_top5"
            ],
            "target_e2e_geo": original["target_e2e_geo"],
            "geu": original["geu"],
            "result_sha256": sha256_file(original_path),
        },
        "autogeo_api_e2e": {
            "rewrite_text_sha256": rewrite["rewrite_text_sha256"],
            "new_target_rank": autogeo["new_target_rank"],
            "rank_delta": autogeo["rank_delta"],
            "entered_top5": autogeo["entered_top5"],
            "top5_document_ids": autogeo["top5_document_ids"],
            "target_source_index_in_top5": autogeo[
                "target_source_index_in_top5"
            ],
            "target_e2e_geo": autogeo["target_e2e_geo"],
            "geu": autogeo["geu"],
            "rewrite_sha256": sha256_file(rewrite_path),
            "result_sha256": sha256_file(autogeo_path),
        },
        "invariants": {
            "same_query_target_pool_r1_ge_geo_geu": True,
            "original_rank_matches_frozen_manifest": True,
            "only_target_reencoded": autogeo["reencoded_document_ids"]
            == [inputs.target_record["target_document_id"]],
            "competitor_invariant_verified": autogeo[
                "competitor_invariant_verified"
            ],
            "filtered_rules_match_original_autogeo": (
                rewrite["rules_file_sha256"]
                == rewrite["original_e0_rules_file_sha256"]
            ),
            "ra_implemented_or_used": False,
        },
        "target_manifest_sha256": EXPECTED_TARGET_MANIFEST_SHA256,
        "corpus_manifest_sha256": EXPECTED_POOL_V2_MANIFEST_SHA256,
        "retriever_config_sha256": EXPECTED_RETRIEVER_CONFIG_SHA256,
        "test50_processed": False,
    }
    destination = (
        config.output_paths["evaluation"]
        / "smoke"
        / f"{query_id}_baseline_e2e_summary.json"
    )
    action, output_sha = _persist_or_reuse(destination, payload, config)
    return {
        "stage": "summary",
        "action": action,
        "output_path": str(destination),
        "output_sha256": output_sha,
        "result": payload,
    }


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--stage",
        required=True,
        choices=("rewrite", "original", "autogeo", "summary"),
    )
    return parser


def main() -> None:
    stage = _build_parser().parse_args().stage
    runners = {
        "rewrite": run_rewrite_stage,
        "original": run_original_stage,
        "autogeo": run_autogeo_stage,
        "summary": run_summary_stage,
    }
    print(json.dumps(runners[stage](), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
