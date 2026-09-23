"""Cached two-run downstream E2E evaluation for frozen E7 DEV20 artifacts."""

from __future__ import annotations

import argparse
import concurrent.futures
import json
import os
import statistics
import tempfile
from pathlib import Path
from typing import Any, Mapping, Sequence

from autogeo.evaluation.generative_engine import generate_answer_gemini, query_prompt
from autogeo.evaluation.metrics.geu_score import process_single_question
from autogeo.utils.constants import MAX_RETRIES, RETRY_DELAY_SECONDS
from autogeo.utils.openrouter import normalize_model_name

from .baseline_e2e_smoke import GEU_METRICS, GE_MODEL, calculate_target_e2e_geo, sha256_text
from .config import E7Config
from .dev20_retrieval_evaluation import (
    EXPECTED_FILTERED_RULES_SHA256,
    EXPECTED_POOL_V2_MANIFEST_SHA256,
    EXPECTED_RETRIEVER_CONFIG_SHA256,
    EXPECTED_TARGET_MANIFEST_SHA256,
    METHOD_IDS,
    SMOKE_QUERY_ID,
    _load_dev_inputs,
)
from .utils import (
    assert_e7_output_path,
    canonical_json_sha256,
    normalized_text_sha256,
    read_json,
    sha256_file,
    write_e7_json_once,
)

EXPECTED_RETRIEVAL_JSONL_SHA256 = "0ccafdf29f2c6af2d0432dcad49d85e85976d5d28654c8ea9c1ab5188dfa6eba"
EXPECTED_RETRIEVAL_SUMMARY_SHA256 = "0c744a39bf24b2bd7013761e01f5622e3d04be1e9d55335b88d335cacdf8bc93"
EXPECTED_RETRIEVAL_IMPLEMENTATION_SHA256 = "4527b452df3b29c726e79fa1488902bf910cc24027034a5f476f18f5e5f91779"
EXPECTED_GENERATIVE_ENGINE_SHA256 = "02e38dc2bd2d4b8b07cd739907b4a72f5770e778ebfc64074acdbd46517e6106"
EXPECTED_GEO_IMPLEMENTATION_SHA256 = "3c12623f138dec50c1c905c92c09aef1f3fe9b0a219a6ff6ffaa404007711b9d"
EXPECTED_GEU_IMPLEMENTATION_SHA256 = "9fb423d8a7ea45ee0a5effd9e6c1eba9624be7d9e3670dab7455c906168d7087"
EXPECTED_OPENROUTER_IMPLEMENTATION_SHA256 = "109c9af9f71eb238c18a22c8128458842038215b2b0b610004d05834c20d747b"
EXPECTED_GE_PROMPT_SHA256 = "697fa43a6582c62decd1b8d7a586ad945b0c32553ba53cbb50f88e71096574f9"
GE_RUNS_PER_UNIQUE_INPUT = 2
GEU_MAX_PARALLEL_OUTPUTS = 5
GEU_SCORE_KEYS = (
    "Precision", "Recall", "Clarity", "Depth", "Balance", "Breadth", "Support",
    "Insightfulness", "KPR", "KPC",
)


def _root(config: E7Config) -> Path:
    return config.output_paths["evaluation"] / "dev20_e2e"


def _contract_path(config: E7Config) -> Path:
    return _root(config) / "ge_evaluation_contract.frozen.json"


def _input_manifest_path(config: E7Config) -> Path:
    return _root(config) / "ge_input_manifest.frozen.json"


def _input_path(config: E7Config, input_hash: str) -> Path:
    return _root(config) / "ge_inputs" / f"{input_hash}.json"


def _raw_path(config: E7Config, input_hash: str, run_index: int) -> Path:
    return _root(config) / "ge_cache" / input_hash / f"run_{run_index}.json"


def _geu_path(config: E7Config, input_hash: str, run_index: int) -> Path:
    return _root(config) / "geu_cache" / input_hash / f"run_{run_index}.json"


def _retrieval_root(config: E7Config) -> Path:
    return config.output_paths["evaluation"] / "dev20_retrieval"


def _persist_json(path: Path, payload: Any, config: E7Config) -> tuple[str, str]:
    if path.exists():
        if read_json(path) != payload:
            raise FileExistsError(f"refusing to replace changed E2E artifact: {path}")
        return "reused_identical", sha256_file(path)
    write_e7_json_once(path, payload, config)
    path.chmod(0o444)
    return "created", sha256_file(path)


def _persist_text(path: Path, text: str, config: E7Config) -> tuple[str, str]:
    destination = assert_e7_output_path(path, config)
    if destination.exists():
        if destination.read_text(encoding="utf-8") != text:
            raise FileExistsError(f"refusing to replace changed E2E artifact: {path}")
        return "reused_identical", sha256_file(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w", encoding="utf-8", dir=destination.parent,
            prefix=f".{destination.name}.", suffix=".tmp", delete=False,
        ) as handle:
            temporary = handle.name
            handle.write(text)
        os.replace(temporary, destination)
    finally:
        if temporary and Path(temporary).exists():
            Path(temporary).unlink()
    destination.chmod(0o444)
    return "created", sha256_file(destination)


def build_ge_contract(config: E7Config | None = None) -> dict[str, Any]:
    config = config or E7Config()
    source_hashes = {
        "generative_engine.py": EXPECTED_GENERATIVE_ENGINE_SHA256,
        "geo_score.py": EXPECTED_GEO_IMPLEMENTATION_SHA256,
        "geu_score.py": EXPECTED_GEU_IMPLEMENTATION_SHA256,
        "openrouter.py": EXPECTED_OPENROUTER_IMPLEMENTATION_SHA256,
        "dev20_retrieval_evaluation.py": EXPECTED_RETRIEVAL_IMPLEMENTATION_SHA256,
    }
    source_paths = {
        "generative_engine.py": config.project_root / "autogeo/evaluation/generative_engine.py",
        "geo_score.py": config.project_root / "autogeo/evaluation/metrics/geo_score.py",
        "geu_score.py": config.project_root / "autogeo/evaluation/metrics/geu_score.py",
        "openrouter.py": config.project_root / "autogeo/utils/openrouter.py",
        "dev20_retrieval_evaluation.py": config.project_root / "retrieval_aware/dev20_retrieval_evaluation.py",
    }
    for name, path in source_paths.items():
        if sha256_file(path) != source_hashes[name]:
            raise ValueError(f"frozen E2E implementation changed: {name}")
    if sha256_text(query_prompt) != EXPECTED_GE_PROMPT_SHA256:
        raise ValueError("GE prompt changed")
    return {
        "schema_version": "e7_dev20_ge_evaluation_contract_v1",
        "status": "frozen",
        "ge": {
            "client": "OpenRouter OpenAI-compatible chat.completions",
            "base_url": "https://openrouter.ai/api/v1",
            "requested_model": GE_MODEL,
            "resolved_model": normalize_model_name(GE_MODEL),
            "model_revision": None,
            "provider_routing": None,
            "prompt_template_sha256": EXPECTED_GE_PROMPT_SHA256,
            "source_numbering": "zero_based_Source_0_to_Source_4",
            "temperature": None,
            "max_tokens": None,
            "system_prompt": None,
            "response_format": None,
            "seed": None,
            "client_timeout_seconds": 120.0,
            "max_retries": MAX_RETRIES,
            "retry_delay_seconds": RETRY_DELAY_SECONDS,
            "runs_per_unique_input": GE_RUNS_PER_UNIQUE_INPUT,
        },
        "cache": {
            "key_fields": ["query", "ordered_top5_source_texts", "ge_contract_sha256"],
            "hash": "canonical_json_sha256",
            "identical_input_shared_across_methods": True,
            "raw_outputs_immutable": True,
        },
        "geo": {
            "metrics": ["word", "pos", "wordpos"],
            "citation_indexing": "zero_based_dynamic_method_specific_top5",
            "target_rank_above_5": {"word": 0.0, "pos": 0.0, "wordpos": 0.0},
        },
        "geu": {
            "implementation": "process_single_question",
            "metrics_requested": list(GEU_METRICS),
            "judge_model": "gpt-4o-mini",
            "resolved_judge_model": normalize_model_name("gpt-4o-mini"),
            "temperature": 0.0,
            "response_format": {"type": "json_object"},
            "parser_retries": 3,
            "per_api_call_retries": 1,
            "max_parallel_unique_outputs": GEU_MAX_PARALLEL_OUTPUTS,
            "missing_keypoints": {"KPR": None, "KPC": None},
            "identical_raw_output_evaluation_shared_across_methods": True,
        },
        "source_file_sha256": source_hashes,
        "retrieval_artifacts": {
            "per_query_sha256": EXPECTED_RETRIEVAL_JSONL_SHA256,
            "summary_sha256": EXPECTED_RETRIEVAL_SUMMARY_SHA256,
            "target_manifest_sha256": EXPECTED_TARGET_MANIFEST_SHA256,
            "corpus_manifest_sha256": EXPECTED_POOL_V2_MANIFEST_SHA256,
            "retriever_config_sha256": EXPECTED_RETRIEVER_CONFIG_SHA256,
            "filtered_rules_sha256": EXPECTED_FILTERED_RULES_SHA256,
        },
        "test50_processed": False,
        "rewrite_calls_allowed": False,
    }


def freeze_contract(config: E7Config | None = None) -> dict[str, Any]:
    config = config or E7Config()
    body = build_ge_contract(config)
    payload = {**body, "contract_sha256": canonical_json_sha256(body)}
    action, digest = _persist_json(_contract_path(config), payload, config)
    return {"stage": "freeze_contract", "action": action, "output_sha256": digest, "result": payload}


def _validate_retrieval_outputs(config: E7Config) -> list[dict[str, Any]]:
    jsonl_path = _retrieval_root(config) / "dev20_retrieval_per_query.jsonl"
    summary_path = _retrieval_root(config) / "dev20_retrieval_summary.json"
    if sha256_file(jsonl_path) != EXPECTED_RETRIEVAL_JSONL_SHA256:
        raise ValueError("frozen DEV20 retrieval per-query artifact changed")
    if sha256_file(summary_path) != EXPECTED_RETRIEVAL_SUMMARY_SHA256:
        raise ValueError("frozen DEV20 retrieval summary changed")
    rows = [json.loads(line) for line in jsonl_path.read_text(encoding="utf-8").splitlines()]
    if len(rows) != 20 or len({row["query_id"] for row in rows}) != 20:
        raise ValueError("retrieval per-query artifact must contain unique DEV20")
    if sum(row["eligible"] is True for row in rows) != 15:
        raise ValueError("retrieval eligibility changed")
    return rows


def derive_ra_top5(original_top5: Sequence[str], target_id: str, final_rank: int) -> list[str]:
    """Derive target-only counterfactual Top-5 from frozen rank and competitor order."""
    if len(original_top5) != 5 or target_id in original_top5:
        raise ValueError("eligible original target must begin outside the original Top-5")
    if final_rank <= 5:
        top5 = [*original_top5[: final_rank - 1], target_id, *original_top5[final_rank - 1 : 4]]
    else:
        top5 = list(original_top5)
    if len(top5) != 5 or len(set(top5)) != 5:
        raise AssertionError("derived RA Top-5 is invalid")
    return top5


def ge_content_key(query: str, ordered_sources: Sequence[str], contract_hash: str) -> str:
    return canonical_json_sha256({
        "query": query,
        "ordered_top5_source_texts": list(ordered_sources),
        "ge_contract_sha256": contract_hash,
    })


def _method_rewrite(config: E7Config, query_id: str, method: str) -> str | None:
    rewrite_root = config.output_paths["rewrites"] / "dev20_retrieval"
    if method == "original_e2e":
        return None
    if method == "autogeo_api_e2e":
        path = rewrite_root / method / query_id / "rewrite.json"
    elif method == "query_aware_autogeo":
        path = rewrite_root / method / query_id / "rewrite.json"
    elif method == "no_feedback_multi_sample":
        path = rewrite_root / method / query_id / "C1.json"
    elif method == "ra_autogeo_e2e":
        result_path = _retrieval_root(config) / "per_query" / f"{query_id}.ra.json"
        result = read_json(result_path)
        source = result["selected_source"]
        if source == "original":
            return None
        if query_id == SMOKE_QUERY_ID:
            path = config.output_paths["rewrites"] / "ra_autogeo_e2e" / query_id / "candidates" / f"{source}.json"
        else:
            path = rewrite_root / method / query_id / "raw_candidates" / f"{source}.json"
    else:
        raise ValueError(f"unsupported E2E method: {method}")
    payload = read_json(path)
    text = payload.get("rewritten_text")
    if not isinstance(text, str) or not text.strip():
        raise ValueError(f"frozen rewrite missing for {query_id}/{method}")
    expected_hash = payload.get("rewrite_hash") or payload.get("rewrite_text_sha256")
    if normalized_text_sha256(text) != expected_hash:
        raise ValueError(f"frozen rewrite hash mismatch for {query_id}/{method}")
    return text


def prepare_inputs(config: E7Config | None = None) -> dict[str, Any]:
    config = config or E7Config()
    if not _contract_path(config).is_file():
        raise FileNotFoundError("freeze GE contract before preparing inputs")
    contract = read_json(_contract_path(config))
    if canonical_json_sha256({k: v for k, v in contract.items() if k != "contract_sha256"}) != contract["contract_sha256"]:
        raise ValueError("frozen GE contract self-hash mismatch")
    destination = _input_manifest_path(config)
    if destination.exists():
        return {"stage": "prepare_inputs", "action": "reused_identical", "result": read_json(destination)}
    inputs = _load_dev_inputs(config)
    rows = _validate_retrieval_outputs(config)
    document_by_id = {doc.document_id: doc for doc in inputs.corpus.documents}
    unique: dict[str, dict[str, Any]] = {}
    consumers: list[dict[str, Any]] = []
    for row in rows:
        if row["eligible"] is not True:
            continue
        query_id = str(row["query_id"])
        target_id = row["target_document_id"]
        original_top5 = row["methods"]["original_e2e"]["top5_document_ids"]
        for method in METHOD_IDS:
            result = row["methods"][method]
            if method == "ra_autogeo_e2e":
                top5_ids = derive_ra_top5(original_top5, target_id, int(result["final_rank"]))
            else:
                top5_ids = list(result["top5_document_ids"])
            target_index = top5_ids.index(target_id) if target_id in top5_ids else None
            entered = int(result["final_rank"]) <= 5
            if entered != (target_index is not None):
                raise AssertionError(f"rank/Top-5 mismatch for {query_id}/{method}")
            if target_index is not None and target_index != int(result["final_rank"]) - 1:
                raise AssertionError(f"dynamic target index mismatch for {query_id}/{method}")
            replacement = _method_rewrite(config, query_id, method)
            sources = []
            for document_id in top5_ids:
                if document_id == target_id and replacement is not None:
                    sources.append(replacement)
                else:
                    sources.append(document_by_id[document_id].text)
            input_hash = ge_content_key(row["query"], sources, contract["contract_sha256"])
            input_payload = {
                "schema_version": "e7_dev20_ge_input_v1",
                "ge_input_sha256": input_hash,
                "ge_contract_sha256": contract["contract_sha256"],
                "query": row["query"],
                "ordered_top5_source_texts": sources,
                "ordered_source_text_sha256": [sha256_text(text) for text in sources],
            }
            if input_hash in unique and unique[input_hash] != input_payload:
                raise AssertionError("GE content hash collision")
            unique[input_hash] = input_payload
            consumers.append({
                "query_id": query_id,
                "method": method,
                "ge_input_sha256": input_hash,
                "top5_document_ids": top5_ids,
                "target_document_id": target_id,
                "target_source_index_in_top5": target_index,
                "final_target_rank": result["final_rank"],
                "final_target_score": result["final_score"],
                "entered_top5": entered,
            })
    if len(consumers) != 75 or len(unique) != 57:
        raise AssertionError(f"unexpected GE input counts: {len(consumers)} consumers/{len(unique)} unique")
    input_hashes = {}
    for input_hash, payload in sorted(unique.items()):
        _, digest = _persist_json(_input_path(config, input_hash), payload, config)
        input_hashes[input_hash] = digest
    keypoints_available = []
    for row in rows:
        if row["eligible"] is True:
            kp_path = config.data_root / "Researchy-GEO" / "key_point" / f"{row['query_id']}_aggregated.json"
            if kp_path.is_file():
                keypoints_available.append(row["query_id"])
    if keypoints_available:
        raise AssertionError("DEV20 keypoint availability changed; contract expects missing KPR/KPC")
    payload = {
        "schema_version": "e7_dev20_ge_input_manifest_v1",
        "status": "frozen",
        "ge_contract_sha256": contract["contract_sha256"],
        "method_query_consumers": 75,
        "unique_ge_inputs": 57,
        "deduplicated_consumers": 18,
        "runs_per_unique_input": 2,
        "expected_raw_generations": 114,
        "consumers": consumers,
        "input_artifact_sha256": input_hashes,
        "eligible_queries": 15,
        "ineligible_queries": 5,
        "keypoints_available_queries": [],
        "missing_keypoints_policy": {"KPR": None, "KPC": None},
        "rewrite_calls_made": False,
        "test50_processed": False,
    }
    action, digest = _persist_json(destination, payload, config)
    return {"stage": "prepare_inputs", "action": action, "output_sha256": digest, "result": payload}


def _load_generation_prerequisites(config: E7Config) -> tuple[dict[str, Any], dict[str, Any]]:
    if not _contract_path(config).is_file() or not _input_manifest_path(config).is_file():
        raise FileNotFoundError("freeze contract and prepare GE inputs first")
    contract = read_json(_contract_path(config))
    manifest = read_json(_input_manifest_path(config))
    if manifest["ge_contract_sha256"] != contract["contract_sha256"]:
        raise ValueError("GE input manifest uses another contract")
    return contract, manifest


def generate_ge(config: E7Config | None = None) -> dict[str, Any]:
    config = config or E7Config()
    contract, manifest = _load_generation_prerequisites(config)
    destination = _root(config) / "ge_generation_manifest.json"
    if destination.exists():
        return {"stage": "generate_ge", "action": "reused_identical", "result": read_json(destination)}
    raw_hashes = {}
    for input_hash in sorted(manifest["input_artifact_sha256"]):
        ge_input = read_json(_input_path(config, input_hash))
        for run_index in range(1, GE_RUNS_PER_UNIQUE_INPUT + 1):
            path = _raw_path(config, input_hash, run_index)
            key = f"{input_hash}:run_{run_index}"
            if path.exists():
                existing = read_json(path)
                if (
                    existing.get("ge_input_sha256") != input_hash
                    or existing.get("run_index") != run_index
                    or existing.get("ge_contract_sha256") != contract["contract_sha256"]
                    or sha256_text(existing.get("raw_output", "")) != existing.get("raw_output_sha256")
                ):
                    raise ValueError(f"changed GE cache artifact: {path}")
                raw_hashes[key] = sha256_file(path)
                continue
            response = generate_answer_gemini(
                ge_input["query"], ge_input["ordered_top5_source_texts"], model_name=GE_MODEL
            )
            if not isinstance(response, str) or not response.strip():
                raise ValueError("GE returned an empty answer")
            payload = {
                "schema_version": "e7_dev20_ge_raw_output_v1",
                "ge_input_sha256": input_hash,
                "ge_contract_sha256": contract["contract_sha256"],
                "run_index": run_index,
                "raw_output": response,
                "raw_output_sha256": sha256_text(response),
                "model": contract["ge"]["resolved_model"],
                "provider_routing": None,
                "temperature": None,
            }
            _, raw_hashes[key] = _persist_json(path, payload, config)
            print(f"GE {len(raw_hashes)}/114 {input_hash[:10]} run {run_index}", flush=True)
    if len(raw_hashes) != 114:
        raise AssertionError("GE generation cache is incomplete")
    payload = {
        "schema_version": "e7_dev20_ge_generation_manifest_v1",
        "ge_contract_sha256": contract["contract_sha256"],
        "unique_ge_inputs": 57,
        "runs_per_unique_input": 2,
        "raw_generation_count": 114,
        "raw_artifact_sha256": raw_hashes,
        "rewrite_calls_made": False,
        "test50_processed": False,
    }
    action, digest = _persist_json(destination, payload, config)
    return {"stage": "generate_ge", "action": action, "output_sha256": digest, "result": payload}


def _evaluate_one_geu(config: E7Config, input_hash: str, run_index: int, contract_hash: str) -> dict[str, Any]:
    ge_input = read_json(_input_path(config, input_hash))
    raw_path = _raw_path(config, input_hash, run_index)
    raw = read_json(raw_path)
    item = {
        "query": ge_input["query"],
        "ori_response": raw["raw_output"],
        "text_list": ge_input["ordered_top5_source_texts"],
    }
    returned, scores = process_single_question(
        f"{input_hash}_run_{run_index}", item, list(GEU_METRICS)
    )
    if returned != f"{input_hash}_run_{run_index}" or not isinstance(scores, dict):
        raise ValueError("GEU evaluator returned invalid identity/result")
    normalized_scores = {key: scores.get(key) for key in GEU_SCORE_KEYS}
    if normalized_scores["KPR"] is not None or normalized_scores["KPC"] is not None:
        raise AssertionError("missing keypoints must remain null")
    for key in GEU_SCORE_KEYS[:-2]:
        value = normalized_scores[key]
        if not isinstance(value, (int, float)):
            raise RuntimeError(f"GEU core metric {key} failed for {input_hash}/run_{run_index}")
    return {
        "schema_version": "e7_dev20_geu_output_v1",
        "ge_input_sha256": input_hash,
        "ge_contract_sha256": contract_hash,
        "run_index": run_index,
        "raw_output_sha256": raw["raw_output_sha256"],
        "raw_output_artifact_sha256": sha256_file(raw_path),
        "scores": normalized_scores,
        "keypoints_available": False,
        "KPR": None,
        "KPC": None,
    }


def evaluate_geu(config: E7Config | None = None) -> dict[str, Any]:
    config = config or E7Config()
    contract, manifest = _load_generation_prerequisites(config)
    if not (_root(config) / "ge_generation_manifest.json").is_file():
        raise FileNotFoundError("generate all GE outputs before GEU")
    destination = _root(config) / "geu_evaluation_manifest.json"
    if destination.exists():
        return {"stage": "evaluate_geu", "action": "reused_identical", "result": read_json(destination)}
    hashes = {}
    pending = []
    for input_hash in sorted(manifest["input_artifact_sha256"]):
        for run_index in range(1, GE_RUNS_PER_UNIQUE_INPUT + 1):
            path = _geu_path(config, input_hash, run_index)
            key = f"{input_hash}:run_{run_index}"
            if path.exists():
                existing = read_json(path)
                if existing.get("ge_input_sha256") != input_hash or existing.get("run_index") != run_index:
                    raise ValueError(f"changed GEU cache artifact: {path}")
                hashes[key] = sha256_file(path)
            else:
                pending.append((input_hash, run_index))
    failures = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=GEU_MAX_PARALLEL_OUTPUTS) as executor:
        future_to_key = {
            executor.submit(_evaluate_one_geu, config, input_hash, run_index, contract["contract_sha256"]):
            (input_hash, run_index)
            for input_hash, run_index in pending
        }
        for future in concurrent.futures.as_completed(future_to_key):
            input_hash, run_index = future_to_key[future]
            key = f"{input_hash}:run_{run_index}"
            try:
                payload = future.result()
                _, hashes[key] = _persist_json(_geu_path(config, input_hash, run_index), payload, config)
                print(f"GEU {len(hashes)}/114 {input_hash[:10]} run {run_index}", flush=True)
            except Exception as error:
                failures.append(f"{key}: {error}")
    if failures:
        raise RuntimeError("GEU failures; completed caches preserved: " + " | ".join(failures))
    if len(hashes) != 114:
        raise AssertionError("GEU cache is incomplete")
    payload = {
        "schema_version": "e7_dev20_geu_evaluation_manifest_v1",
        "ge_contract_sha256": contract["contract_sha256"],
        "evaluated_unique_outputs": 114,
        "geu_artifact_sha256": hashes,
        "KPR": None,
        "KPC": None,
        "test50_processed": False,
    }
    action, digest = _persist_json(destination, payload, config)
    return {"stage": "evaluate_geu", "action": action, "output_sha256": digest, "result": payload}


def strict_mean(values: Sequence[Any]) -> float | None:
    """Return a mean only when every repeated value is numeric; never coerce null to zero."""
    if not values or not all(isinstance(value, (int, float)) for value in values):
        return None
    return statistics.fmean(values)


def evaluate_methods(config: E7Config | None = None) -> dict[str, Any]:
    config = config or E7Config()
    _, manifest = _load_generation_prerequisites(config)
    if not (_root(config) / "geu_evaluation_manifest.json").is_file():
        raise FileNotFoundError("finish GEU cache before method aggregation")
    rows = _validate_retrieval_outputs(config)
    consumer_map = {(item["query_id"], item["method"]): item for item in manifest["consumers"]}
    output_rows = []
    for row in rows:
        if row["eligible"] is not True:
            output_rows.append({
                "query_id": row["query_id"], "query": row["query"], "eligible": False,
                "status": "ineligible", "methods": None,
                "selection_reason": row["selection_reason"],
            })
            continue
        methods = {}
        for method in METHOD_IDS:
            consumer = consumer_map[(row["query_id"], method)]
            input_hash = consumer["ge_input_sha256"]
            run_results = []
            for run_index in range(1, GE_RUNS_PER_UNIQUE_INPUT + 1):
                raw_path = _raw_path(config, input_hash, run_index)
                geu_path = _geu_path(config, input_hash, run_index)
                raw = read_json(raw_path)
                geu = read_json(geu_path)
                geo = calculate_target_e2e_geo(
                    raw["raw_output"], consumer["target_source_index_in_top5"], ge_top_k=5
                )
                if consumer["final_target_rank"] > 5 and geo != {"word": 0.0, "pos": 0.0, "wordpos": 0.0}:
                    raise AssertionError("retrieval miss must have zero target E2E GEO")
                run_results.append({
                    "run_index": run_index,
                    "raw_output_artifact_path": str(raw_path),
                    "raw_output_artifact_sha256": sha256_file(raw_path),
                    "raw_output_sha256": raw["raw_output_sha256"],
                    "geu_artifact_path": str(geu_path),
                    "geu_artifact_sha256": sha256_file(geu_path),
                    "target_e2e_geo": geo,
                    "geu": geu["scores"],
                })
            aggregate_geo = {
                key: strict_mean([run["target_e2e_geo"][key] for run in run_results])
                for key in ("word", "pos", "wordpos")
            }
            aggregate_geu = {
                key: strict_mean([run["geu"][key] for run in run_results])
                for key in GEU_SCORE_KEYS
            }
            if aggregate_geu["KPR"] is not None or aggregate_geu["KPC"] is not None:
                raise AssertionError("KPR/KPC null values were coerced")
            methods[method] = {
                "final_target_rank": consumer["final_target_rank"],
                "final_target_score": consumer["final_target_score"],
                "entered_top5": consumer["entered_top5"],
                "top5_document_ids": consumer["top5_document_ids"],
                "target_source_index_in_top5": consumer["target_source_index_in_top5"],
                "ge_input_sha256": input_hash,
                "ge_input_shared_consumer_count": sum(
                    other["ge_input_sha256"] == input_hash for other in manifest["consumers"]
                ),
                "runs": run_results,
                "mean_target_e2e_geo": aggregate_geo,
                "mean_geu": aggregate_geu,
                "ge_executed_despite_target_miss": consumer["final_target_rank"] > 5,
            }
        output_rows.append({
            "query_id": row["query_id"], "query": row["query"], "eligible": True,
            "status": "evaluated", "target_document_id": row["target_document_id"],
            "target_source": row["target_source"], "methods": methods,
        })
    if len(output_rows) != 20 or sum(row["eligible"] for row in output_rows) != 15:
        raise AssertionError("E2E per-query output lost DEV20 records")
    text = "".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in output_rows)
    path = _root(config) / "dev20_e2e_per_query.jsonl"
    action, digest = _persist_text(path, text, config)
    return {"stage": "evaluate_methods", "action": action, "output_sha256": digest, "records": 20}


def _metric_stats(rows: Sequence[Mapping[str, Any]], method: str) -> dict[str, Any]:
    method_rows = [row["methods"][method] for row in rows]
    geo = {
        key: strict_mean([item["mean_target_e2e_geo"][key] for item in method_rows])
        for key in ("word", "pos", "wordpos")
    }
    geu = {
        key: strict_mean([item["mean_geu"][key] for item in method_rows])
        for key in GEU_SCORE_KEYS
    }
    return {
        "n": len(method_rows),
        "hit_at_5_count": sum(item["entered_top5"] for item in method_rows),
        "hit_at_5": statistics.fmean(item["entered_top5"] for item in method_rows),
        "mean_target_e2e_geo": geo,
        "mean_geu": geu,
        "geu_non_null_count": {
            key: sum(isinstance(item["mean_geu"][key], (int, float)) for item in method_rows)
            for key in GEU_SCORE_KEYS
        },
    }


def paired_downstream(rows: Sequence[Mapping[str, Any]], baseline: str, treatment: str) -> dict[str, Any]:
    metric_paths = {
        "geo_word": ("mean_target_e2e_geo", "word"),
        "geo_pos": ("mean_target_e2e_geo", "pos"),
        "geo_wordpos": ("mean_target_e2e_geo", "wordpos"),
        **{f"geu_{key}": ("mean_geu", key) for key in GEU_SCORE_KEYS},
    }
    metrics = {}
    for label, (group, key) in metric_paths.items():
        pairs = [
            (row["methods"][baseline][group][key], row["methods"][treatment][group][key])
            for row in rows
        ]
        valid = [(left, right) for left, right in pairs if isinstance(left, (int, float)) and isinstance(right, (int, float))]
        if not valid:
            metrics[label] = {"n": 0, "mean_treatment_minus_baseline": None, "wins": 0, "ties": 0, "losses": 0}
            continue
        deltas = [right - left for left, right in valid]
        wins = sum(delta > 1e-12 for delta in deltas)
        losses = sum(delta < -1e-12 for delta in deltas)
        metrics[label] = {
            "n": len(valid),
            "mean_treatment_minus_baseline": statistics.fmean(deltas),
            "median_treatment_minus_baseline": statistics.median(deltas),
            "wins": wins,
            "ties": len(valid) - wins - losses,
            "losses": losses,
        }
    return {"baseline": baseline, "treatment": treatment, "metrics": metrics}


def _report(summary: Mapping[str, Any]) -> str:
    labels = {
        "original_e2e": "Original-E2E",
        "autogeo_api_e2e": "AutoGEO_API-E2E",
        "query_aware_autogeo": "Query-Aware-E2E",
        "no_feedback_multi_sample": "No-Feedback-C1-E2E",
        "ra_autogeo_e2e": "RA-AutoGEO-E2E",
    }
    lines = [
        "# E7 DEV20 Downstream E2E Evaluation", "",
        "- DEV20: 20 total, 15 eligible, 5 ineligible retained.",
        "- Unique GE inputs: 57; two frozen generations each (114 raw outputs).",
        "- Identical query + ordered Top-5 texts + GE config share the same cached outputs.",
        "- KPR/KPC: null for every method because all 15 eligible queries lack keypoint files.",
        "- TEST50 processed: no.", "",
        "## Macro means over 15 eligible queries", "",
        "| Method | Hit@5 | GEO word | GEO pos | GEO wordpos | Precision | Recall | Quality mean | KPR | KPC |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    quality_keys = ("Clarity", "Depth", "Balance", "Breadth", "Support", "Insightfulness")
    for method in METHOD_IDS:
        stats = summary["method_statistics"][method]
        geo = stats["mean_target_e2e_geo"]
        geu = stats["mean_geu"]
        quality_mean = strict_mean([geu[key] for key in quality_keys])
        lines.append(
            f"| {labels[method]} | {stats['hit_at_5_count']}/15 | {geo['word']:.6f} | "
            f"{geo['pos']:.6f} | {geo['wordpos']:.6f} | {geu['Precision']:.6f} | "
            f"{geu['Recall']:.6f} | {quality_mean:.6f} | null | null |"
        )
    lines.extend(["", "## Paired downstream comparisons", ""])
    for name, comparison in summary["paired_comparisons"].items():
        metric = comparison["metrics"]["geo_wordpos"]
        lines.append(
            f"- {name}: GEO wordpos treatment-minus-baseline mean = "
            f"{metric['mean_treatment_minus_baseline']:.6f}; wins/ties/losses = "
            f"{metric['wins']}/{metric['ties']}/{metric['losses']}."
        )
    lines.append("")
    return "\n".join(lines)


def summarize(config: E7Config | None = None) -> dict[str, Any]:
    config = config or E7Config()
    per_query_path = _root(config) / "dev20_e2e_per_query.jsonl"
    if not per_query_path.is_file():
        raise FileNotFoundError("aggregate per-query E2E results first")
    rows = [json.loads(line) for line in per_query_path.read_text(encoding="utf-8").splitlines()]
    eligible = [row for row in rows if row["eligible"] is True]
    method_statistics = {method: _metric_stats(eligible, method) for method in METHOD_IDS}
    paired = {
        "A_autogeo_vs_query_aware": paired_downstream(eligible, "autogeo_api_e2e", "query_aware_autogeo"),
        "B_query_aware_vs_ra": paired_downstream(eligible, "query_aware_autogeo", "ra_autogeo_e2e"),
        "C_no_feedback_c1_vs_ra": paired_downstream(eligible, "no_feedback_multi_sample", "ra_autogeo_e2e"),
    }
    summary = {
        "schema_version": "e7_dev20_e2e_summary_v1",
        "dataset_name": "DEV20",
        "total_queries": 20,
        "eligible_queries": 15,
        "ineligible_queries": 5,
        "eligibility_rate": 0.75,
        "ineligible_in_promotion_paired_metrics": False,
        "unique_ge_inputs": 57,
        "runs_per_unique_ge_input": 2,
        "raw_ge_outputs": 114,
        "method_statistics": method_statistics,
        "paired_comparisons": paired,
        "KPR": None,
        "KPC": None,
        "test50_processed": False,
        "rewrite_calls_made": False,
    }
    summary_path = _root(config) / "dev20_e2e_summary.json"
    report_path = _root(config) / "dev20_e2e_report.md"
    summary_action, summary_hash = _persist_json(summary_path, summary, config)
    report_action, report_hash = _persist_text(report_path, _report(summary), config)
    return {
        "stage": "summarize",
        "actions": {"summary": summary_action, "report": report_action},
        "sha256": {"summary": summary_hash, "report": report_hash},
        "result": summary,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--stage", required=True,
        choices=("freeze_contract", "prepare_inputs", "generate_ge", "evaluate_geu", "evaluate_methods", "summarize", "all"),
    )
    args = parser.parse_args()
    runners = {
        "freeze_contract": freeze_contract,
        "prepare_inputs": prepare_inputs,
        "generate_ge": generate_ge,
        "evaluate_geu": evaluate_geu,
        "evaluate_methods": evaluate_methods,
        "summarize": summarize,
    }
    if args.stage == "all":
        order = ("freeze_contract", "prepare_inputs", "generate_ge", "evaluate_geu", "evaluate_methods", "summarize")
        result = {stage: runners[stage]() for stage in order}
    else:
        result = runners[args.stage]()
    print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
