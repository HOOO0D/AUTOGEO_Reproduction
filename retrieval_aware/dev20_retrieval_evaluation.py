"""Frozen DEV20 retrieval-only development evaluation for E7.

No-feedback generation is completed and C1 is frozen before R1 is loaded.
Query 90444 reuses its immutable smoke artifacts instead of being resampled.
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

from autogeo.utils import call_gemini

from .baseline_e2e_smoke import (
    AUTOGEO_REWRITE_MODEL,
    EXPECTED_POOL_V2_EMBEDDINGS_SHA256,
    EXPECTED_POOL_V2_MANIFEST_SHA256,
    EXPECTED_POOL_V2_METADATA_SHA256,
    EXPECTED_RETRIEVER_CONFIG_SHA256,
    EXPECTED_RULE_FILE_SHA256,
    EXPECTED_TARGET_MANIFEST_SHA256,
    _cache_paths,
    _load_retriever,
    _load_rules,
    rewrite_selected_target_with_stock_autogeo,
    sha256_text,
)
from .config import E7Config
from .frozen_local_retriever import FrozenLocalCorpus, load_local_corpus
from .ra_e2e_smoke import (
    REWRITE_TEMPERATURE,
    ROUND_CANDIDATES,
    RankedOption,
    build_ra_prompt,
    select_retrieval_best,
)
from .retrieval_ablation_smoke import (
    EXPECTED_FILTERED_RULES_SHA256,
    NO_FEEDBACK_CANDIDATE_IDS,
    build_query_aware_no_feedback_prompt,
)
from .utils import (
    assert_e7_output_path,
    canonical_json_sha256,
    normalized_text_sha256,
    read_json,
    sha256_file,
    write_e7_json_once,
)

EXPECTED_QUERY_AWARE_IMPLEMENTATION_SHA256 = "e9d5c516f089223a43d4ab5780e2abaf582445fad4526ebcfbdfd52ccfad023b"
EXPECTED_RA_IMPLEMENTATION_SHA256 = "9109d499ee60b7350b0e9d19d1b6240ef8678db3ea52dc707edd18f11ce70e4e"
EXPECTED_AUTOGEO_CORE_SHA256 = "460dd484a88861448fe436b55001a91b31013606abaac4fdbb90ea0b53453805"
EXPECTED_GEMINI_WRAPPER_SHA256 = "7867128d953a15fe9a7a13e814c4ceeaba8bf7d706d4d75995c8e4958b8c6b3c"
SMOKE_QUERY_ID = "90444"
SMOKE_ARTIFACT_SHA256 = {
    "autogeo": "88bb6d3a0dc5c76f06690740a63d9766578c63764be84625086a2f48a989ad6d",
    "query_aware": "8a5424375b3c1ba4954931ddfac15f6f99c310169844a17719f91883482a1dcd",
    "C1": "0be0cb813fc6be8f286a6523b65d4204506142c98a13988e4659f50abc21bd68",
    "C2": "9e9eeb51bde31164476d2407b7d8113dbe0982482e39c08ac0040911b6cad184",
    "C3": "1aa68d57da3fbf935e61c5ed047265f818b24c8cb49478f1b5b34fef091188f8",
    "selection": "e243b32114902f6dbd947c42401b9435d127ead372c0d2ed9e30b18fe247b5be",
    "ra_trajectory": "68dccfeb0313c7a54747f8718e64bf5733eac2a4ca1e773de5040b011dc3fdf4",
}
METHOD_IDS = (
    "original_e2e",
    "autogeo_api_e2e",
    "query_aware_autogeo",
    "no_feedback_multi_sample",
    "ra_autogeo_e2e",
)


@dataclass(frozen=True)
class DevInputs:
    records: tuple[dict[str, Any], ...]
    eligible_records: tuple[dict[str, Any], ...]
    corpus: FrozenLocalCorpus
    target_manifest_path: Path
    corpus_manifest_path: Path
    rules_path: Path
    filtered_rules: tuple[str, ...]


def _root(config: E7Config) -> Path:
    return config.output_paths["evaluation"] / "dev20_retrieval"


def _rewrite_root(config: E7Config) -> Path:
    return config.output_paths["rewrites"] / "dev20_retrieval"


def _baseline_rewrite_path(config: E7Config, method: str, query_id: str) -> Path:
    return _rewrite_root(config) / method / query_id / "rewrite.json"


def _nf_candidate_path(config: E7Config, query_id: str, candidate_id: str) -> Path:
    return _rewrite_root(config) / "no_feedback_multi_sample" / query_id / f"{candidate_id}.json"


def _nf_selection_path(config: E7Config, query_id: str) -> Path:
    return _rewrite_root(config) / "no_feedback_multi_sample" / query_id / "selection.pre_retrieval.json"


def _ra_root(config: E7Config, query_id: str) -> Path:
    return _rewrite_root(config) / "ra_autogeo_e2e" / query_id


def _baseline_result_path(config: E7Config, query_id: str) -> Path:
    return _root(config) / "per_query" / f"{query_id}.baseline.json"


def _ra_result_path(config: E7Config, query_id: str) -> Path:
    return _root(config) / "per_query" / f"{query_id}.ra.json"


def _persist(path: Path, payload: Any, config: E7Config) -> tuple[str, str]:
    if path.exists():
        if read_json(path) != payload:
            raise FileExistsError(f"refusing to replace changed DEV20 artifact: {path}")
        return "reused_identical", sha256_file(path)
    write_e7_json_once(path, payload, config)
    path.chmod(0o444)
    return "created", sha256_file(path)


def _write_text_once(path: Path, text: str, config: E7Config) -> tuple[str, str]:
    destination = assert_e7_output_path(path, config)
    if destination.exists():
        if destination.read_text(encoding="utf-8") != text:
            raise FileExistsError(f"refusing to replace changed DEV20 report: {path}")
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


def _load_dev_inputs(config: E7Config) -> DevInputs:
    config.validate()
    if (config.ge_top_k, config.num_candidates_per_round, config.max_rounds) != (5, 3, 2):
        raise ValueError("frozen Top-5/candidate/round protocol changed")
    source_dir = Path(__file__).resolve().parent
    frozen_sources = {
        source_dir / "retrieval_ablation_smoke.py": EXPECTED_QUERY_AWARE_IMPLEMENTATION_SHA256,
        source_dir / "ra_e2e_smoke.py": EXPECTED_RA_IMPLEMENTATION_SHA256,
        config.project_root / "autogeo" / "rewriters" / "core.py": EXPECTED_AUTOGEO_CORE_SHA256,
        config.project_root / "autogeo" / "utils" / "gemini.py": EXPECTED_GEMINI_WRAPPER_SHA256,
    }
    for path, expected in frozen_sources.items():
        if sha256_file(path) != expected:
            raise ValueError(f"frozen implementation changed: {path}")

    target_path = config.output_paths["targets"] / "dev_targets_r1_pool_v2.frozen.json"
    corpus_path = config.output_paths["corpus"] / "pool_v2" / "dev_corpus_manifest.json"
    if sha256_file(target_path) != EXPECTED_TARGET_MANIFEST_SHA256:
        raise ValueError("frozen DEV target manifest changed")
    if sha256_file(corpus_path) != EXPECTED_POOL_V2_MANIFEST_SHA256:
        raise ValueError("frozen Pool V2 manifest changed")
    records = read_json(target_path).get("records")
    if not isinstance(records, list) or len(records) != 20:
        raise ValueError("frozen development set must contain DEV20")
    if len({str(item.get("query_id")) for item in records}) != 20:
        raise ValueError("DEV20 query identities are not unique")
    eligible = [dict(item) for item in records if item.get("eligible") is True]
    ineligible = [item for item in records if item.get("eligible") is False]
    if (len(eligible), len(ineligible)) != (15, 5):
        raise ValueError("frozen DEV20 eligibility changed")

    corpus = load_local_corpus(corpus_path)
    if len(corpus.documents) != 44_264 or corpus.corpus_split != "dev":
        raise ValueError("frozen DEV Pool V2 identity changed")
    for record in eligible:
        document_id = record.get("target_document_id")
        if not isinstance(document_id, str):
            raise ValueError("eligible record lacks frozen target identity")
        document = corpus.documents[corpus.index_of(document_id)]
        if document.text_hash != record.get("target_text_hash"):
            raise ValueError("frozen target hash differs from Pool V2")
        rank = record.get("initial_r1_rank")
        if not isinstance(rank, int) or not 6 <= rank <= 100:
            raise ValueError("eligible target rank left frozen range 6..100")
    for record in ineligible:
        fields = (record.get("target_document_id"), record.get("initial_r1_rank"), record.get("initial_r1_score"))
        if fields != (None, None, None):
            raise ValueError("ineligible record unexpectedly has a selected target")

    rules_path, rules_sha, filtered_sha, _ = _load_rules(config)
    rules = read_json(rules_path).get("filtered_rules")
    if rules_sha != EXPECTED_RULE_FILE_SHA256 or filtered_sha != EXPECTED_FILTERED_RULES_SHA256:
        raise ValueError("frozen AutoGEO rules changed")
    if not isinstance(rules, list) or canonical_json_sha256(rules) != EXPECTED_FILTERED_RULES_SHA256:
        raise ValueError("filtered_rules payload changed")
    embedding_path, metadata_path = _cache_paths(config)
    if sha256_file(embedding_path) != EXPECTED_POOL_V2_EMBEDDINGS_SHA256:
        raise ValueError("frozen Pool V2 embeddings changed")
    if sha256_file(metadata_path) != EXPECTED_POOL_V2_METADATA_SHA256:
        raise ValueError("frozen Pool V2 embedding metadata changed")
    return DevInputs(
        records=tuple(dict(item) for item in records),
        eligible_records=tuple(eligible),
        corpus=corpus,
        target_manifest_path=target_path,
        corpus_manifest_path=corpus_path,
        rules_path=rules_path,
        filtered_rules=tuple(rules),
    )


def _target_text(inputs: DevInputs, record: Mapping[str, Any]) -> str:
    document = inputs.corpus.documents[inputs.corpus.index_of(record["target_document_id"])]
    if document.text_hash != record["target_text_hash"]:
        raise AssertionError("target identity/hash changed")
    return document.text


def _smoke_path(config: E7Config, name: str) -> Path:
    ablation = config.output_root / "ablations" / "retrieval_side" / SMOKE_QUERY_ID
    paths = {
        "autogeo": config.output_paths["rewrites"] / "autogeo_api_e2e" / "90444.json",
        "query_aware": ablation / "query_aware" / "rewrite.json",
        "C1": ablation / "no_feedback_multi_sample" / "C1.json",
        "C2": ablation / "no_feedback_multi_sample" / "C2.json",
        "C3": ablation / "no_feedback_multi_sample" / "C3.json",
        "selection": ablation / "no_feedback_multi_sample" / "selection.pre_retrieval.json",
        "ra_trajectory": config.output_paths["rewrites"] / "ra_autogeo_e2e" / "90444" / "trajectory.json",
    }
    path = paths[name]
    if sha256_file(path) != SMOKE_ARTIFACT_SHA256[name]:
        raise ValueError(f"frozen 90444 smoke artifact changed: {name}")
    return path


def _rewrite_payload(method_id: str, candidate_id: str, record: Mapping[str, Any], rewritten: str,
                     prompt_hash: str | None, source_path: Path | None) -> dict[str, Any]:
    return {
        "schema_version": "e7_dev20_retrieval_rewrite_v1",
        "method_id": method_id,
        "candidate_id": candidate_id,
        "query_id": record["query_id"],
        "query": record["query"],
        "target_document_id": record["target_document_id"],
        "original_target_hash": record["target_text_hash"],
        "rewritten_text": rewritten,
        "rewrite_hash": normalized_text_sha256(rewritten),
        "rewrite_raw_text_sha256": sha256_text(rewritten),
        "prompt_sha256": prompt_hash,
        "rewrite_model": AUTOGEO_REWRITE_MODEL,
        "generation_config": {"temperature": REWRITE_TEMPERATURE, "top_p": None, "seed": None, "max_tokens": None},
        "rules_file_sha256": EXPECTED_RULE_FILE_SHA256,
        "filtered_rules_sha256": EXPECTED_FILTERED_RULES_SHA256,
        "target_manifest_sha256": EXPECTED_TARGET_MANIFEST_SHA256,
        "corpus_manifest_sha256": EXPECTED_POOL_V2_MANIFEST_SHA256,
        "r1_rank_provided": False,
        "r1_score_provided": False,
        "retrieval_feedback_provided": False,
        "iterative_refinement_used": False,
        "r1_evaluation_performed_before_artifact_freeze": False,
        "source": "frozen_90444_smoke" if source_path else "dev20_fresh_generation",
        "source_artifact_path": str(source_path) if source_path else None,
        "source_artifact_sha256": sha256_file(source_path) if source_path else None,
    }


def _create_rewrite(config: E7Config, inputs: DevInputs, record: Mapping[str, Any], method_id: str,
                    candidate_id: str, prompt: str | None, smoke_name: str | None) -> dict[str, Any]:
    query_id = str(record["query_id"])
    path = (_nf_candidate_path(config, query_id, candidate_id)
            if method_id == "no_feedback_multi_sample"
            else _baseline_rewrite_path(config, method_id, query_id))
    prompt_hash = sha256_text(prompt) if prompt is not None else None
    if path.exists():
        payload = read_json(path)
        required = {
            "query_id": query_id, "target_document_id": record["target_document_id"],
            "original_target_hash": record["target_text_hash"], "method_id": method_id,
            "candidate_id": candidate_id, "prompt_sha256": prompt_hash,
        }
        if any(payload.get(key) != value for key, value in required.items()):
            raise ValueError(f"reused rewrite identity/config changed: {path}")
        return payload

    source_path = None
    if smoke_name:
        source_path = _smoke_path(config, smoke_name)
        source = read_json(source_path)
        rewritten = source.get("rewritten_text")
        if prompt_hash is not None and source.get("prompt_sha256") != prompt_hash:
            raise ValueError("frozen smoke prompt differs from frozen DEV protocol")
    elif method_id == "autogeo_api_e2e":
        rewritten = rewrite_selected_target_with_stock_autogeo(
            _target_text(inputs, record), rules_path=inputs.rules_path
        )
    else:
        if prompt is None:
            raise ValueError("query-aware generation requires its frozen prompt")
        rewritten = call_gemini(prompt, model_name=AUTOGEO_REWRITE_MODEL, temperature=REWRITE_TEMPERATURE)
    if not isinstance(rewritten, str) or not rewritten.strip():
        raise ValueError(f"empty rewrite for {query_id}/{candidate_id}")
    payload = _rewrite_payload(method_id, candidate_id, record, rewritten, prompt_hash, source_path)
    _persist(path, payload, config)
    print(f"generated {query_id} {method_id} {candidate_id}", flush=True)
    return payload


def _freeze_c1(config: E7Config, record: Mapping[str, Any]) -> dict[str, Any]:
    query_id = str(record["query_id"])
    hashes = {cid: sha256_file(_nf_candidate_path(config, query_id, cid)) for cid in NO_FEEDBACK_CANDIDATE_IDS}
    payload = {
        "schema_version": "e7_dev20_no_feedback_selection_pre_retrieval_v1",
        "query_id": query_id,
        "candidate_ids": list(NO_FEEDBACK_CANDIDATE_IDS),
        "selected_candidate_id": "C1",
        "selection_policy": "precommitted_first_sample",
        "candidate_artifact_sha256": hashes,
        "selected_before_r1_evaluation": True,
        "r1_rank_or_score_available_at_selection": False,
        "diagnostic_results_used_for_selection": False,
    }
    _persist(_nf_selection_path(config, query_id), payload, config)
    return payload


def run_generate_baselines(config: E7Config | None = None) -> dict[str, Any]:
    """Generate AutoGEO/query-aware/no-feedback without loading R1."""
    config = config or E7Config()
    inputs = _load_dev_inputs(config)
    destination = _root(config) / "baseline_generation_manifest.json"
    if destination.exists():
        return {"stage": "generate_baselines", "action": "reused_identical", "result": read_json(destination)}
    artifacts = {}
    for record in inputs.eligible_records:
        query_id = str(record["query_id"])
        prompt = build_query_aware_no_feedback_prompt(
            query=record["query"], original_document=_target_text(inputs, record),
            filtered_rules=inputs.filtered_rules,
        )
        smoke = query_id == SMOKE_QUERY_ID
        auto = _create_rewrite(config, inputs, record, "autogeo_api_e2e", "AUTO1", None,
                               "autogeo" if smoke else None)
        qa = _create_rewrite(config, inputs, record, "query_aware_autogeo", "QA1", prompt,
                             "query_aware" if smoke else None)
        candidates = [
            _create_rewrite(config, inputs, record, "no_feedback_multi_sample", cid, prompt,
                            cid if smoke else None)
            for cid in NO_FEEDBACK_CANDIDATE_IDS
        ]
        if smoke:
            _smoke_path(config, "selection")
        selection = _freeze_c1(config, record)
        artifacts[query_id] = {
            "autogeo_sha256": sha256_file(_baseline_rewrite_path(config, "autogeo_api_e2e", query_id)),
            "query_aware_sha256": sha256_file(_baseline_rewrite_path(config, "query_aware_autogeo", query_id)),
            "no_feedback_sha256": selection["candidate_artifact_sha256"],
            "selection_sha256": sha256_file(_nf_selection_path(config, query_id)),
            "shared_query_aware_prompt_sha256": qa["prompt_sha256"],
            "same_prompt_for_qa_and_c1_c2_c3": all(x["prompt_sha256"] == qa["prompt_sha256"] for x in candidates),
            "smoke_artifacts_reused": smoke,
            "autogeo_rewrite_hash": auto["rewrite_hash"],
        }
    payload = {
        "schema_version": "e7_dev20_baseline_generation_manifest_v1",
        "dataset_name": "DEV20", "total_queries": 20, "eligible_queries": 15, "ineligible_queries": 5,
        "artifacts": artifacts,
        "all_c1_selections_frozen_before_any_r1_evaluation": True,
        "r1_loaded_or_evaluated": False, "ge_geo_geu_executed": False, "test50_processed": False,
    }
    action, digest = _persist(destination, payload, config)
    return {"stage": "generate_baselines", "action": action, "output_sha256": digest, "result": payload}


def _original_result(record: Mapping[str, Any], retriever, inputs: DevInputs, config: E7Config) -> dict[str, Any]:
    ranking = retriever.retrieve(record["query"], inputs.corpus, top_k=len(inputs.corpus.documents))
    hit = next(item for item in ranking if item.document_id == record["target_document_id"])
    if hit.rank != record["initial_r1_rank"] or abs(hit.score - record["initial_r1_score"]) > 1e-7:
        raise AssertionError("Original-E2E differs from frozen initial R1 result")
    return {
        "initial_rank": hit.rank, "final_rank": hit.rank,
        "initial_score": hit.score, "final_score": hit.score,
        "rank_gain": 0, "entered_top5": False,
        "top5_document_ids": [x.document_id for x in ranking[:config.ge_top_k]],
        "rewrite_called": False,
    }


def _evaluate_rewrite(record: Mapping[str, Any], artifact: Mapping[str, Any], retriever,
                      inputs: DevInputs, config: E7Config) -> dict[str, Any]:
    if artifact.get("target_document_id") != record["target_document_id"]:
        raise AssertionError("method target identity changed")
    result = retriever.rerank_with_rewritten_target(
        record["query"], inputs.corpus, record["target_document_id"], artifact["rewritten_text"],
        top_k=config.ge_top_k,
    )
    if result.original_target_rank != record["initial_r1_rank"]:
        raise AssertionError("counterfactual rerank changed original target rank")
    if result.reencoded_document_ids != (record["target_document_id"],) or not result.competitor_invariant_verified:
        raise AssertionError("target-only/N-1 competitor invariant failed")
    return {
        "initial_rank": result.original_target_rank, "final_rank": result.new_target_rank,
        "initial_score": result.original_target_score, "final_score": result.new_target_score,
        "rank_gain": result.original_target_rank - result.new_target_rank,
        "entered_top5": result.new_target_rank <= config.ge_top_k,
        "top5_document_ids": [x.document_id for x in result.new_top_k],
        "rewrite_hash": artifact["rewrite_hash"],
        "only_target_embedding_recomputed": True, "competitor_invariant_verified": True,
    }


def run_evaluate_baselines(config: E7Config | None = None) -> dict[str, Any]:
    config = config or E7Config()
    inputs = _load_dev_inputs(config)
    generation_path = _root(config) / "baseline_generation_manifest.json"
    if not generation_path.is_file():
        raise FileNotFoundError("all no-feedback generations/selections must finish before R1 evaluation")
    generation = read_json(generation_path)
    if generation.get("all_c1_selections_frozen_before_any_r1_evaluation") is not True:
        raise ValueError("C1 selections were not globally frozen before R1 evaluation")
    destination = _root(config) / "baseline_evaluation_manifest.json"
    if destination.exists():
        return {"stage": "evaluate_baselines", "action": "reused_identical", "result": read_json(destination)}
    retriever = _load_retriever(type("Inputs", (), {"corpus": inputs.corpus})(), config)
    hashes = {}
    for record in inputs.records:
        query_id = str(record["query_id"])
        output = _baseline_result_path(config, query_id)
        if record["eligible"] is not True:
            payload = {
                "schema_version": "e7_dev20_baseline_retrieval_result_v1",
                "query_id": query_id, "query": record["query"], "eligible": False,
                "status": "ineligible", "methods": None, "reason": record["selection_reason"],
            }
            _persist(output, payload, config)
            hashes[query_id] = sha256_file(output)
            continue
        original = _original_result(record, retriever, inputs, config)
        auto = read_json(_baseline_rewrite_path(config, "autogeo_api_e2e", query_id))
        qa = read_json(_baseline_rewrite_path(config, "query_aware_autogeo", query_id))
        selection_path = _nf_selection_path(config, query_id)
        selection = read_json(selection_path)
        if selection.get("selected_candidate_id") != "C1" or selection.get("selected_before_r1_evaluation") is not True:
            raise AssertionError("formal no-feedback output was not precommitted C1")
        diagnostics = []
        for cid in NO_FEEDBACK_CANDIDATE_IDS:
            diagnostic = _evaluate_rewrite(record, read_json(_nf_candidate_path(config, query_id, cid)),
                                           retriever, inputs, config)
            diagnostic["candidate_id"] = cid
            diagnostics.append(diagnostic)
        selected = next(item for item in diagnostics if item["candidate_id"] == "C1")
        payload = {
            "schema_version": "e7_dev20_baseline_retrieval_result_v1",
            "query_id": query_id, "query": record["query"], "eligible": True,
            "status": "evaluated", "target_document_id": record["target_document_id"],
            "target_text_hash": record["target_text_hash"],
            "methods": {
                "original_e2e": original,
                "autogeo_api_e2e": _evaluate_rewrite(record, auto, retriever, inputs, config),
                "query_aware_autogeo": _evaluate_rewrite(record, qa, retriever, inputs, config),
                "no_feedback_multi_sample": {
                    **selected, "formal_selected_candidate": "C1",
                    "candidate_diagnostics": diagnostics, "diagnostics_used_for_selection": False,
                    "selection_artifact_sha256": sha256_file(selection_path),
                },
            },
            "ge_geo_geu_executed": False,
        }
        _persist(output, payload, config)
        hashes[query_id] = sha256_file(output)
        print(f"evaluated baselines {query_id}", flush=True)
    embedding_path, metadata_path = _cache_paths(config)
    if (sha256_file(embedding_path), sha256_file(metadata_path)) != (
        EXPECTED_POOL_V2_EMBEDDINGS_SHA256, EXPECTED_POOL_V2_METADATA_SHA256
    ):
        raise AssertionError("baseline evaluation modified frozen Pool V2 cache")
    payload = {
        "schema_version": "e7_dev20_baseline_evaluation_manifest_v1",
        "dataset_name": "DEV20", "result_sha256": hashes,
        "eligible_queries": 15, "ineligible_queries": 5,
        "formal_no_feedback_output": "C1", "ge_geo_geu_executed": False, "test50_processed": False,
    }
    action, digest = _persist(destination, payload, config)
    return {"stage": "evaluate_baselines", "action": action, "output_sha256": digest, "result": payload}


def _ra_original(record: Mapping[str, Any], baseline: Mapping[str, Any], inputs: DevInputs) -> RankedOption:
    result = baseline["methods"]["original_e2e"]
    return RankedOption(
        source_id="original", text=_target_text(inputs, record), rewrite_hash=record["target_text_hash"],
        rank=result["final_rank"], score=result["final_score"],
        top5_document_ids=tuple(result["top5_document_ids"]), parent_candidate_id=None,
    )


def _ra_raw(config: E7Config, record: Mapping[str, Any], original: RankedOption, parent: RankedOption,
            round_index: int, candidate_id: str, rules: Sequence[str]) -> dict[str, Any]:
    path = _ra_root(config, str(record["query_id"])) / "raw_candidates" / f"{candidate_id}.json"
    prompt = build_ra_prompt(
        candidate_id=candidate_id, query=record["query"], original_document=original.text,
        filtered_rules=rules, current_best=parent, initial_rank=original.rank, round_index=round_index,
    )
    prompt_hash = sha256_text(prompt)
    if path.exists():
        payload = read_json(path)
        if payload.get("prompt_sha256") != prompt_hash or payload.get("parent_candidate_id") != parent.source_id:
            raise ValueError("reused RA raw candidate has different frozen prompt/parent")
        return payload
    rewritten = call_gemini(prompt, model_name=AUTOGEO_REWRITE_MODEL, temperature=REWRITE_TEMPERATURE)
    if not isinstance(rewritten, str) or not rewritten.strip():
        raise ValueError("RA backend returned an empty candidate")
    payload = {
        "schema_version": "e7_dev20_ra_raw_candidate_v1",
        "query_id": record["query_id"], "target_document_id": record["target_document_id"],
        "original_target_hash": record["target_text_hash"], "round": round_index,
        "candidate_id": candidate_id, "parent_candidate_id": parent.source_id,
        "rewritten_text": rewritten, "rewrite_hash": normalized_text_sha256(rewritten),
        "rewrite_raw_text_sha256": sha256_text(rewritten), "prompt_sha256": prompt_hash,
        "rewrite_model": AUTOGEO_REWRITE_MODEL,
        "generation_config": {"temperature": REWRITE_TEMPERATURE, "top_p": None, "seed": None, "max_tokens": None},
        "filtered_rules_sha256": EXPECTED_FILTERED_RULES_SHA256,
        "retrieval_feedback": {"source": parent.source_id, "rank": parent.rank, "score": parent.score},
        "geo_geu_used": False,
    }
    _persist(path, payload, config)
    print(f"generated RA {record['query_id']} round {round_index} {candidate_id}", flush=True)
    return payload


def _rank_ra(record: Mapping[str, Any], raw: Mapping[str, Any], parent: RankedOption,
             retriever, inputs: DevInputs, config: E7Config) -> RankedOption:
    result = retriever.rerank_with_rewritten_target(
        record["query"], inputs.corpus, record["target_document_id"], raw["rewritten_text"],
        top_k=config.ge_top_k,
    )
    if result.original_target_rank != record["initial_r1_rank"]:
        raise AssertionError("RA changed frozen original rank")
    if result.reencoded_document_ids != (record["target_document_id"],) or not result.competitor_invariant_verified:
        raise AssertionError("RA target-only counterfactual invariant failed")
    return RankedOption(
        source_id=raw["candidate_id"], text=raw["rewritten_text"], rewrite_hash=raw["rewrite_hash"],
        rank=result.new_target_rank, score=result.new_target_score,
        top5_document_ids=tuple(x.document_id for x in result.new_top_k),
        parent_candidate_id=parent.source_id,
    )


def _run_ra_query(config: E7Config, record: Mapping[str, Any], baseline: Mapping[str, Any],
                  retriever, inputs: DevInputs) -> dict[str, Any]:
    query_id = str(record["query_id"])
    output = _ra_result_path(config, query_id)
    if output.exists():
        return read_json(output)
    if query_id == SMOKE_QUERY_ID:
        source_path = _smoke_path(config, "ra_trajectory")
        trajectory = read_json(source_path)
        round2 = trajectory.get("round_2", {})
        payload = {
            "schema_version": "e7_dev20_ra_retrieval_result_v1",
            "query_id": query_id, "query": record["query"], "eligible": True,
            "target_document_id": record["target_document_id"],
            "initial_rank": trajectory["initial_rank"], "final_rank": trajectory["final_rank"],
            "initial_score": trajectory["initial_score"], "final_score": trajectory["final_score"],
            "rank_gain": trajectory["rank_gain"], "entered_top5": trajectory["entered_top5"],
            "rounds_used": 1 if round2.get("executed") is False else 2,
            "candidate_count": len(trajectory.get("candidates", [])),
            "early_stop": round2.get("executed") is False,
            "selected_source": trajectory["final_selected_source"], "stop_reason": trajectory["stop_reason"],
            "smoke_artifact_reused": True, "source_artifact_path": str(source_path),
            "source_artifact_sha256": sha256_file(source_path),
            "current_best_preserved": True, "geo_geu_used_for_selection": False,
        }
        if payload["initial_rank"] != record["initial_r1_rank"]:
            raise AssertionError("90444 reused RA trajectory has wrong original rank")
        _persist(output, payload, config)
        return payload

    original = _ra_original(record, baseline, inputs)
    parent = original
    rounds = []
    all_candidates = []
    early_stop = False
    for round_index in (1, 2):
        ranked = []
        for cid in ROUND_CANDIDATES[round_index]:
            raw = _ra_raw(config, record, original, parent, round_index, cid, inputs.filtered_rules)
            ranked.append(_rank_ra(record, raw, parent, retriever, inputs, config))
        selected = select_retrieval_best((parent, *ranked))
        candidates = []
        for option in ranked:
            raw_path = _ra_root(config, query_id) / "raw_candidates" / f"{option.source_id}.json"
            item = {
                "round": round_index, "candidate_id": option.source_id,
                "parent_candidate_id": parent.source_id, "raw_artifact_sha256": sha256_file(raw_path),
                "rewrite_hash": option.rewrite_hash, "rank": option.rank, "score": option.score,
                "rank_gain_vs_original": original.rank - option.rank,
                "selected": option.source_id == selected.source_id,
                "entered_top5": option.rank <= config.ge_top_k,
            }
            candidates.append(item)
            all_candidates.append(item)
        round_payload = {
            "round": round_index, "parent_source": parent.source_id,
            "selection_pool": [parent.source_id, *ROUND_CANDIDATES[round_index]],
            "selection_policy": ["minimum_rank", "maximum_score", "candidate_id_ascending"],
            "selected_source": selected.source_id, "selected_rank": selected.rank,
            "selected_score": selected.score, "entered_top5": selected.rank <= config.ge_top_k,
            "current_best_preserved_in_selection_set": True, "candidates": candidates,
        }
        _persist(_ra_root(config, query_id) / f"round_{round_index}.json", round_payload, config)
        rounds.append(round_payload)
        parent = selected
        if selected.rank <= config.ge_top_k:
            early_stop = round_index < config.max_rounds
            break
    payload = {
        "schema_version": "e7_dev20_ra_retrieval_result_v1",
        "query_id": query_id, "query": record["query"], "eligible": True,
        "target_document_id": record["target_document_id"],
        "initial_rank": original.rank, "final_rank": parent.rank,
        "initial_score": original.score, "final_score": parent.score,
        "rank_gain": original.rank - parent.rank, "entered_top5": parent.rank <= config.ge_top_k,
        "rounds_used": len(rounds), "candidate_count": len(all_candidates), "early_stop": early_stop,
        "selected_source": parent.source_id,
        "stop_reason": "entered_top5" if parent.rank <= config.ge_top_k else "max_rounds_reached",
        "smoke_artifact_reused": False, "current_best_preserved": True,
        "selection_policy": ["minimum_rank", "maximum_score", "candidate_id_ascending"],
        "rounds": rounds, "candidates": all_candidates,
        "monotonic_non_worsening_verified": (
            parent.rank < original.rank or (parent.rank == original.rank and parent.score >= original.score)
        ),
        "geo_geu_used_for_selection": False,
    }
    _persist(output, payload, config)
    print(f"evaluated RA {query_id}: {original.rank}->{parent.rank}", flush=True)
    return payload


def run_ra(config: E7Config | None = None) -> dict[str, Any]:
    config = config or E7Config()
    inputs = _load_dev_inputs(config)
    if not (_root(config) / "baseline_evaluation_manifest.json").is_file():
        raise FileNotFoundError("baseline retrieval evaluation must finish before RA")
    destination = _root(config) / "ra_evaluation_manifest.json"
    if destination.exists():
        return {"stage": "run_ra", "action": "reused_identical", "result": read_json(destination)}
    retriever = _load_retriever(type("Inputs", (), {"corpus": inputs.corpus})(), config)
    hashes = {}
    for record in inputs.records:
        query_id = str(record["query_id"])
        if record["eligible"] is not True:
            payload = {
                "schema_version": "e7_dev20_ra_retrieval_result_v1",
                "query_id": query_id, "query": record["query"], "eligible": False,
                "status": "ineligible", "reason": record["selection_reason"],
            }
            _persist(_ra_result_path(config, query_id), payload, config)
        else:
            baseline = read_json(_baseline_result_path(config, query_id))
            _run_ra_query(config, record, baseline, retriever, inputs)
        hashes[query_id] = sha256_file(_ra_result_path(config, query_id))
    embedding_path, metadata_path = _cache_paths(config)
    if (sha256_file(embedding_path), sha256_file(metadata_path)) != (
        EXPECTED_POOL_V2_EMBEDDINGS_SHA256, EXPECTED_POOL_V2_METADATA_SHA256
    ):
        raise AssertionError("RA modified frozen Pool V2 cache")
    payload = {
        "schema_version": "e7_dev20_ra_evaluation_manifest_v1", "dataset_name": "DEV20",
        "result_sha256": hashes, "eligible_queries": 15, "ineligible_queries": 5,
        "num_candidates_per_round": 3, "max_rounds": 2, "strict_top5_early_stop": True,
        "ge_geo_geu_used_for_selection": False, "test50_processed": False,
    }
    action, digest = _persist(destination, payload, config)
    return {"stage": "run_ra", "action": action, "output_sha256": digest, "result": payload}


def _method_metrics(records: Sequence[Mapping[str, Any]], method: str) -> dict[str, Any]:
    results = [item["methods"][method] for item in records]
    ranks = [int(item["final_rank"]) for item in results]
    gains = [int(item["rank_gain"]) for item in results]
    hits = sum(bool(item["entered_top5"]) for item in results)
    return {
        "n": len(results), "hit_at_5_count": hits, "hit_at_5": hits / len(results),
        "mean_final_rank": statistics.fmean(ranks), "median_final_rank": statistics.median(ranks),
        "mean_rank_gain": statistics.fmean(gains), "median_rank_gain": statistics.median(gains),
        "target_mrr": statistics.fmean(1.0 / rank for rank in ranks),
    }


def paired_comparison(records: Sequence[Mapping[str, Any]], baseline_method: str,
                      treatment_method: str) -> dict[str, Any]:
    pairs = [(item["methods"][baseline_method], item["methods"][treatment_method]) for item in records]
    deltas = [int(left["final_rank"]) - int(right["final_rank"]) for left, right in pairs]
    wins = sum(delta > 0 for delta in deltas)
    losses = sum(delta < 0 for delta in deltas)
    return {
        "baseline": baseline_method, "treatment": treatment_method,
        "positive_rank_delta_favors_treatment": True,
        "treatment_rank_wins": wins, "rank_ties": len(deltas) - wins - losses,
        "treatment_rank_losses": losses,
        "mean_paired_final_rank_improvement": statistics.fmean(deltas),
        "median_paired_final_rank_improvement": statistics.median(deltas),
        "treatment_hit5_gains": sum(not left["entered_top5"] and right["entered_top5"] for left, right in pairs),
        "treatment_hit5_losses": sum(left["entered_top5"] and not right["entered_top5"] for left, right in pairs),
        "per_query_rank_delta": {record["query_id"]: delta for record, delta in zip(records, deltas)},
    }


def _report(summary: Mapping[str, Any]) -> str:
    lines = [
        "# E7 DEV20 Retrieval-Side Development Evaluation", "",
        "- DEV queries: 20", "- Eligible frozen targets: 15", "- Ineligible queries retained: 5",
        "- Eligibility rate: 15/20 (0.75)", "- GE/GEO/GEU executed: no", "- TEST50 processed: no", "",
        "## Eligible-target retrieval metrics", "",
        "| Method | Hit@5 | Mean rank | Median rank | Mean gain | Median gain | Target MRR |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    labels = {
        "original_e2e": "Original-E2E", "autogeo_api_e2e": "AutoGEO_API-E2E",
        "query_aware_autogeo": "Query-Aware AutoGEO",
        "no_feedback_multi_sample": "No-Feedback C1", "ra_autogeo_e2e": "RA-AutoGEO",
    }
    for method in METHOD_IDS:
        item = summary["method_statistics"][method]
        lines.append(
            f"| {labels[method]} | {item['hit_at_5_count']}/15 ({item['hit_at_5']:.3f}) | "
            f"{item['mean_final_rank']:.3f} | {item['median_final_rank']:.3f} | "
            f"{item['mean_rank_gain']:.3f} | {item['median_rank_gain']:.3f} | {item['target_mrr']:.6f} |"
        )
    lines.extend(["", "## Frozen paired comparisons", ""])
    for key, item in summary["paired_comparisons"].items():
        lines.append(
            f"- {key}: treatment rank wins/ties/losses = "
            f"{item['treatment_rank_wins']}/{item['rank_ties']}/{item['treatment_rank_losses']}; "
            f"mean paired rank improvement = {item['mean_paired_final_rank_improvement']:.3f}; "
            f"Hit@5 gains/losses = {item['treatment_hit5_gains']}/{item['treatment_hit5_losses']}."
        )
    lines.extend([
        "", "C2/C3 are post-hoc diagnostics only; formal No-Feedback is precommitted C1.",
        "The five ineligible queries are reported as ineligible and are not retrieval failures.", "",
    ])
    return "\n".join(lines)


def run_summary(config: E7Config | None = None) -> dict[str, Any]:
    config = config or E7Config()
    inputs = _load_dev_inputs(config)
    if not (_root(config) / "ra_evaluation_manifest.json").is_file():
        raise FileNotFoundError("RA evaluation must finish before summary")
    combined = []
    eligible = []
    for record in inputs.records:
        query_id = str(record["query_id"])
        baseline = read_json(_baseline_result_path(config, query_id))
        ra = read_json(_ra_result_path(config, query_id))
        if record["eligible"] is not True:
            item = {
                "query_id": query_id, "query": record["query"], "eligible": False,
                "status": "ineligible", "target_document_id": None, "methods": None,
                "selection_reason": record["selection_reason"],
            }
        else:
            methods = dict(baseline["methods"])
            methods["ra_autogeo_e2e"] = {
                key: ra[key] for key in (
                    "initial_rank", "final_rank", "initial_score", "final_score", "rank_gain",
                    "entered_top5", "rounds_used", "candidate_count", "early_stop", "selected_source",
                )
            }
            item = {
                "query_id": query_id, "query": record["query"], "eligible": True,
                "status": "evaluated", "target_document_id": record["target_document_id"],
                "target_source": record["target_source"], "initial_rank": record["initial_r1_rank"],
                "initial_score": record["initial_r1_score"], "methods": methods,
            }
            eligible.append(item)
        combined.append(item)
    statistics_by_method = {method: _method_metrics(eligible, method) for method in METHOD_IDS}
    comparisons = {
        "A_autogeo_vs_query_aware": paired_comparison(eligible, "autogeo_api_e2e", "query_aware_autogeo"),
        "B_query_aware_vs_ra": paired_comparison(eligible, "query_aware_autogeo", "ra_autogeo_e2e"),
        "C_no_feedback_c1_vs_ra": paired_comparison(eligible, "no_feedback_multi_sample", "ra_autogeo_e2e"),
    }
    summary = {
        "schema_version": "e7_dev20_retrieval_summary_v1", "dataset_name": "DEV20",
        "total_queries": 20, "eligible_queries": 15, "ineligible_queries": 5,
        "eligibility_rate": 0.75, "ineligible_counted_as_retrieval_failure": False,
        "method_statistics": statistics_by_method, "paired_comparisons": comparisons,
        "formal_no_feedback_output": "C1", "no_feedback_c2_c3_role": "posthoc_diagnostic_only",
        "retriever_id": "r1_independent_dense", "retriever_config_sha256": EXPECTED_RETRIEVER_CONFIG_SHA256,
        "corpus_manifest_sha256": EXPECTED_POOL_V2_MANIFEST_SHA256,
        "target_manifest_sha256": EXPECTED_TARGET_MANIFEST_SHA256,
        "filtered_rules_sha256": EXPECTED_FILTERED_RULES_SHA256,
        "ge_geo_geu_executed": False, "test50_processed": False,
    }
    jsonl = "".join(json.dumps(item, ensure_ascii=False, sort_keys=True) + "\n" for item in combined)
    jsonl_action, jsonl_sha = _write_text_once(_root(config) / "dev20_retrieval_per_query.jsonl", jsonl, config)
    summary_action, summary_sha = _persist(_root(config) / "dev20_retrieval_summary.json", summary, config)
    report_action, report_sha = _write_text_once(_root(config) / "dev20_retrieval_report.md", _report(summary), config)
    return {
        "stage": "summary",
        "actions": {"jsonl": jsonl_action, "summary": summary_action, "report": report_action},
        "sha256": {"jsonl": jsonl_sha, "summary": summary_sha, "report": report_sha},
        "result": summary,
    }


def run_validate(config: E7Config | None = None) -> dict[str, Any]:
    inputs = _load_dev_inputs(config or E7Config())
    return {
        "stage": "validate", "dataset_name": "DEV20", "total": len(inputs.records),
        "eligible": len(inputs.eligible_records), "ineligible": len(inputs.records) - len(inputs.eligible_records),
        "candidate_count": 3, "max_rounds": 2, "smoke_90444_will_be_reused": True,
        "external_calls_made": False,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--stage", required=True,
        choices=("validate", "generate_baselines", "evaluate_baselines", "run_ra", "summary", "all"),
    )
    args = parser.parse_args()
    runners = {
        "validate": run_validate, "generate_baselines": run_generate_baselines,
        "evaluate_baselines": run_evaluate_baselines, "run_ra": run_ra, "summary": run_summary,
    }
    if args.stage == "all":
        result = {name: runners[name]() for name in ("generate_baselines", "evaluate_baselines", "run_ra", "summary")}
    else:
        result = runners[args.stage]()
    print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
