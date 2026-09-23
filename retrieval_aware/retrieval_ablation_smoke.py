"""Query-awareness and no-feedback sampling ablations for frozen query 90444."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Mapping, Sequence

from autogeo.utils import call_gemini

from .baseline_e2e_smoke import (
    AUTOGEO_REWRITE_MODEL,
    EXPECTED_POOL_V2_EMBEDDINGS_SHA256,
    EXPECTED_POOL_V2_METADATA_SHA256,
    EXPECTED_RETRIEVER_CONFIG_SHA256,
    EXPECTED_RULE_FILE_SHA256,
    EXPECTED_SMOKE_QUERY_ID,
    EXPECTED_TARGET_MANIFEST_SHA256,
    _cache_paths,
    _load_retriever,
    load_smoke_inputs,
    sha256_text,
)
from .config import E7Config
from .utils import (
    canonical_json_sha256,
    normalized_text_sha256,
    read_json,
    sha256_file,
    write_e7_json_once,
)


EXPECTED_AUTOGEO_CORE_SHA256 = (
    "460dd484a88861448fe436b55001a91b31013606abaac4fdbb90ea0b53453805"
)
EXPECTED_FILTERED_RULES_SHA256 = (
    "1b1d918ce956c07f3eb142b9465e5d5cf59786135a7a1c287aae3e58a091aff0"
)
REWRITE_TEMPERATURE = 0.7
QUERY_AWARE_CANDIDATE_ID = "QA1"
NO_FEEDBACK_CANDIDATE_IDS = ("C1", "C2", "C3")
FORBIDDEN_FEEDBACK_TERMS = (
    "r1",
    "retrieval rank",
    "retrieval score",
    "retrieval feedback",
    "previous retrieval",
    "previous best",
    "iterative refinement",
)


def build_query_aware_no_feedback_prompt(
    *,
    query: str,
    original_document: str,
    filtered_rules: Sequence[str],
) -> str:
    """One shared prompt for both ablations, with query but no retriever signal."""
    if not filtered_rules or not all(isinstance(rule, str) for rule in filtered_rules):
        raise ValueError("ablation requires unchanged AutoGEO filtered rules")
    rules = "- " + "\n- ".join(filtered_rules)
    prompt = f"""Here is the original factual source:

{original_document}

## User Query

{query}

Rewrite the source so that it directly addresses and aligns with the user's
information need while preserving the original facts and natural readability.

## Quality Guidelines to Follow

{rules}

## Strict Factual and Writing Constraints

- The Original Source Document is the sole factual source of truth.
- Do not add facts unsupported by the Original Source Document.
- Do not change the original factual claims.
- Do not fabricate facts, statistics, citations, authorities, or entities.
- Do not use keyword stuffing or mechanical repetition of the query.
- Return only the rewritten document, with no preamble or commentary.
""".strip()
    lowered = prompt.casefold()
    for forbidden in FORBIDDEN_FEEDBACK_TERMS:
        if forbidden in lowered:
            raise AssertionError(f"no-feedback prompt leaked forbidden term: {forbidden}")
    return prompt


def freeze_no_feedback_selection(
    candidate_artifact_sha256: Mapping[str, str],
) -> dict[str, object]:
    """Freeze C1 before any rank/score fields exist or R1 is evaluated."""
    if tuple(candidate_artifact_sha256) != NO_FEEDBACK_CANDIDATE_IDS:
        raise ValueError("no-feedback generation must contain exactly C1/C2/C3")
    if not all(
        isinstance(value, str) and len(value) == 64
        for value in candidate_artifact_sha256.values()
    ):
        raise ValueError("candidate artifact hashes must be SHA256 strings")
    return {
        "schema_version": "e7_no_feedback_selection_pre_retrieval_v1",
        "query_id": EXPECTED_SMOKE_QUERY_ID,
        "candidate_ids": list(NO_FEEDBACK_CANDIDATE_IDS),
        "selected_candidate_id": "C1",
        "selection_policy": "precommitted_first_sample",
        "candidate_artifact_sha256": dict(candidate_artifact_sha256),
        "selected_before_r1_evaluation": True,
        "r1_rank_or_score_available_at_selection": False,
        "diagnostic_results_used_for_selection": False,
    }


def selected_no_feedback_result(
    selection: Mapping[str, object],
    diagnostics: Sequence[Mapping[str, object]],
) -> Mapping[str, object]:
    """Resolve only the precommitted C1, regardless of diagnostic outcomes."""
    selected_id = selection.get("selected_candidate_id")
    if selected_id != "C1":
        raise ValueError("formal no-feedback output must remain C1")
    by_id = {item.get("candidate_id"): item for item in diagnostics}
    if set(by_id) != set(NO_FEEDBACK_CANDIDATE_IDS):
        raise ValueError("diagnostics must contain C1/C2/C3 exactly once")
    return by_id[selected_id]


def _root(config: E7Config) -> Path:
    return (
        config.output_root
        / "ablations"
        / "retrieval_side"
        / EXPECTED_SMOKE_QUERY_ID
    )


def _query_aware_path(config: E7Config) -> Path:
    return _root(config) / "query_aware" / "rewrite.json"


def _no_feedback_path(config: E7Config, candidate_id: str) -> Path:
    return _root(config) / "no_feedback_multi_sample" / f"{candidate_id}.json"


def _selection_path(config: E7Config) -> Path:
    return _root(config) / "no_feedback_multi_sample" / "selection.pre_retrieval.json"


def _generation_manifest_path(config: E7Config) -> Path:
    return _root(config) / "generation_manifest.json"


def _evaluation_path(config: E7Config) -> Path:
    return _root(config) / "retrieval_evaluation.json"


def _summary_path(config: E7Config) -> Path:
    return _root(config) / "summary.json"


def _persist(
    path: Path,
    payload: dict[str, object],
    config: E7Config,
) -> tuple[str, str]:
    if path.exists():
        if read_json(path) != payload:
            raise FileExistsError(f"refusing to replace changed ablation artifact: {path}")
        return "reused_identical", sha256_file(path)
    write_e7_json_once(path, payload, config)
    path.chmod(0o444)
    return "created", sha256_file(path)


def _load_inputs(config: E7Config):
    inputs = load_smoke_inputs(config)
    if inputs.target_record["query_id"] != EXPECTED_SMOKE_QUERY_ID:
        raise ValueError("retrieval ablation smoke query changed")
    if sha256_file(config.project_root / "autogeo" / "rewriters" / "core.py") != (
        EXPECTED_AUTOGEO_CORE_SHA256
    ):
        raise ValueError("original AutoGEO core changed")
    rules_payload = read_json(inputs.rules_path)
    filtered_rules = rules_payload.get("filtered_rules")
    if canonical_json_sha256(filtered_rules) != EXPECTED_FILTERED_RULES_SHA256:
        raise ValueError("filtered_rules changed")
    return inputs, tuple(filtered_rules)


def _rewrite_payload(
    *,
    method_id: str,
    candidate_id: str,
    rewritten_text: str,
    prompt_sha256: str,
    inputs,
    generation_sequence_index: int,
) -> dict[str, object]:
    return {
        "schema_version": "e7_retrieval_ablation_rewrite_v1",
        "method_id": method_id,
        "candidate_id": candidate_id,
        "query_id": EXPECTED_SMOKE_QUERY_ID,
        "query": inputs.target_record["query"],
        "target_document_id": inputs.target_record["target_document_id"],
        "original_target_hash": inputs.target_record["target_text_hash"],
        "rewritten_text": rewritten_text,
        "rewrite_hash": normalized_text_sha256(rewritten_text),
        "rewrite_raw_text_sha256": sha256_text(rewritten_text),
        "prompt_sha256": prompt_sha256,
        "generation_sequence_index": generation_sequence_index,
        "rewrite_model": AUTOGEO_REWRITE_MODEL,
        "generation_config": {
            "temperature": REWRITE_TEMPERATURE,
            "top_p": None,
            "seed": None,
            "max_tokens": None,
        },
        "rules_file_sha256": EXPECTED_RULE_FILE_SHA256,
        "filtered_rules_sha256": EXPECTED_FILTERED_RULES_SHA256,
        "r1_rank_provided": False,
        "r1_score_provided": False,
        "retrieval_feedback_provided": False,
        "previous_retrieval_result_provided": False,
        "iterative_refinement_used": False,
        "ra_candidate_or_trajectory_read": False,
        "r1_evaluation_performed_before_artifact_freeze": False,
        "target_manifest_sha256": EXPECTED_TARGET_MANIFEST_SHA256,
        "corpus_manifest_sha256": inputs.corpus.manifest_sha256,
    }


def run_generation(config: E7Config | None = None) -> dict[str, object]:
    """Generate all samples and freeze C1 selection without loading/evaluating R1."""
    config = config or E7Config()
    inputs, filtered_rules = _load_inputs(config)
    manifest_path = _generation_manifest_path(config)
    if manifest_path.exists():
        payload = read_json(manifest_path)
        return {
            "stage": "generate",
            "action": "reused_identical",
            "output_path": str(manifest_path),
            "output_sha256": sha256_file(manifest_path),
            "result": payload,
        }
    expected_paths = [
        _query_aware_path(config),
        *(_no_feedback_path(config, item) for item in NO_FEEDBACK_CANDIDATE_IDS),
        _selection_path(config),
    ]
    if any(path.exists() for path in expected_paths):
        raise FileExistsError("partial ablation generation artifacts require manual audit")

    original_document = inputs.corpus.documents[
        inputs.corpus.index_of(inputs.target_record["target_document_id"])
    ].text
    prompt = build_query_aware_no_feedback_prompt(
        query=inputs.target_record["query"],
        original_document=original_document,
        filtered_rules=filtered_rules,
    )
    prompt_hash = sha256_text(prompt)
    requests = (
        ("query_aware_autogeo", QUERY_AWARE_CANDIDATE_ID),
        *(("no_feedback_multi_sample", item) for item in NO_FEEDBACK_CANDIDATE_IDS),
    )
    generated: list[tuple[str, str, str]] = []
    for method_id, candidate_id in requests:
        rewritten = call_gemini(
            prompt,
            model_name=AUTOGEO_REWRITE_MODEL,
            temperature=REWRITE_TEMPERATURE,
        )
        if not isinstance(rewritten, str) or not rewritten.strip():
            raise ValueError(f"rewrite backend returned empty {candidate_id}")
        generated.append((method_id, candidate_id, rewritten))

    artifact_hashes: dict[str, str] = {}
    for index, (method_id, candidate_id, rewritten) in enumerate(generated, start=1):
        path = (
            _query_aware_path(config)
            if method_id == "query_aware_autogeo"
            else _no_feedback_path(config, candidate_id)
        )
        payload = _rewrite_payload(
            method_id=method_id,
            candidate_id=candidate_id,
            rewritten_text=rewritten,
            prompt_sha256=prompt_hash,
            inputs=inputs,
            generation_sequence_index=index,
        )
        _, artifact_hashes[candidate_id] = _persist(path, payload, config)

    no_feedback_hashes = {
        candidate_id: artifact_hashes[candidate_id]
        for candidate_id in NO_FEEDBACK_CANDIDATE_IDS
    }
    selection = freeze_no_feedback_selection(no_feedback_hashes)
    selection_action, selection_sha = _persist(
        _selection_path(config), selection, config
    )
    manifest: dict[str, object] = {
        "schema_version": "e7_retrieval_ablation_generation_manifest_v1",
        "query_id": EXPECTED_SMOKE_QUERY_ID,
        "shared_prompt_sha256": prompt_hash,
        "same_prompt_and_input_for_all_four_generations": True,
        "rewrite_model": AUTOGEO_REWRITE_MODEL,
        "generation_config": {
            "temperature": REWRITE_TEMPERATURE,
            "top_p": None,
            "seed": None,
            "max_tokens": None,
        },
        "query_aware_artifact_sha256": artifact_hashes[QUERY_AWARE_CANDIDATE_ID],
        "no_feedback_artifact_sha256": no_feedback_hashes,
        "selection_artifact_sha256": selection_sha,
        "selection_action": selection_action,
        "selected_no_feedback_output": "C1",
        "selection_frozen_before_any_r1_evaluation": True,
        "r1_loaded_or_evaluated_during_generation": False,
        "ra_candidate_or_trajectory_read": False,
        "ge_geo_geu_executed": False,
    }
    action, output_sha = _persist(manifest_path, manifest, config)
    return {
        "stage": "generate",
        "action": action,
        "output_path": str(manifest_path),
        "output_sha256": output_sha,
        "result": manifest,
    }


def _evaluate_rewrite(payload, retriever, inputs, config: E7Config):
    rewritten = payload.get("rewritten_text")
    if not isinstance(rewritten, str) or not rewritten.strip():
        raise ValueError("ablation rewrite artifact lacks text")
    if normalized_text_sha256(rewritten) != payload.get("rewrite_hash"):
        raise ValueError("ablation rewrite hash mismatch")
    result = retriever.rerank_with_rewritten_target(
        inputs.target_record["query"],
        inputs.corpus,
        inputs.target_record["target_document_id"],
        rewritten,
        top_k=config.ge_top_k,
    )
    if result.original_target_rank != inputs.target_record["initial_r1_rank"]:
        raise AssertionError("ablation original rank differs from frozen target")
    if result.reencoded_document_ids != (inputs.target_record["target_document_id"],):
        raise AssertionError("ablation re-encoded more than target D*")
    if not result.competitor_invariant_verified:
        raise AssertionError("ablation competitor invariant failed")
    return {
        "candidate_id": payload["candidate_id"],
        "rewrite_hash": payload["rewrite_hash"],
        "rank": result.new_target_rank,
        "score": result.new_target_score,
        "rank_gain": inputs.target_record["initial_r1_rank"] - result.new_target_rank,
        "entered_top5": result.new_target_rank <= config.ge_top_k,
        "top5_document_ids": [item.document_id for item in result.new_top_k],
        "only_target_embedding_recomputed": True,
        "competitor_invariant_verified": True,
    }


def run_retrieval_evaluation(config: E7Config | None = None) -> dict[str, object]:
    config = config or E7Config()
    inputs, _ = _load_inputs(config)
    generation_manifest_path = _generation_manifest_path(config)
    selection_path = _selection_path(config)
    if not generation_manifest_path.is_file() or not selection_path.is_file():
        raise FileNotFoundError("generation and pre-retrieval selection must finish first")
    generation = read_json(generation_manifest_path)
    selection = read_json(selection_path)
    if generation.get("selection_frozen_before_any_r1_evaluation") is not True:
        raise ValueError("no-feedback C1 selection was not precommitted")
    if selection.get("selected_candidate_id") != "C1":
        raise ValueError("formal no-feedback output changed from C1")
    destination = _evaluation_path(config)
    if destination.exists():
        payload = read_json(destination)
        return {
            "stage": "evaluate",
            "action": "reused_identical",
            "output_path": str(destination),
            "output_sha256": sha256_file(destination),
            "result": payload,
        }

    rewrites = [read_json(_query_aware_path(config))]
    rewrites.extend(
        read_json(_no_feedback_path(config, candidate_id))
        for candidate_id in NO_FEEDBACK_CANDIDATE_IDS
    )
    prompt_hashes = {payload.get("prompt_sha256") for payload in rewrites}
    if prompt_hashes != {generation["shared_prompt_sha256"]}:
        raise AssertionError("ablation generations did not use the same prompt")
    if any(
        payload.get(key) is not False
        for payload in rewrites
        for key in (
            "r1_rank_provided",
            "r1_score_provided",
            "retrieval_feedback_provided",
            "previous_retrieval_result_provided",
            "iterative_refinement_used",
            "ra_candidate_or_trajectory_read",
        )
    ):
        raise AssertionError("no-feedback generation leaked retriever/RA information")

    retriever = _load_retriever(inputs, config)
    query_aware = _evaluate_rewrite(rewrites[0], retriever, inputs, config)
    diagnostics = [
        _evaluate_rewrite(payload, retriever, inputs, config)
        for payload in rewrites[1:]
    ]
    selected = selected_no_feedback_result(selection, diagnostics)
    embedding_path, metadata_path = _cache_paths(config)
    if sha256_file(embedding_path) != EXPECTED_POOL_V2_EMBEDDINGS_SHA256:
        raise AssertionError("ablation modified frozen Pool V2 embeddings")
    if sha256_file(metadata_path) != EXPECTED_POOL_V2_METADATA_SHA256:
        raise AssertionError("ablation modified frozen cache metadata")
    payload: dict[str, object] = {
        "schema_version": "e7_retrieval_side_ablation_evaluation_v1",
        "query_id": EXPECTED_SMOKE_QUERY_ID,
        "query": inputs.target_record["query"],
        "target_document_id": inputs.target_record["target_document_id"],
        "original_target_hash": inputs.target_record["target_text_hash"],
        "initial_rank": inputs.target_record["initial_r1_rank"],
        "initial_score": inputs.target_record["initial_r1_score"],
        "query_aware_autogeo": query_aware,
        "no_feedback_multi_sample": {
            "selected_candidate_id": "C1",
            "selected_result": dict(selected),
            "candidate_diagnostics": diagnostics,
            "selection_artifact_sha256": sha256_file(selection_path),
            "selected_before_r1_evaluation": True,
            "diagnostic_results_used_for_selection": False,
        },
        "shared_controls": {
            "prompt_sha256": generation["shared_prompt_sha256"],
            "query": inputs.target_record["query"],
            "target_document_id": inputs.target_record["target_document_id"],
            "original_target_hash": inputs.target_record["target_text_hash"],
            "filtered_rules_sha256": EXPECTED_FILTERED_RULES_SHA256,
            "rewrite_model": AUTOGEO_REWRITE_MODEL,
            "generation_config": generation["generation_config"],
            "retriever_id": "r1_independent_dense",
            "retriever_config_sha256": EXPECTED_RETRIEVER_CONFIG_SHA256,
            "corpus_manifest_sha256": inputs.corpus.manifest_sha256,
        },
        "ge_geo_geu_executed": False,
        "ra_candidate_or_trajectory_read": False,
        "test50_processed": False,
        "dev15_batch_run": False,
    }
    action, output_sha = _persist(destination, payload, config)
    return {
        "stage": "evaluate",
        "action": action,
        "output_path": str(destination),
        "output_sha256": output_sha,
        "result": payload,
    }


def run_summary(config: E7Config | None = None) -> dict[str, object]:
    config = config or E7Config()
    inputs, _ = _load_inputs(config)
    evaluation_path = _evaluation_path(config)
    if not evaluation_path.is_file():
        raise FileNotFoundError("ablation retrieval evaluation is missing")
    evaluation = read_json(evaluation_path)
    original_path = config.output_paths["evaluation"] / "smoke" / "90444_original_e2e.json"
    autogeo_path = config.output_paths["evaluation"] / "smoke" / "90444_autogeo_api_e2e.json"
    ra_path = config.output_paths["evaluation"] / "smoke" / "90444_ra_autogeo_e2e.json"
    original = read_json(original_path)
    autogeo = read_json(autogeo_path)
    ra = read_json(ra_path)
    initial_rank = inputs.target_record["initial_r1_rank"]

    def result(rank, score):
        return {
            "final_rank": rank,
            "r1_score": score,
            "rank_gain": initial_rank - rank,
            "entered_top5": rank <= config.ge_top_k,
        }

    query_aware = evaluation["query_aware_autogeo"]
    no_feedback = evaluation["no_feedback_multi_sample"]
    selected = no_feedback["selected_result"]
    payload: dict[str, object] = {
        "schema_version": "e7_retrieval_side_ablation_summary_v1",
        "query_id": EXPECTED_SMOKE_QUERY_ID,
        "target_document_id": inputs.target_record["target_document_id"],
        "methods": {
            "original_e2e": result(
                original["final_target_rank"], original["final_target_score"]
            ),
            "autogeo_api_e2e": result(
                autogeo["new_target_rank"], autogeo["new_target_score"]
            ),
            "query_aware_autogeo": result(
                query_aware["rank"], query_aware["score"]
            ),
            "no_feedback_multi_sample_c1": result(
                selected["rank"], selected["score"]
            ),
            "ra_autogeo_e2e": result(
                ra["final_target_rank"], ra["final_target_score"]
            ),
        },
        "no_feedback_diagnostics": no_feedback["candidate_diagnostics"],
        "no_feedback_formal_selected_output": "C1",
        "diagnostics_used_for_selection": False,
        "ge_geo_geu_executed": False,
        "ra_final_result_used_for_descriptive_comparison_only": True,
        "ra_candidate_or_trajectory_read": False,
        "test50_processed": False,
        "dev15_batch_run": False,
    }
    action, output_sha = _persist(_summary_path(config), payload, config)
    return {
        "stage": "summary",
        "action": action,
        "output_path": str(_summary_path(config)),
        "output_sha256": output_sha,
        "result": payload,
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", required=True, choices=("generate", "evaluate", "summary"))
    return parser


def main() -> None:
    stage = _parser().parse_args().stage
    runners = {
        "generate": run_generation,
        "evaluate": run_retrieval_evaluation,
        "summary": run_summary,
    }
    print(json.dumps(runners[stage](), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
