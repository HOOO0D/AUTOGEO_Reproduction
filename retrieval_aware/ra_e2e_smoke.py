"""Two-round retriever-in-the-loop RA-AutoGEO smoke for frozen query 90444."""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Sequence

from autogeo.evaluation.generative_engine import generate_answer_gemini
from autogeo.utils import call_gemini

from .baseline_e2e_smoke import (
    AUTOGEO_REWRITE_MODEL,
    EXPECTED_POOL_V2_EMBEDDINGS_SHA256,
    EXPECTED_POOL_V2_METADATA_SHA256,
    EXPECTED_RETRIEVER_CONFIG_SHA256,
    EXPECTED_RULE_FILE_SHA256,
    EXPECTED_SMOKE_QUERY_ID,
    EXPECTED_TARGET_MANIFEST_SHA256,
    GE_MODEL,
    _cache_paths,
    _load_retriever,
    build_method_top5_sources,
    calculate_existing_geu,
    calculate_target_e2e_geo,
    load_smoke_inputs,
    sha256_text,
)
from .config import E7Config
from .protocol import ProtocolId
from .utils import (
    canonical_json_sha256,
    normalized_text_sha256,
    read_json,
    sha256_file,
    write_e7_json_once,
)


EXPECTED_AUTOGEO_REWRITE_ARTIFACT_SHA256 = (
    "88bb6d3a0dc5c76f06690740a63d9766578c63764be84625086a2f48a989ad6d"
)
EXPECTED_AUTOGEO_CORE_SHA256 = (
    "460dd484a88861448fe436b55001a91b31013606abaac4fdbb90ea0b53453805"
)
EXPECTED_GEMINI_WRAPPER_SHA256 = (
    "7867128d953a15fe9a7a13e814c4ceeaba8bf7d706d4d75995c8e4958b8c6b3c"
)
EXPECTED_FILTERED_RULES_SHA256 = (
    "1b1d918ce956c07f3eb142b9465e5d5cf59786135a7a1c287aae3e58a091aff0"
)

REWRITE_TEMPERATURE = 0.7
ROUND_CANDIDATES = {1: ("C1", "C2", "C3"), 2: ("C4", "C5", "C6")}


@dataclass(frozen=True)
class RankedOption:
    source_id: str
    text: str
    rewrite_hash: str
    rank: int
    score: float
    top5_document_ids: tuple[str, ...]
    parent_candidate_id: str | None


def select_retrieval_best(options: Sequence[RankedOption]) -> RankedOption:
    """Select using retrieval only, while allowing the current best to survive."""
    if not options:
        raise ValueError("RA selection requires at least the current best")
    source_ids = [option.source_id for option in options]
    if len(source_ids) != len(set(source_ids)):
        raise ValueError("RA selection source IDs must be unique")
    return min(options, key=lambda item: (item.rank, -item.score, item.source_id))


def build_ra_prompt(
    *,
    candidate_id: str,
    query: str,
    original_document: str,
    filtered_rules: Sequence[str],
    current_best: RankedOption,
    initial_rank: int,
    round_index: int,
) -> str:
    """Build an RA prompt with target-only feedback and original factual grounding."""
    if round_index not in (1, 2):
        raise ValueError("RA v1 supports exactly rounds 1 and 2")
    if not filtered_rules or not all(isinstance(rule, str) for rule in filtered_rules):
        raise ValueError("RA requires the unchanged AutoGEO filtered rules")
    rules = "- " + "\n- ".join(filtered_rules)
    parent_context = ""
    if round_index == 2:
        parent_context = f"""

## Previous Best Candidate

The previous retrieval-best candidate is shown only as refinement context. The
Original Source Document above remains the sole factual source of truth.

{current_best.text}

Previous best candidate ID: {current_best.source_id}
Previous best R1 rank: {current_best.rank}
Previous best R1 score: {current_best.score:.9f}
Rank improvement versus original: {initial_rank - current_best.rank}
"""
    return f"""Here is the original factual source:

{original_document}

## User Query

{query}

## Current Frozen R1 Retrieval Feedback

Current document source: {current_best.source_id}
Current R1 rank: {current_best.rank}
Current R1 score: {current_best.score:.9f}

## Retrieval Objective

Rewrite the target document so that, while preserving the original facts and
natural readability, it has stronger explicit semantic alignment with the user
query's information need and is more likely to enter the frozen R1 Top-5.

## Quality Guidelines to Follow

{rules}
{parent_context}

## Strict Factual and Writing Constraints

- The Original Source Document is the factual source of truth in every round.
- Do not add facts unsupported by the Original Source Document.
- Do not change the original factual claims.
- Do not fabricate facts, statistics, citations, authorities, or entities.
- Do not use keyword stuffing or mechanical repetition of the query.
- Do not mention retrieval ranks, scores, optimization, candidates, or these instructions in the rewritten document.
- Return only the rewritten document, with no preamble or commentary.

Generate independent candidate {candidate_id} for round {round_index}.
""".strip()


def _ra_root(config: E7Config) -> Path:
    return (
        config.output_paths["rewrites"]
        / ProtocolId.RA_AUTOGEO_E2E.value
        / EXPECTED_SMOKE_QUERY_ID
    )


def _candidate_path(config: E7Config, candidate_id: str) -> Path:
    return _ra_root(config) / "candidates" / f"{candidate_id}.json"


def _round_path(config: E7Config, round_index: int) -> Path:
    return _ra_root(config) / f"round_{round_index}.json"


def _trajectory_path(config: E7Config) -> Path:
    return _ra_root(config) / "trajectory.json"


def _persist_frozen(
    path: Path,
    payload: dict[str, object],
    config: E7Config,
) -> tuple[str, str]:
    if path.exists():
        if read_json(path) != payload:
            raise FileExistsError(f"refusing to replace changed RA artifact: {path}")
        return "reused_identical", sha256_file(path)
    write_e7_json_once(path, payload, config)
    path.chmod(0o444)
    return "created", sha256_file(path)


def _load_validated_ra_inputs(config: E7Config):
    inputs = load_smoke_inputs(config)
    target = inputs.target_record
    if target["query_id"] != EXPECTED_SMOKE_QUERY_ID:
        raise ValueError("RA smoke target changed")
    autogeo_path = (
        config.output_paths["rewrites"]
        / ProtocolId.AUTOGEO_API_E2E.value
        / f"{EXPECTED_SMOKE_QUERY_ID}.json"
    )
    if sha256_file(autogeo_path) != EXPECTED_AUTOGEO_REWRITE_ARTIFACT_SHA256:
        raise ValueError("frozen AutoGEO_API-E2E sibling artifact changed")
    autogeo = read_json(autogeo_path)
    if autogeo.get("target_text_hash") != target["target_text_hash"]:
        raise AssertionError("RA and AutoGEO original target hashes differ")
    if autogeo.get("target_document_id") != target["target_document_id"]:
        raise AssertionError("RA and AutoGEO target identities differ")
    if autogeo.get("filtered_rules_sha256") != inputs.filtered_rules_sha256:
        raise AssertionError("RA and AutoGEO filtered_rules hashes differ")
    if inputs.filtered_rules_sha256 != EXPECTED_FILTERED_RULES_SHA256:
        raise AssertionError("RA filtered_rules no longer match frozen AutoGEO")
    if autogeo.get("rewrite_model") != AUTOGEO_REWRITE_MODEL:
        raise AssertionError("RA and AutoGEO must share the rewrite backend model")
    core_path = config.project_root / "autogeo" / "rewriters" / "core.py"
    gemini_path = config.project_root / "autogeo" / "utils" / "gemini.py"
    if sha256_file(core_path) != EXPECTED_AUTOGEO_CORE_SHA256:
        raise ValueError("original AutoGEO core implementation changed")
    if sha256_file(gemini_path) != EXPECTED_GEMINI_WRAPPER_SHA256:
        raise ValueError("original AutoGEO Gemini backend wrapper changed")
    rules_payload = read_json(inputs.rules_path)
    filtered_rules = rules_payload.get("filtered_rules")
    if canonical_json_sha256(filtered_rules) != EXPECTED_FILTERED_RULES_SHA256:
        raise ValueError("active filtered_rules changed")
    return inputs, autogeo, tuple(filtered_rules)


def _original_option(inputs, retriever, config: E7Config) -> RankedOption:
    target = inputs.target_record
    original_document = inputs.corpus.documents[
        inputs.corpus.index_of(target["target_document_id"])
    ]
    ranking = retriever.retrieve(
        target["query"],
        inputs.corpus,
        top_k=len(inputs.corpus.documents),
    )
    hit = next(item for item in ranking if item.document_id == target["target_document_id"])
    if hit.rank != target["initial_r1_rank"]:
        raise AssertionError("RA Round 0 rank differs from frozen manifest")
    if abs(hit.score - target["initial_r1_score"]) > 1e-7:
        raise AssertionError("RA Round 0 score differs from frozen manifest")
    return RankedOption(
        source_id="original",
        text=original_document.text,
        rewrite_hash=original_document.text_hash,
        rank=hit.rank,
        score=hit.score,
        top5_document_ids=tuple(item.document_id for item in ranking[: config.ge_top_k]),
        parent_candidate_id=None,
    )


def _option_from_candidate(payload: Mapping[str, object]) -> RankedOption:
    required_strings = ("candidate_id", "rewritten_text", "rewrite_hash")
    if not all(isinstance(payload.get(key), str) for key in required_strings):
        raise ValueError("candidate artifact lacks identity/text/hash")
    top5 = payload.get("top5_document_ids")
    if not isinstance(top5, list) or len(top5) != 5:
        raise ValueError("candidate artifact lacks actual Top-5")
    return RankedOption(
        source_id=payload["candidate_id"],
        text=payload["rewritten_text"],
        rewrite_hash=payload["rewrite_hash"],
        rank=int(payload["rank"]),
        score=float(payload["score"]),
        top5_document_ids=tuple(top5),
        parent_candidate_id=payload.get("parent_candidate_id"),
    )


def _rank_candidate(
    *,
    candidate_id: str,
    round_index: int,
    parent: RankedOption,
    rewritten_text: str,
    inputs,
    retriever,
    config: E7Config,
) -> RankedOption:
    target = inputs.target_record
    result = retriever.rerank_with_rewritten_target(
        target["query"],
        inputs.corpus,
        target["target_document_id"],
        rewritten_text,
        top_k=config.ge_top_k,
    )
    if result.original_target_rank != target["initial_r1_rank"]:
        raise AssertionError("RA candidate changed the sibling original rank")
    if result.reencoded_document_ids != (target["target_document_id"],):
        raise AssertionError("RA candidate re-encoded more than target D*")
    if not result.competitor_invariant_verified:
        raise AssertionError("RA competitor invariant failed")
    return RankedOption(
        source_id=candidate_id,
        text=rewritten_text,
        rewrite_hash=normalized_text_sha256(rewritten_text),
        rank=result.new_target_rank,
        score=result.new_target_score,
        top5_document_ids=tuple(item.document_id for item in result.new_top_k),
        parent_candidate_id=parent.source_id,
    )


def _candidate_payload(
    *,
    option: RankedOption,
    round_index: int,
    parent: RankedOption,
    original: RankedOption,
    selected: RankedOption,
    prompt_sha256: str,
    inputs,
) -> dict[str, object]:
    entered = option.rank <= 5
    is_selected = option.source_id == selected.source_id
    stop_reason = None
    if is_selected:
        stop_reason = "entered_top5" if selected.rank <= 5 else (
            "continue_to_round_2" if round_index == 1 else "max_rounds_reached"
        )
    return {
        "schema_version": "e7_ra_autogeo_candidate_v1",
        "protocol_id": ProtocolId.RA_AUTOGEO_E2E.value,
        "query_id": EXPECTED_SMOKE_QUERY_ID,
        "round": round_index,
        "candidate_id": option.source_id,
        "parent_candidate_id": parent.source_id,
        "target_document_id": inputs.target_record["target_document_id"],
        "original_target_hash": inputs.target_record["target_text_hash"],
        "rewritten_text": option.text,
        "rewrite_hash": option.rewrite_hash,
        "rewrite_raw_text_sha256": sha256_text(option.text),
        "prompt_sha256": prompt_sha256,
        "rank": option.rank,
        "score": option.score,
        "rank_delta_vs_original": original.rank - option.rank,
        "rank_delta_vs_parent": parent.rank - option.rank,
        "entered_top5": entered,
        "top5_document_ids": list(option.top5_document_ids),
        "selected": is_selected,
        "selection_reason": (
            "minimum_rank_then_highest_score_then_candidate_id_including_current_best"
            if is_selected
            else "not_selected_by_frozen_retrieval_only_order"
        ),
        "stop_reason": stop_reason,
        "rules_file_sha256": EXPECTED_RULE_FILE_SHA256,
        "filtered_rules_sha256": EXPECTED_FILTERED_RULES_SHA256,
        "rewrite_model": AUTOGEO_REWRITE_MODEL,
        "rewrite_temperature": REWRITE_TEMPERATURE,
        "retriever_id": "r1_independent_dense",
        "retriever_config_sha256": EXPECTED_RETRIEVER_CONFIG_SHA256,
        "corpus_manifest_sha256": inputs.corpus.manifest_sha256,
        "target_manifest_sha256": EXPECTED_TARGET_MANIFEST_SHA256,
        "only_target_embedding_recomputed": True,
        "competitor_invariant_verified": True,
        "geo_geu_used_for_selection": False,
    }


def _generate_round_candidates(
    *,
    round_index: int,
    parent: RankedOption,
    original: RankedOption,
    inputs,
    filtered_rules: Sequence[str],
    retriever,
    config: E7Config,
) -> tuple[list[RankedOption], dict[str, str]]:
    options: list[RankedOption] = []
    prompt_hashes: dict[str, str] = {}
    for candidate_id in ROUND_CANDIDATES[round_index]:
        prompt = build_ra_prompt(
            candidate_id=candidate_id,
            query=inputs.target_record["query"],
            original_document=original.text,
            filtered_rules=filtered_rules,
            current_best=parent,
            initial_rank=original.rank,
            round_index=round_index,
        )
        prompt_hashes[candidate_id] = sha256_text(prompt)
        rewritten = call_gemini(
            prompt,
            model_name=AUTOGEO_REWRITE_MODEL,
            temperature=REWRITE_TEMPERATURE,
        )
        if not isinstance(rewritten, str) or not rewritten.strip():
            raise ValueError(f"RA rewrite backend returned empty {candidate_id}")
        options.append(
            _rank_candidate(
                candidate_id=candidate_id,
                round_index=round_index,
                parent=parent,
                rewritten_text=rewritten,
                inputs=inputs,
                retriever=retriever,
                config=config,
            )
        )
    return options, prompt_hashes


def _validate_cache_unchanged(config: E7Config) -> None:
    embedding_path, metadata_path = _cache_paths(config)
    if sha256_file(embedding_path) != EXPECTED_POOL_V2_EMBEDDINGS_SHA256:
        raise AssertionError("RA modified frozen Pool V2 embeddings")
    if sha256_file(metadata_path) != EXPECTED_POOL_V2_METADATA_SHA256:
        raise AssertionError("RA modified frozen Pool V2 cache metadata")


def run_round1(config: E7Config | None = None) -> dict[str, object]:
    config = config or E7Config()
    inputs, _, filtered_rules = _load_validated_ra_inputs(config)
    round_path = _round_path(config, 1)
    if round_path.exists():
        payload = read_json(round_path)
        return {
            "stage": "round_1",
            "action": "reused_identical",
            "output_path": str(round_path),
            "output_sha256": sha256_file(round_path),
            "result": payload,
        }
    if any(_candidate_path(config, candidate_id).exists() for candidate_id in ROUND_CANDIDATES[1]):
        raise FileExistsError("partial Round 1 candidate artifacts require manual audit")

    retriever = _load_retriever(inputs, config)
    original = _original_option(inputs, retriever, config)
    candidates, prompt_hashes = _generate_round_candidates(
        round_index=1,
        parent=original,
        original=original,
        inputs=inputs,
        filtered_rules=filtered_rules,
        retriever=retriever,
        config=config,
    )
    selected = select_retrieval_best((original, *candidates))
    candidate_hashes: dict[str, str] = {}
    for option in candidates:
        payload = _candidate_payload(
            option=option,
            round_index=1,
            parent=original,
            original=original,
            selected=selected,
            prompt_sha256=prompt_hashes[option.source_id],
            inputs=inputs,
        )
        _, candidate_hashes[option.source_id] = _persist_frozen(
            _candidate_path(config, option.source_id), payload, config
        )
    stop_reason = "entered_top5" if selected.rank <= 5 else "continue_to_round_2"
    round_payload: dict[str, object] = {
        "schema_version": "e7_ra_autogeo_round_v1",
        "query_id": EXPECTED_SMOKE_QUERY_ID,
        "round": 1,
        "parent_candidate_id": "original",
        "candidate_ids": list(ROUND_CANDIDATES[1]),
        "candidate_artifact_sha256": candidate_hashes,
        "selection_pool": ["original", *ROUND_CANDIDATES[1]],
        "selection_policy": ["minimum_rank", "maximum_score", "candidate_id_ascending"],
        "selected_source": selected.source_id,
        "selected_rank": selected.rank,
        "selected_score": selected.score,
        "entered_top5": selected.rank <= 5,
        "stop_reason": stop_reason,
        "current_best_preserved_as_candidate": True,
        "geo_geu_used_for_selection": False,
    }
    action, output_sha = _persist_frozen(round_path, round_payload, config)
    _validate_cache_unchanged(config)
    return {
        "stage": "round_1",
        "action": action,
        "output_path": str(round_path),
        "output_sha256": output_sha,
        "result": round_payload,
    }


def _load_round_best(round_payload: Mapping[str, object], inputs, retriever, config):
    source = round_payload.get("selected_source")
    if source == "original":
        return _original_option(inputs, retriever, config)
    if not isinstance(source, str):
        raise ValueError("round selected_source is invalid")
    return _option_from_candidate(read_json(_candidate_path(config, source)))


def _write_trajectory(
    *,
    inputs,
    original: RankedOption,
    round1: Mapping[str, object],
    round2: Mapping[str, object],
    final: RankedOption,
    config: E7Config,
) -> tuple[str, str]:
    candidates = []
    for round_index in (1, 2):
        for candidate_id in ROUND_CANDIDATES[round_index]:
            path = _candidate_path(config, candidate_id)
            if path.exists():
                item = read_json(path)
                candidates.append(
                    {
                        key: item.get(key)
                        for key in (
                            "round",
                            "candidate_id",
                            "parent_candidate_id",
                            "rewrite_hash",
                            "rank",
                            "score",
                            "rank_delta_vs_original",
                            "rank_delta_vs_parent",
                            "selected",
                            "entered_top5",
                            "selection_reason",
                            "stop_reason",
                        )
                    }
                )
    payload: dict[str, object] = {
        "schema_version": "e7_ra_autogeo_trajectory_v1",
        "protocol_id": ProtocolId.RA_AUTOGEO_E2E.value,
        "query_id": EXPECTED_SMOKE_QUERY_ID,
        "target_document_id": inputs.target_record["target_document_id"],
        "original_target_hash": inputs.target_record["target_text_hash"],
        "initial_rank": original.rank,
        "initial_score": original.score,
        "round_0": {
            "selected_source": "original",
            "rank": original.rank,
            "score": original.score,
        },
        "round_1": dict(round1),
        "round_2": dict(round2),
        "candidates": candidates,
        "final_selected_source": final.source_id,
        "final_rank": final.rank,
        "final_score": final.score,
        "rank_gain": original.rank - final.rank,
        "entered_top5": final.rank <= config.ge_top_k,
        "stop_reason": round2["stop_reason"],
        "monotonic_non_worsening_verified": (
            final.rank < original.rank
            or (final.rank == original.rank and final.score >= original.score)
        ),
        "filtered_rules_sha256": EXPECTED_FILTERED_RULES_SHA256,
        "retriever_config_sha256": EXPECTED_RETRIEVER_CONFIG_SHA256,
        "corpus_manifest_sha256": inputs.corpus.manifest_sha256,
        "target_manifest_sha256": EXPECTED_TARGET_MANIFEST_SHA256,
        "geo_geu_used_for_selection": False,
    }
    action, output_sha = _persist_frozen(_trajectory_path(config), payload, config)
    return action, output_sha


def run_round2(config: E7Config | None = None) -> dict[str, object]:
    config = config or E7Config()
    inputs, _, filtered_rules = _load_validated_ra_inputs(config)
    round1_path = _round_path(config, 1)
    if not round1_path.is_file():
        raise FileNotFoundError("Round 1 must complete before Round 2")
    round1 = read_json(round1_path)
    round2_path = _round_path(config, 2)
    if round2_path.exists() and _trajectory_path(config).exists():
        payload = read_json(round2_path)
        return {
            "stage": "round_2",
            "action": "reused_identical",
            "output_path": str(round2_path),
            "output_sha256": sha256_file(round2_path),
            "trajectory_sha256": sha256_file(_trajectory_path(config)),
            "result": payload,
        }
    if any(_candidate_path(config, candidate_id).exists() for candidate_id in ROUND_CANDIDATES[2]):
        raise FileExistsError("partial Round 2 candidate artifacts require manual audit")

    retriever = _load_retriever(inputs, config)
    original = _original_option(inputs, retriever, config)
    parent = _load_round_best(round1, inputs, retriever, config)
    if parent.rank <= config.ge_top_k:
        round_payload: dict[str, object] = {
            "schema_version": "e7_ra_autogeo_round_v1",
            "query_id": EXPECTED_SMOKE_QUERY_ID,
            "round": 2,
            "executed": False,
            "candidate_ids": [],
            "selection_pool": [parent.source_id],
            "selected_source": parent.source_id,
            "selected_rank": parent.rank,
            "selected_score": parent.score,
            "entered_top5": True,
            "stop_reason": "entered_top5",
            "early_stop_after_round": 1,
            "geo_geu_used_for_selection": False,
        }
        action, output_sha = _persist_frozen(round2_path, round_payload, config)
        trajectory_action, trajectory_sha = _write_trajectory(
            inputs=inputs,
            original=original,
            round1=round1,
            round2=round_payload,
            final=parent,
            config=config,
        )
        return {
            "stage": "round_2",
            "action": action,
            "output_path": str(round2_path),
            "output_sha256": output_sha,
            "trajectory_action": trajectory_action,
            "trajectory_sha256": trajectory_sha,
            "result": round_payload,
        }

    candidates, prompt_hashes = _generate_round_candidates(
        round_index=2,
        parent=parent,
        original=original,
        inputs=inputs,
        filtered_rules=filtered_rules,
        retriever=retriever,
        config=config,
    )
    selected = select_retrieval_best((parent, *candidates))
    candidate_hashes: dict[str, str] = {}
    for option in candidates:
        payload = _candidate_payload(
            option=option,
            round_index=2,
            parent=parent,
            original=original,
            selected=selected,
            prompt_sha256=prompt_hashes[option.source_id],
            inputs=inputs,
        )
        _, candidate_hashes[option.source_id] = _persist_frozen(
            _candidate_path(config, option.source_id), payload, config
        )
    stop_reason = "entered_top5" if selected.rank <= 5 else "max_rounds_reached"
    round_payload = {
        "schema_version": "e7_ra_autogeo_round_v1",
        "query_id": EXPECTED_SMOKE_QUERY_ID,
        "round": 2,
        "executed": True,
        "parent_candidate_id": parent.source_id,
        "candidate_ids": list(ROUND_CANDIDATES[2]),
        "candidate_artifact_sha256": candidate_hashes,
        "selection_pool": [parent.source_id, *ROUND_CANDIDATES[2]],
        "selection_policy": ["minimum_rank", "maximum_score", "candidate_id_ascending"],
        "selected_source": selected.source_id,
        "selected_rank": selected.rank,
        "selected_score": selected.score,
        "entered_top5": selected.rank <= 5,
        "stop_reason": stop_reason,
        "current_best_preserved_as_candidate": True,
        "geo_geu_used_for_selection": False,
    }
    action, output_sha = _persist_frozen(round2_path, round_payload, config)
    trajectory_action, trajectory_sha = _write_trajectory(
        inputs=inputs,
        original=original,
        round1=round1,
        round2=round_payload,
        final=selected,
        config=config,
    )
    _validate_cache_unchanged(config)
    return {
        "stage": "round_2",
        "action": action,
        "output_path": str(round2_path),
        "output_sha256": output_sha,
        "trajectory_action": trajectory_action,
        "trajectory_sha256": trajectory_sha,
        "result": round_payload,
    }


def _load_final_option(trajectory, inputs, retriever, config):
    source = trajectory.get("final_selected_source")
    if source == "original":
        return _original_option(inputs, retriever, config)
    if not isinstance(source, str):
        raise ValueError("trajectory final source is invalid")
    return _option_from_candidate(read_json(_candidate_path(config, source)))


def run_evaluation(config: E7Config | None = None) -> dict[str, object]:
    config = config or E7Config()
    inputs, _, _ = _load_validated_ra_inputs(config)
    trajectory_path = _trajectory_path(config)
    if not trajectory_path.is_file():
        raise FileNotFoundError("RA trajectory must complete before final evaluation")
    trajectory = read_json(trajectory_path)
    destination = (
        config.output_paths["evaluation"]
        / "smoke"
        / f"{EXPECTED_SMOKE_QUERY_ID}_ra_autogeo_e2e.json"
    )
    if destination.exists():
        payload = read_json(destination)
        return {
            "stage": ProtocolId.RA_AUTOGEO_E2E.value,
            "action": "reused_identical",
            "output_path": str(destination),
            "output_sha256": sha256_file(destination),
            "result": payload,
        }

    retriever = _load_retriever(inputs, config)
    original = _original_option(inputs, retriever, config)
    final = _load_final_option(trajectory, inputs, retriever, config)
    target_id = inputs.target_record["target_document_id"]
    if final.source_id == "original":
        final_rank = original.rank
        final_score = original.score
        top5_hits = tuple(
            __import__(
                "retrieval_aware.frozen_local_retriever",
                fromlist=["RetrievalHit"],
            ).RetrievalHit(document_id=document_id, rank=index, score=0.0)
            for index, document_id in enumerate(original.top5_document_ids, start=1)
        )
        replacement = None
    else:
        reranked = retriever.rerank_with_rewritten_target(
            inputs.target_record["query"],
            inputs.corpus,
            target_id,
            final.text,
            top_k=config.ge_top_k,
        )
        if reranked.reencoded_document_ids != (target_id,):
            raise AssertionError("final RA rerank encoded more than D*")
        final_rank = reranked.new_target_rank
        final_score = reranked.new_target_score
        top5_hits = reranked.new_top_k
        replacement = final.text
    if final_rank != trajectory["final_rank"] or abs(final_score - trajectory["final_score"]) > 1e-7:
        raise AssertionError("final RA evaluation differs from selected trajectory")
    top5_ids, top5_sources = build_method_top5_sources(
        inputs.corpus,
        top5_hits,
        target_document_id=target_id,
        replacement_target_text=replacement,
    )
    if top5_ids != final.top5_document_ids:
        raise AssertionError("final RA actual Top-5 differs from selection rerank")
    target_source_index = top5_ids.index(target_id) if target_id in top5_ids else None
    if (final_rank <= config.ge_top_k) != (target_source_index is not None):
        raise AssertionError("final RA rank and dynamic source index disagree")
    response = generate_answer_gemini(
        inputs.target_record["query"],
        list(top5_sources),
        model_name=GE_MODEL,
    )
    target_geo = calculate_target_e2e_geo(
        response,
        target_source_index,
        ge_top_k=config.ge_top_k,
    )
    geu, keypoints_available = calculate_existing_geu(
        EXPECTED_SMOKE_QUERY_ID,
        inputs.target_record["query"],
        response,
        top5_sources,
        config,
    )
    _validate_cache_unchanged(config)
    payload: dict[str, object] = {
        "schema_version": "e7_ra_autogeo_e2e_smoke_v1",
        "protocol_id": ProtocolId.RA_AUTOGEO_E2E.value,
        "query_id": EXPECTED_SMOKE_QUERY_ID,
        "query": inputs.target_record["query"],
        "target_document_id": target_id,
        "target_source": inputs.target_record["target_source"],
        "original_target_hash": inputs.target_record["target_text_hash"],
        "selected_source": final.source_id,
        "final_rewrite_hash": final.rewrite_hash,
        "initial_r1_rank": original.rank,
        "initial_r1_score": original.score,
        "final_target_rank": final_rank,
        "final_target_score": final_score,
        "rank_gain": original.rank - final_rank,
        "entered_top5": final_rank <= config.ge_top_k,
        "top5_document_ids": list(top5_ids),
        "target_source_index_in_top5": target_source_index,
        "target_e2e_geo": target_geo,
        "ge_response": response,
        "geu": geu,
        "keypoints_available": keypoints_available,
        "ge_generation_parameters": {
            "model": GE_MODEL,
            "temperature": None,
            "top_p": None,
            "seed": None,
            "max_tokens": None,
            "provider": None,
            "note": "unspecified values use the existing model/provider defaults",
        },
        "rewrite_generation_parameters": {
            "model": AUTOGEO_REWRITE_MODEL,
            "temperature": REWRITE_TEMPERATURE,
            "top_p": None,
            "seed": None,
            "max_tokens": None,
        },
        "trajectory_path": str(trajectory_path.resolve()),
        "trajectory_sha256": sha256_file(trajectory_path),
        "filtered_rules_sha256": EXPECTED_FILTERED_RULES_SHA256,
        "rules_file_sha256": EXPECTED_RULE_FILE_SHA256,
        "retriever_id": "r1_independent_dense",
        "retriever_config_sha256": EXPECTED_RETRIEVER_CONFIG_SHA256,
        "corpus_manifest_sha256": inputs.corpus.manifest_sha256,
        "embedding_cache_sha256": EXPECTED_POOL_V2_EMBEDDINGS_SHA256,
        "target_manifest_sha256": EXPECTED_TARGET_MANIFEST_SHA256,
        "only_target_embedding_recomputed": final.source_id != "original",
        "competitor_invariant_verified": True,
        "test50_processed": False,
    }
    action, output_sha = _persist_frozen(destination, payload, config)
    return {
        "stage": ProtocolId.RA_AUTOGEO_E2E.value,
        "action": action,
        "output_path": str(destination),
        "output_sha256": output_sha,
        "result": payload,
    }


def run_summary(config: E7Config | None = None) -> dict[str, object]:
    config = config or E7Config()
    inputs, _, _ = _load_validated_ra_inputs(config)
    original_path = config.output_paths["evaluation"] / "smoke" / "90444_original_e2e.json"
    autogeo_path = config.output_paths["evaluation"] / "smoke" / "90444_autogeo_api_e2e.json"
    ra_path = config.output_paths["evaluation"] / "smoke" / "90444_ra_autogeo_e2e.json"
    for path in (original_path, autogeo_path, ra_path, _trajectory_path(config)):
        if not path.is_file():
            raise FileNotFoundError(f"required sibling result is missing: {path}")
    original = read_json(original_path)
    autogeo = read_json(autogeo_path)
    ra = read_json(ra_path)
    trajectory = read_json(_trajectory_path(config))
    if len({original["target_document_id"], autogeo["target_document_id"], ra["target_document_id"]}) != 1:
        raise AssertionError("Track B smoke siblings use different targets")
    payload: dict[str, object] = {
        "schema_version": "e7_ra_autogeo_smoke_summary_v1",
        "query_id": EXPECTED_SMOKE_QUERY_ID,
        "target_document_id": inputs.target_record["target_document_id"],
        "original": {"rank": original["final_target_rank"]},
        "autogeo_api": {"rank": autogeo["new_target_rank"]},
        "ra_autogeo": {
            "round_1": trajectory["round_1"],
            "round_2": trajectory["round_2"],
            "candidates": trajectory["candidates"],
            "selected_source": ra["selected_source"],
            "final_rank": ra["final_target_rank"],
            "rank_gain": ra["rank_gain"],
            "entered_top5": ra["entered_top5"],
            "target_e2e_geo": ra["target_e2e_geo"],
            "geu": ra["geu"],
            "ge_generation_parameters": ra["ge_generation_parameters"],
        },
        "invariants": {
            "same_original_target_hash_as_autogeo": True,
            "same_filtered_rules_hash_as_autogeo": True,
            "same_rewrite_backend_model_as_autogeo": True,
            "only_target_embeddings_recomputed": True,
            "competitors_unchanged": True,
            "r1_unchanged": True,
            "pool_v2_unchanged": True,
            "target_manifest_unchanged": True,
            "original_autogeo_core_unchanged": True,
        },
        "test50_processed": False,
        "dev15_batch_run": False,
    }
    destination = config.output_paths["evaluation"] / "smoke" / "90444_ra_summary.json"
    action, output_sha = _persist_frozen(destination, payload, config)
    return {
        "stage": "summary",
        "action": action,
        "output_path": str(destination),
        "output_sha256": output_sha,
        "result": payload,
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--stage",
        required=True,
        choices=("round1", "round2", "evaluate", "summary"),
    )
    return parser


def main() -> None:
    stage = _parser().parse_args().stage
    runners = {
        "round1": run_round1,
        "round2": run_round2,
        "evaluate": run_evaluation,
        "summary": run_summary,
    }
    print(json.dumps(runners[stage](), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
