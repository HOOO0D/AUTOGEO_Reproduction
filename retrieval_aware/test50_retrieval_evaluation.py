"""Frozen TEST50 retrieval-only evaluation for E7.

This sibling runner inherits the completed DEV20 protocol verbatim.  It has no
``all`` stage: generation, retrieval evaluation, RA, and summary must be run
explicitly.  Importing the module performs no Test50 reads, model loads, or
external calls.
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

from autogeo.utils import call_gemini

from .baseline_e2e_smoke import (
    AUTOGEO_REWRITE_MODEL,
    EXPECTED_RETRIEVER_CONFIG_SHA256,
    EXPECTED_RULE_FILE_SHA256,
    _load_rules,
    rewrite_selected_target_with_stock_autogeo,
    sha256_text,
)
from .config import E7Config
from .frozen_local_retriever import (
    FrozenLocalCorpus,
    FrozenLocalRetriever,
    load_embedding_cache,
    load_local_corpus,
)
from .independent_retriever import BGEBaseEnV15Encoder
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
from .test50_pool_v2 import test_pool_v2_paths
from .test50_pool_v2_embeddings import (
    FROZEN_R1_CONFIG_SHA256,
    TEST_POOL_V2_CACHE_NAMESPACE,
    assert_frozen_r1_contract,
)
from .utils import (
    assert_e7_output_path,
    canonical_json_sha256,
    normalized_text_sha256,
    read_json,
    sha256_file,
    write_e7_json_once,
)


METHOD_IDS = (
    "original_e2e",
    "autogeo_api_e2e",
    "query_aware_autogeo",
    "no_feedback_multi_sample",
    "ra_autogeo_e2e",
)
EXPECTED_TEST_QUERY_COUNT = 50

# Frozen DEV code/artifacts are the protocol authority for this sibling runner.
EXPECTED_DEV_RETRIEVAL_IMPLEMENTATION_SHA256 = (
    "4527b452df3b29c726e79fa1488902bf910cc24027034a5f476f18f5e5f91779"
)
EXPECTED_DEV_QUERY_AWARE_IMPLEMENTATION_SHA256 = (
    "e9d5c516f089223a43d4ab5780e2abaf582445fad4526ebcfbdfd52ccfad023b"
)
EXPECTED_DEV_RA_IMPLEMENTATION_SHA256 = (
    "9109d499ee60b7350b0e9d19d1b6240ef8678db3ea52dc707edd18f11ce70e4e"
)
EXPECTED_AUTOGEO_CORE_SHA256 = (
    "460dd484a88861448fe436b55001a91b31013606abaac4fdbb90ea0b53453805"
)
EXPECTED_GEMINI_WRAPPER_SHA256 = (
    "7867128d953a15fe9a7a13e814c4ceeaba8bf7d706d4d75995c8e4958b8c6b3c"
)
EXPECTED_DEV_RETRIEVAL_SUMMARY_SHA256 = (
    "0c744a39bf24b2bd7013761e01f5622e3d04be1e9d55335b88d335cacdf8bc93"
)
EXPECTED_DEV_RETRIEVAL_PER_QUERY_SHA256 = (
    "0ccafdf29f2c6af2d0432dcad49d85e85976d5d28654c8ea9c1ab5188dfa6eba"
)
EXPECTED_PROMPT_PROBE_SHA256 = {
    "query_aware_and_no_feedback": (
        "ea45429cd663aaf549e0f36687230eaa062f3e159e03ed0c332a3188a0f8063a"
    ),
    "ra_round_1": (
        "c8801648f634b7f67f3b4b6a05390daa03e293beb340f73df7af13ac28154fd3"
    ),
    "ra_round_2": (
        "4288fe4adc603d1390bbb07d27eb6e15fdba52d61befbd5cd3b2b7bde17c9c1d"
    ),
}


@dataclass(frozen=True)
class TestInputs:
    records: tuple[dict[str, Any], ...]
    eligible_records: tuple[dict[str, Any], ...]
    corpus: FrozenLocalCorpus
    target_manifest_path: Path
    corpus_manifest_path: Path
    target_manifest_sha256: str
    corpus_manifest_sha256: str
    rules_path: Path
    filtered_rules: tuple[str, ...]
    protocol_contract: dict[str, Any]


@dataclass(frozen=True)
class RAPolicyResult:
    selected: RankedOption
    rounds: tuple[dict[str, Any], ...]
    candidates: tuple[RankedOption, ...]
    early_stop: bool


def _root(config: E7Config) -> Path:
    return config.output_paths["evaluation"] / "test50_retrieval"


def _rewrite_root(config: E7Config) -> Path:
    return config.output_paths["rewrites"] / "test50_retrieval"


def _baseline_rewrite_path(config: E7Config, method: str, query_id: str) -> Path:
    return _rewrite_root(config) / method / query_id / "rewrite.json"


def _nf_candidate_path(config: E7Config, query_id: str, candidate_id: str) -> Path:
    return (
        _rewrite_root(config)
        / "no_feedback_multi_sample"
        / query_id
        / f"{candidate_id}.json"
    )


def _nf_selection_path(config: E7Config, query_id: str) -> Path:
    return (
        _rewrite_root(config)
        / "no_feedback_multi_sample"
        / query_id
        / "selection.pre_retrieval.json"
    )


def _ra_root(config: E7Config, query_id: str) -> Path:
    return _rewrite_root(config) / "ra_autogeo_e2e" / query_id


def _baseline_result_path(config: E7Config, query_id: str) -> Path:
    return _root(config) / "per_query" / f"{query_id}.baseline.json"


def _ra_result_path(config: E7Config, query_id: str) -> Path:
    return _root(config) / "per_query" / f"{query_id}.ra.json"


def _test_cache_paths(config: E7Config) -> tuple[Path, Path]:
    root = (
        config.output_paths["embeddings"]
        / TEST_POOL_V2_CACHE_NAMESPACE
        / "test_corpus_cache"
    )
    return root / "embeddings.npy", root / "metadata.json"


def _persist(path: Path, payload: Any, config: E7Config) -> tuple[str, str]:
    if path.exists():
        if read_json(path) != payload:
            raise FileExistsError(f"refusing to replace changed TEST50 artifact: {path}")
        if path.stat().st_mode & 0o222:
            raise PermissionError(f"immutable TEST50 artifact became writable: {path}")
        return "reused_identical", sha256_file(path)
    write_e7_json_once(path, payload, config)
    path.chmod(0o444)
    return "created", sha256_file(path)


def _write_text_once(path: Path, text: str, config: E7Config) -> tuple[str, str]:
    destination = assert_e7_output_path(path, config)
    if destination.exists():
        if destination.read_text(encoding="utf-8") != text:
            raise FileExistsError(f"refusing to replace changed TEST50 report: {path}")
        if destination.stat().st_mode & 0o222:
            raise PermissionError(f"immutable TEST50 report became writable: {path}")
        return "reused_identical", sha256_file(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary_name: str | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=destination.parent,
            prefix=f".{destination.name}.",
            suffix=".tmp",
            delete=False,
        ) as handle:
            temporary_name = handle.name
            handle.write(text)
        os.replace(temporary_name, destination)
    finally:
        if temporary_name is not None and Path(temporary_name).exists():
            Path(temporary_name).unlink()
    destination.chmod(0o444)
    return "created", sha256_file(destination)


def prompt_probe_hashes() -> dict[str, str]:
    query = "__E7_PROTOCOL_QUERY__"
    document = "__E7_PROTOCOL_DOCUMENT__"
    rules = ("__E7_RULE_1__", "__E7_RULE_2__")
    query_aware = build_query_aware_no_feedback_prompt(
        query=query,
        original_document=document,
        filtered_rules=rules,
    )
    original = RankedOption(
        source_id="original",
        text=document,
        rewrite_hash="h",
        rank=7,
        score=0.123456789,
        top5_document_ids=("a", "b", "c", "d", "e"),
        parent_candidate_id=None,
    )
    round_1 = build_ra_prompt(
        candidate_id="C1",
        query=query,
        original_document=document,
        filtered_rules=rules,
        current_best=original,
        initial_rank=7,
        round_index=1,
    )
    previous = RankedOption(
        source_id="C1",
        text="__E7_PREVIOUS_BEST__",
        rewrite_hash="h2",
        rank=6,
        score=0.234567891,
        top5_document_ids=("a", "b", "c", "d", "e"),
        parent_candidate_id="original",
    )
    round_2 = build_ra_prompt(
        candidate_id="C4",
        query=query,
        original_document=document,
        filtered_rules=rules,
        current_best=previous,
        initial_rank=7,
        round_index=2,
    )
    return {
        "query_aware_and_no_feedback": sha256_text(query_aware),
        "ra_round_1": sha256_text(round_1),
        "ra_round_2": sha256_text(round_2),
    }


def validate_prompt_contract(
    expected: Mapping[str, str] = EXPECTED_PROMPT_PROBE_SHA256,
) -> dict[str, str]:
    actual = prompt_probe_hashes()
    if actual != dict(expected):
        raise ValueError(f"frozen DEV prompt behavior changed: {actual}")
    return actual


def validate_protocol_scalars(
    *,
    ge_top_k: int,
    candidate_count: int,
    max_rounds: int,
    r1_config_sha256: str,
) -> None:
    if (ge_top_k, candidate_count, max_rounds) != (5, 3, 2):
        raise ValueError("frozen Top-5/candidate/round protocol changed")
    if r1_config_sha256 != FROZEN_R1_CONFIG_SHA256:
        raise ValueError("frozen R1 config hash changed")


def validate_dev_protocol_contract(config: E7Config) -> dict[str, Any]:
    config.validate()
    r1_hash = assert_frozen_r1_contract(config)
    validate_protocol_scalars(
        ge_top_k=config.ge_top_k,
        candidate_count=config.num_candidates_per_round,
        max_rounds=config.max_rounds,
        r1_config_sha256=r1_hash,
    )
    if tuple(NO_FEEDBACK_CANDIDATE_IDS) != ("C1", "C2", "C3"):
        raise ValueError("DEV no-feedback candidate IDs changed")
    if ROUND_CANDIDATES != {1: ("C1", "C2", "C3"), 2: ("C4", "C5", "C6")}:
        raise ValueError("DEV RA round candidate IDs changed")
    if AUTOGEO_REWRITE_MODEL != "gemini-2.5-pro" or REWRITE_TEMPERATURE != 0.7:
        raise ValueError("DEV rewrite backend/model/generation config changed")

    source_dir = Path(__file__).resolve().parent
    frozen_files = {
        source_dir / "dev20_retrieval_evaluation.py": EXPECTED_DEV_RETRIEVAL_IMPLEMENTATION_SHA256,
        source_dir / "retrieval_ablation_smoke.py": EXPECTED_DEV_QUERY_AWARE_IMPLEMENTATION_SHA256,
        source_dir / "ra_e2e_smoke.py": EXPECTED_DEV_RA_IMPLEMENTATION_SHA256,
        config.project_root / "autogeo" / "rewriters" / "core.py": EXPECTED_AUTOGEO_CORE_SHA256,
        config.project_root / "autogeo" / "utils" / "gemini.py": EXPECTED_GEMINI_WRAPPER_SHA256,
        config.output_paths["evaluation"] / "dev20_retrieval" / "dev20_retrieval_summary.json": EXPECTED_DEV_RETRIEVAL_SUMMARY_SHA256,
        config.output_paths["evaluation"] / "dev20_retrieval" / "dev20_retrieval_per_query.jsonl": EXPECTED_DEV_RETRIEVAL_PER_QUERY_SHA256,
    }
    for path, expected in frozen_files.items():
        if sha256_file(path) != expected:
            raise ValueError(f"frozen DEV protocol artifact changed: {path}")
    dev_summary = read_json(
        config.output_paths["evaluation"]
        / "dev20_retrieval"
        / "dev20_retrieval_summary.json"
    )
    if dev_summary.get("formal_no_feedback_output") != "C1":
        raise ValueError("DEV formal no-feedback output changed")
    if dev_summary.get("retriever_config_sha256") != r1_hash:
        raise ValueError("DEV summary R1 config changed")
    if dev_summary.get("filtered_rules_sha256") != EXPECTED_FILTERED_RULES_SHA256:
        raise ValueError("DEV summary filtered_rules changed")
    prompt_hashes = validate_prompt_contract()

    # Execute retrieval-only selection probes to bind rank/score/ID order and
    # current-best regression protection to the frozen DEV selector.
    base = RankedOption("original", "o", "h0", 7, 0.2, (), None)
    better_low_score = RankedOption("C1", "1", "h1", 6, 0.1, (), "original")
    better_high_score = RankedOption("C2", "2", "h2", 6, 0.3, (), "original")
    if select_retrieval_best((base, better_low_score, better_high_score)).source_id != "C2":
        raise ValueError("DEV RA minimum-rank/maximum-score selection changed")
    tie_a = RankedOption("C1", "1", "h1", 6, 0.3, (), "original")
    tie_b = RankedOption("C2", "2", "h2", 6, 0.3, (), "original")
    if select_retrieval_best((tie_b, tie_a)).source_id != "C1":
        raise ValueError("DEV RA candidate-ID tie-break changed")
    worse = RankedOption("C1", "1", "h1", 8, 0.9, (), "original")
    if select_retrieval_best((base, worse)).source_id != "original":
        raise ValueError("DEV RA regression protection changed")

    rules_path, rules_sha, filtered_sha, _ = _load_rules(config)
    rules = read_json(rules_path).get("filtered_rules")
    if rules_sha != EXPECTED_RULE_FILE_SHA256:
        raise ValueError("DEV AutoGEO rules file changed")
    if filtered_sha != EXPECTED_FILTERED_RULES_SHA256:
        raise ValueError("DEV filtered_rules hash changed")
    if not isinstance(rules, list) or canonical_json_sha256(rules) != filtered_sha:
        raise ValueError("DEV filtered_rules payload changed")
    contract = {
        "schema_version": "e7_test50_inherited_dev_retrieval_contract_v1",
        "dev_protocol_file_sha256": {
            str(path.relative_to(config.project_root)): expected
            for path, expected in frozen_files.items()
        },
        "prompt_probe_sha256": prompt_hashes,
        "filtered_rules_sha256": filtered_sha,
        "rules_file_sha256": rules_sha,
        "rewrite_model": AUTOGEO_REWRITE_MODEL,
        "rewrite_temperature": REWRITE_TEMPERATURE,
        "candidate_count": 3,
        "max_rounds": 2,
        "ra_selection_policy": [
            "minimum_rank",
            "maximum_score",
            "candidate_id_ascending",
        ],
        "current_best_in_selection_set": True,
        "strict_top5_early_stop": True,
        "retriever_config_sha256": r1_hash,
    }
    return {**contract, "contract_sha256": canonical_json_sha256(contract)}


def _load_test_inputs(config: E7Config) -> TestInputs:
    contract = validate_dev_protocol_contract(config)
    target_path = config.output_paths["targets"] / "test50_targets_r1_pool_v2.frozen.json"
    corpus_path = test_pool_v2_paths(config)["manifest"]
    target_sha = sha256_file(target_path)
    corpus_sha = sha256_file(corpus_path)
    target_manifest = read_json(target_path)
    if target_manifest.get("schema_version") != "e7_test50_targets_r1_pool_v2_frozen_v1":
        raise ValueError("unsupported frozen TEST50 target manifest")
    if target_manifest.get("split") != "test" or target_manifest.get("status") != "frozen":
        raise ValueError("TEST50 target manifest is not frozen test data")
    if target_manifest.get("retriever_config_sha256") != FROZEN_R1_CONFIG_SHA256:
        raise ValueError("TEST50 target manifest R1 config changed")
    if target_manifest.get("corpus_manifest_sha256") != corpus_sha:
        raise ValueError("TEST50 target/corpus manifest hashes disagree")
    records = target_manifest.get("records")
    if not isinstance(records, list) or len(records) != EXPECTED_TEST_QUERY_COUNT:
        raise ValueError("frozen TEST50 target manifest must retain 50 queries")
    if len({str(record.get("query_id")) for record in records}) != len(records):
        raise ValueError("TEST50 query IDs are not unique")
    corpus = load_local_corpus(corpus_path)
    if corpus.corpus_split != "test":
        raise ValueError("TEST50 retrieval cannot consume a DEV corpus")
    eligible: list[dict[str, Any]] = []
    for record in records:
        if record.get("eligible") is True:
            document_id = record.get("target_document_id")
            rank = record.get("initial_r1_rank")
            if not isinstance(document_id, str) or not isinstance(rank, int):
                raise ValueError("eligible TEST50 row lacks frozen target/rank")
            if not 6 <= rank <= 100:
                raise ValueError("eligible TEST50 target left rank range 6..100")
            document = corpus.documents[corpus.index_of(document_id)]
            if document.text_hash != record.get("target_text_hash"):
                raise ValueError("frozen TEST50 target hash differs from corpus")
            eligible.append(dict(record))
        elif record.get("eligible") is False:
            null_fields = (
                record.get("target_document_id"),
                record.get("target_text_hash"),
                record.get("initial_r1_rank"),
                record.get("initial_r1_score"),
            )
            if null_fields != (None, None, None, None):
                raise ValueError("ineligible TEST50 row contains a selected target")
        else:
            raise ValueError("TEST50 row eligibility must be boolean")
    rules_path, rules_sha, filtered_sha, _ = _load_rules(config)
    rules = read_json(rules_path).get("filtered_rules")
    if rules_sha != EXPECTED_RULE_FILE_SHA256 or filtered_sha != EXPECTED_FILTERED_RULES_SHA256:
        raise ValueError("TEST50 runner rules differ from frozen DEV rules")
    return TestInputs(
        records=tuple(dict(record) for record in records),
        eligible_records=tuple(eligible),
        corpus=corpus,
        target_manifest_path=target_path,
        corpus_manifest_path=corpus_path,
        target_manifest_sha256=target_sha,
        corpus_manifest_sha256=corpus_sha,
        rules_path=rules_path,
        filtered_rules=tuple(rules),
        protocol_contract=contract,
    )


def _target_text(inputs: TestInputs, record: Mapping[str, Any]) -> str:
    document = inputs.corpus.documents[
        inputs.corpus.index_of(record["target_document_id"])
    ]
    if document.text_hash != record["target_text_hash"]:
        raise AssertionError("TEST50 target identity/hash changed")
    return document.text


def _rewrite_payload(
    *,
    inputs: TestInputs,
    record: Mapping[str, Any],
    method_id: str,
    candidate_id: str,
    rewritten: str,
    prompt_hash: str | None,
) -> dict[str, Any]:
    return {
        "schema_version": "e7_test50_retrieval_rewrite_v1",
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
        "generation_config": {
            "temperature": REWRITE_TEMPERATURE,
            "top_p": None,
            "seed": None,
            "max_tokens": None,
        },
        "rules_file_sha256": EXPECTED_RULE_FILE_SHA256,
        "filtered_rules_sha256": EXPECTED_FILTERED_RULES_SHA256,
        "target_manifest_sha256": inputs.target_manifest_sha256,
        "corpus_manifest_sha256": inputs.corpus_manifest_sha256,
        "protocol_contract_sha256": inputs.protocol_contract["contract_sha256"],
        "r1_rank_provided": False,
        "r1_score_provided": False,
        "retrieval_feedback_provided": False,
        "iterative_refinement_used": False,
        "r1_evaluation_performed_before_artifact_freeze": False,
        "source": "test50_fresh_generation",
    }


def _validate_rewrite_artifact(
    payload: Mapping[str, Any],
    *,
    inputs: TestInputs,
    record: Mapping[str, Any],
    method_id: str,
    candidate_id: str,
    prompt_hash: str | None,
) -> None:
    expected = {
        "schema_version": "e7_test50_retrieval_rewrite_v1",
        "method_id": method_id,
        "candidate_id": candidate_id,
        "query_id": record["query_id"],
        "query": record["query"],
        "target_document_id": record["target_document_id"],
        "original_target_hash": record["target_text_hash"],
        "prompt_sha256": prompt_hash,
        "rewrite_model": AUTOGEO_REWRITE_MODEL,
        "rules_file_sha256": EXPECTED_RULE_FILE_SHA256,
        "filtered_rules_sha256": EXPECTED_FILTERED_RULES_SHA256,
        "target_manifest_sha256": inputs.target_manifest_sha256,
        "corpus_manifest_sha256": inputs.corpus_manifest_sha256,
        "protocol_contract_sha256": inputs.protocol_contract["contract_sha256"],
    }
    if any(payload.get(key) != value for key, value in expected.items()):
        raise ValueError("reused TEST50 rewrite identity/prompt/config changed")
    rewritten = payload.get("rewritten_text")
    if not isinstance(rewritten, str) or not rewritten.strip():
        raise ValueError("reused TEST50 rewrite is empty")
    if payload.get("rewrite_hash") != normalized_text_sha256(rewritten):
        raise ValueError("reused TEST50 normalized rewrite hash changed")
    if payload.get("rewrite_raw_text_sha256") != sha256_text(rewritten):
        raise ValueError("reused TEST50 raw rewrite hash changed")
    expected_generation = {
        "temperature": REWRITE_TEMPERATURE,
        "top_p": None,
        "seed": None,
        "max_tokens": None,
    }
    if payload.get("generation_config") != expected_generation:
        raise ValueError("reused TEST50 generation config changed")
    forbidden_true = (
        "r1_rank_provided",
        "r1_score_provided",
        "retrieval_feedback_provided",
        "iterative_refinement_used",
        "r1_evaluation_performed_before_artifact_freeze",
    )
    if any(payload.get(field) is not False for field in forbidden_true):
        raise ValueError("baseline rewrite received forbidden retrieval information")


def _create_rewrite(
    config: E7Config,
    inputs: TestInputs,
    record: Mapping[str, Any],
    method_id: str,
    candidate_id: str,
    prompt: str | None,
    *,
    generator: Callable[..., str] = call_gemini,
    stock_rewriter: Callable[..., str] = rewrite_selected_target_with_stock_autogeo,
) -> dict[str, Any]:
    query_id = str(record["query_id"])
    path = (
        _nf_candidate_path(config, query_id, candidate_id)
        if method_id == "no_feedback_multi_sample"
        else _baseline_rewrite_path(config, method_id, query_id)
    )
    prompt_hash = sha256_text(prompt) if prompt is not None else None
    if path.exists():
        payload = read_json(path)
        _validate_rewrite_artifact(
            payload,
            inputs=inputs,
            record=record,
            method_id=method_id,
            candidate_id=candidate_id,
            prompt_hash=prompt_hash,
        )
        return dict(payload)
    if method_id == "autogeo_api_e2e":
        rewritten = stock_rewriter(
            _target_text(inputs, record),
            rules_path=inputs.rules_path,
        )
    else:
        if prompt is None:
            raise ValueError("query-aware rewrite requires frozen prompt")
        rewritten = generator(
            prompt,
            model_name=AUTOGEO_REWRITE_MODEL,
            temperature=REWRITE_TEMPERATURE,
        )
    if not isinstance(rewritten, str) or not rewritten.strip():
        raise ValueError("TEST50 rewrite backend returned empty text")
    payload = _rewrite_payload(
        inputs=inputs,
        record=record,
        method_id=method_id,
        candidate_id=candidate_id,
        rewritten=rewritten,
        prompt_hash=prompt_hash,
    )
    _persist(path, payload, config)
    return payload


def _freeze_c1(
    config: E7Config,
    inputs: TestInputs,
    record: Mapping[str, Any],
    prompt_hash: str,
) -> dict[str, Any]:
    query_id = str(record["query_id"])
    hashes: dict[str, str] = {}
    for candidate_id in NO_FEEDBACK_CANDIDATE_IDS:
        path = _nf_candidate_path(config, query_id, candidate_id)
        payload = read_json(path)
        _validate_rewrite_artifact(
            payload,
            inputs=inputs,
            record=record,
            method_id="no_feedback_multi_sample",
            candidate_id=candidate_id,
            prompt_hash=prompt_hash,
        )
        hashes[candidate_id] = sha256_file(path)
    payload = {
        "schema_version": "e7_test50_no_feedback_selection_pre_retrieval_v1",
        "query_id": query_id,
        "candidate_ids": list(NO_FEEDBACK_CANDIDATE_IDS),
        "selected_candidate_id": "C1",
        "selection_policy": "precommitted_first_sample",
        "candidate_artifact_sha256": hashes,
        "shared_prompt_sha256": prompt_hash,
        "selected_before_r1_evaluation": True,
        "r1_rank_or_score_available_at_selection": False,
        "diagnostic_results_used_for_selection": False,
    }
    _persist(_nf_selection_path(config, query_id), payload, config)
    return payload


def _validate_generation_manifest(
    config: E7Config,
    inputs: TestInputs,
    manifest: Mapping[str, Any],
) -> None:
    if manifest.get("schema_version") != "e7_test50_baseline_generation_manifest_v1":
        raise ValueError("unsupported TEST50 baseline generation manifest")
    if manifest.get("all_c1_selections_frozen_before_any_r1_evaluation") is not True:
        raise ValueError("TEST50 C1 selections were not precommitted")
    if manifest.get("r1_loaded_or_evaluated") is not False:
        raise ValueError("R1 was used during TEST50 baseline generation")
    artifacts = manifest.get("artifacts")
    if not isinstance(artifacts, dict) or set(artifacts) != {
        str(record["query_id"]) for record in inputs.eligible_records
    }:
        raise ValueError("TEST50 generation manifest query set changed")
    for record in inputs.eligible_records:
        query_id = str(record["query_id"])
        item = artifacts[query_id]
        paths = {
            "autogeo_sha256": _baseline_rewrite_path(config, "autogeo_api_e2e", query_id),
            "query_aware_sha256": _baseline_rewrite_path(config, "query_aware_autogeo", query_id),
            "selection_sha256": _nf_selection_path(config, query_id),
        }
        for key, path in paths.items():
            if item.get(key) != sha256_file(path):
                raise ValueError("TEST50 generation artifact changed after manifest freeze")
        selection = read_json(paths["selection_sha256"])
        if selection.get("selected_candidate_id") != "C1":
            raise ValueError("TEST50 formal no-feedback selection changed")
        candidate_hashes = {
            cid: sha256_file(_nf_candidate_path(config, query_id, cid))
            for cid in NO_FEEDBACK_CANDIDATE_IDS
        }
        if item.get("no_feedback_sha256") != candidate_hashes:
            raise ValueError("TEST50 no-feedback candidate artifact changed")


def run_generate_baselines(
    config: E7Config | None = None,
    *,
    _inputs: TestInputs | None = None,
    _generator: Callable[..., str] = call_gemini,
    _stock_rewriter: Callable[..., str] = rewrite_selected_target_with_stock_autogeo,
) -> dict[str, Any]:
    """Generate/reuse baselines without loading or evaluating R1."""
    config = config or E7Config()
    inputs = _inputs or _load_test_inputs(config)
    destination = _root(config) / "baseline_generation_manifest.json"
    if destination.exists():
        payload = read_json(destination)
        _validate_generation_manifest(config, inputs, payload)
        return {"stage": "generate_baselines", "action": "reused_identical", "result": payload}
    artifacts: dict[str, Any] = {}
    for record in inputs.eligible_records:
        query_id = str(record["query_id"])
        prompt = build_query_aware_no_feedback_prompt(
            query=record["query"],
            original_document=_target_text(inputs, record),
            filtered_rules=inputs.filtered_rules,
        )
        prompt_hash = sha256_text(prompt)
        auto = _create_rewrite(
            config,
            inputs,
            record,
            "autogeo_api_e2e",
            "AUTO1",
            None,
            generator=_generator,
            stock_rewriter=_stock_rewriter,
        )
        qa = _create_rewrite(
            config,
            inputs,
            record,
            "query_aware_autogeo",
            "QA1",
            prompt,
            generator=_generator,
            stock_rewriter=_stock_rewriter,
        )
        candidates = [
            _create_rewrite(
                config,
                inputs,
                record,
                "no_feedback_multi_sample",
                candidate_id,
                prompt,
                generator=_generator,
                stock_rewriter=_stock_rewriter,
            )
            for candidate_id in NO_FEEDBACK_CANDIDATE_IDS
        ]
        selection = _freeze_c1(config, inputs, record, prompt_hash)
        artifacts[query_id] = {
            "autogeo_sha256": sha256_file(
                _baseline_rewrite_path(config, "autogeo_api_e2e", query_id)
            ),
            "query_aware_sha256": sha256_file(
                _baseline_rewrite_path(config, "query_aware_autogeo", query_id)
            ),
            "no_feedback_sha256": selection["candidate_artifact_sha256"],
            "selection_sha256": sha256_file(_nf_selection_path(config, query_id)),
            "shared_query_aware_prompt_sha256": prompt_hash,
            "same_prompt_for_qa_and_c1_c2_c3": (
                qa["prompt_sha256"] == prompt_hash
                and all(candidate["prompt_sha256"] == prompt_hash for candidate in candidates)
            ),
            "autogeo_rewrite_hash": auto["rewrite_hash"],
        }
    payload = {
        "schema_version": "e7_test50_baseline_generation_manifest_v1",
        "dataset_name": "TEST50",
        "total_queries": len(inputs.records),
        "eligible_queries": len(inputs.eligible_records),
        "ineligible_queries": len(inputs.records) - len(inputs.eligible_records),
        "artifacts": artifacts,
        "protocol_contract_sha256": inputs.protocol_contract["contract_sha256"],
        "all_c1_selections_frozen_before_any_r1_evaluation": True,
        "r1_loaded_or_evaluated": False,
        "ge_geo_geu_executed": False,
        "test50_processed": True,
    }
    action, digest = _persist(destination, payload, config)
    return {"stage": "generate_baselines", "action": action, "output_sha256": digest, "result": payload}


def _load_test_retriever(inputs: TestInputs, config: E7Config) -> FrozenLocalRetriever:
    encoder = BGEBaseEnV15Encoder(config, execution_device="cpu")
    if canonical_json_sha256(encoder.retriever_specification.to_dict()) != FROZEN_R1_CONFIG_SHA256:
        raise ValueError("loaded TEST query encoder differs from frozen DEV R1")
    cache = load_embedding_cache(
        inputs.corpus,
        encoder.specification,
        config,
        cache_namespace=TEST_POOL_V2_CACHE_NAMESPACE,
        retriever_specification=encoder.retriever_specification,
    )
    return FrozenLocalRetriever(encoder, cache)


def _original_result(
    record: Mapping[str, Any],
    retriever: Any,
    inputs: TestInputs,
    config: E7Config,
) -> dict[str, Any]:
    ranking = retriever.retrieve(
        record["query"], inputs.corpus, top_k=len(inputs.corpus.documents)
    )
    hit = next(
        item for item in ranking if item.document_id == record["target_document_id"]
    )
    if hit.rank != record["initial_r1_rank"]:
        raise AssertionError("Original-E2E final rank differs from frozen initial rank")
    if abs(hit.score - record["initial_r1_score"]) > 1e-7:
        raise AssertionError("Original-E2E score differs from frozen initial score")
    if hit.rank <= config.ge_top_k:
        raise AssertionError("eligible promotion target unexpectedly starts in Top-5")
    return {
        "initial_rank": hit.rank,
        "final_rank": hit.rank,
        "initial_score": hit.score,
        "final_score": hit.score,
        "rank_gain": 0,
        "entered_top5": False,
        "top5_document_ids": [item.document_id for item in ranking[: config.ge_top_k]],
        "rewrite_called": False,
    }


def _evaluate_rewrite(
    record: Mapping[str, Any],
    artifact: Mapping[str, Any],
    retriever: Any,
    inputs: TestInputs,
    config: E7Config,
) -> dict[str, Any]:
    if artifact.get("target_document_id") != record["target_document_id"]:
        raise AssertionError("method changed frozen TEST target identity")
    if artifact.get("original_target_hash") != record["target_text_hash"]:
        raise AssertionError("method changed frozen TEST target hash")
    result = retriever.rerank_with_rewritten_target(
        record["query"],
        inputs.corpus,
        record["target_document_id"],
        artifact["rewritten_text"],
        top_k=config.ge_top_k,
    )
    if result.original_target_rank != record["initial_r1_rank"]:
        raise AssertionError("counterfactual rerank changed original target rank")
    if result.reencoded_document_ids != (record["target_document_id"],):
        raise AssertionError("counterfactual rerank encoded more than target")
    if not result.competitor_invariant_verified:
        raise AssertionError("N-1 competitor invariant failed")
    return {
        "initial_rank": result.original_target_rank,
        "final_rank": result.new_target_rank,
        "initial_score": result.original_target_score,
        "final_score": result.new_target_score,
        "rank_gain": result.original_target_rank - result.new_target_rank,
        "entered_top5": result.new_target_rank <= config.ge_top_k,
        "top5_document_ids": [item.document_id for item in result.new_top_k],
        "rewrite_hash": artifact["rewrite_hash"],
        "only_target_embedding_recomputed": True,
        "competitor_invariant_verified": True,
    }


def formal_no_feedback_result(
    selection: Mapping[str, Any],
    candidate_artifacts: Mapping[str, Mapping[str, Any]],
    evaluator: Callable[[Mapping[str, Any]], dict[str, Any]],
) -> dict[str, Any]:
    """Evaluate all diagnostics while making C1 the immutable formal output."""
    if selection.get("selected_candidate_id") != "C1":
        raise ValueError("formal no-feedback output must remain C1")
    if selection.get("selected_before_r1_evaluation") is not True:
        raise ValueError("C1 was not frozen before R1 evaluation")
    if selection.get("r1_rank_or_score_available_at_selection") is not False:
        raise ValueError("C1 selection observed forbidden R1 information")
    if tuple(candidate_artifacts) != tuple(NO_FEEDBACK_CANDIDATE_IDS):
        raise ValueError("no-feedback candidates must be ordered C1/C2/C3")
    diagnostics = []
    for candidate_id in NO_FEEDBACK_CANDIDATE_IDS:
        result = evaluator(candidate_artifacts[candidate_id])
        diagnostics.append({**result, "candidate_id": candidate_id})
    selected = next(item for item in diagnostics if item["candidate_id"] == "C1")
    return {
        **selected,
        "formal_selected_candidate": "C1",
        "candidate_diagnostics": diagnostics,
        "diagnostics_used_for_selection": False,
    }


def _validate_selection_artifact(selection: Mapping[str, Any]) -> None:
    if selection.get("schema_version") != "e7_test50_no_feedback_selection_pre_retrieval_v1":
        raise ValueError("unsupported TEST50 C1 selection artifact")
    if selection.get("candidate_ids") != ["C1", "C2", "C3"]:
        raise ValueError("TEST50 no-feedback candidate set changed")
    if selection.get("selected_candidate_id") != "C1":
        raise ValueError("TEST50 formal no-feedback output changed")
    if selection.get("selected_before_r1_evaluation") is not True:
        raise ValueError("TEST50 C1 was not frozen before R1")
    if selection.get("r1_rank_or_score_available_at_selection") is not False:
        raise ValueError("TEST50 C1 selection used R1 information")
    if selection.get("diagnostic_results_used_for_selection") is not False:
        raise ValueError("TEST50 C2/C3 diagnostics influenced formal selection")


def run_evaluate_baselines(
    config: E7Config | None = None,
    *,
    _inputs: TestInputs | None = None,
    _retriever: Any | None = None,
) -> dict[str, Any]:
    config = config or E7Config()
    inputs = _inputs or _load_test_inputs(config)
    generation_path = _root(config) / "baseline_generation_manifest.json"
    if not generation_path.is_file():
        raise FileNotFoundError("complete baseline generation before loading R1")
    generation = read_json(generation_path)
    _validate_generation_manifest(config, inputs, generation)
    destination = _root(config) / "baseline_evaluation_manifest.json"
    if destination.exists():
        return {"stage": "evaluate_baselines", "action": "reused_identical", "result": read_json(destination)}
    embedding_path, metadata_path = _test_cache_paths(config)
    cache_before = (sha256_file(embedding_path), sha256_file(metadata_path))
    retriever = _retriever or _load_test_retriever(inputs, config)
    hashes: dict[str, str] = {}
    for record in inputs.records:
        query_id = str(record["query_id"])
        output = _baseline_result_path(config, query_id)
        if record["eligible"] is not True:
            payload = {
                "schema_version": "e7_test50_baseline_retrieval_result_v1",
                "query_id": query_id,
                "query": record["query"],
                "eligible": False,
                "status": "ineligible",
                "methods": None,
                "reason": record["selection_reason"],
            }
            _persist(output, payload, config)
            hashes[query_id] = sha256_file(output)
            continue
        original = _original_result(record, retriever, inputs, config)
        auto = read_json(_baseline_rewrite_path(config, "autogeo_api_e2e", query_id))
        qa = read_json(_baseline_rewrite_path(config, "query_aware_autogeo", query_id))
        selection_path = _nf_selection_path(config, query_id)
        selection = read_json(selection_path)
        _validate_selection_artifact(selection)
        candidates = {
            candidate_id: read_json(_nf_candidate_path(config, query_id, candidate_id))
            for candidate_id in NO_FEEDBACK_CANDIDATE_IDS
        }
        no_feedback = formal_no_feedback_result(
            selection,
            candidates,
            lambda artifact: _evaluate_rewrite(record, artifact, retriever, inputs, config),
        )
        no_feedback["selection_artifact_sha256"] = sha256_file(selection_path)
        payload = {
            "schema_version": "e7_test50_baseline_retrieval_result_v1",
            "query_id": query_id,
            "query": record["query"],
            "eligible": True,
            "status": "evaluated",
            "target_document_id": record["target_document_id"],
            "target_text_hash": record["target_text_hash"],
            "methods": {
                "original_e2e": original,
                "autogeo_api_e2e": _evaluate_rewrite(record, auto, retriever, inputs, config),
                "query_aware_autogeo": _evaluate_rewrite(record, qa, retriever, inputs, config),
                "no_feedback_multi_sample": no_feedback,
            },
            "ge_geo_geu_executed": False,
        }
        _persist(output, payload, config)
        hashes[query_id] = sha256_file(output)
    if (sha256_file(embedding_path), sha256_file(metadata_path)) != cache_before:
        raise AssertionError("baseline evaluation modified frozen TEST cache")
    payload = {
        "schema_version": "e7_test50_baseline_evaluation_manifest_v1",
        "dataset_name": "TEST50",
        "result_sha256": hashes,
        "eligible_queries": len(inputs.eligible_records),
        "ineligible_queries": len(inputs.records) - len(inputs.eligible_records),
        "formal_no_feedback_output": "C1",
        "ge_geo_geu_executed": False,
        "test50_processed": True,
    }
    action, digest = _persist(destination, payload, config)
    return {"stage": "evaluate_baselines", "action": action, "output_sha256": digest, "result": payload}


def execute_ra_policy(
    original: RankedOption,
    candidate_provider: Callable[[int, RankedOption], Sequence[RankedOption]],
    *,
    ge_top_k: int = 5,
    max_rounds: int = 2,
) -> RAPolicyResult:
    """Apply the frozen DEV RA policy with current-best regression protection."""
    parent = original
    rounds: list[dict[str, Any]] = []
    all_candidates: list[RankedOption] = []
    early_stop = False
    for round_index in range(1, max_rounds + 1):
        ranked = tuple(candidate_provider(round_index, parent))
        expected_ids = tuple(ROUND_CANDIDATES[round_index])
        if tuple(option.source_id for option in ranked) != expected_ids:
            raise ValueError("RA round candidate IDs/order changed")
        if any(option.parent_candidate_id != parent.source_id for option in ranked):
            raise ValueError("RA candidate parent provenance changed")
        selected = select_retrieval_best((parent, *ranked))
        if selected.rank > parent.rank or (
            selected.rank == parent.rank and selected.score < parent.score
        ):
            raise AssertionError("RA accepted a retrieval regression")
        rounds.append(
            {
                "round": round_index,
                "parent_source": parent.source_id,
                "selection_pool": [parent.source_id, *expected_ids],
                "selection_policy": [
                    "minimum_rank",
                    "maximum_score",
                    "candidate_id_ascending",
                ],
                "selected_source": selected.source_id,
                "selected_rank": selected.rank,
                "selected_score": selected.score,
                "entered_top5": selected.rank <= ge_top_k,
                "current_best_preserved_in_selection_set": True,
            }
        )
        all_candidates.extend(ranked)
        parent = selected
        if parent.rank <= ge_top_k:
            early_stop = round_index < max_rounds
            break
    return RAPolicyResult(
        selected=parent,
        rounds=tuple(rounds),
        candidates=tuple(all_candidates),
        early_stop=early_stop,
    )


def _ra_raw(
    config: E7Config,
    inputs: TestInputs,
    record: Mapping[str, Any],
    original: RankedOption,
    parent: RankedOption,
    round_index: int,
    candidate_id: str,
    *,
    generator: Callable[..., str] = call_gemini,
) -> dict[str, Any]:
    path = _ra_root(config, str(record["query_id"])) / "raw_candidates" / f"{candidate_id}.json"
    prompt = build_ra_prompt(
        candidate_id=candidate_id,
        query=record["query"],
        original_document=original.text,
        filtered_rules=inputs.filtered_rules,
        current_best=parent,
        initial_rank=original.rank,
        round_index=round_index,
    )
    prompt_hash = sha256_text(prompt)
    if path.exists():
        payload = read_json(path)
        expected = {
            "query_id": record["query_id"],
            "target_document_id": record["target_document_id"],
            "original_target_hash": record["target_text_hash"],
            "round": round_index,
            "candidate_id": candidate_id,
            "parent_candidate_id": parent.source_id,
            "prompt_sha256": prompt_hash,
            "rewrite_model": AUTOGEO_REWRITE_MODEL,
            "filtered_rules_sha256": EXPECTED_FILTERED_RULES_SHA256,
        }
        if any(payload.get(key) != value for key, value in expected.items()):
            raise ValueError("reused RA candidate prompt/identity/config changed")
        rewritten = payload.get("rewritten_text")
        if not isinstance(rewritten, str) or payload.get("rewrite_hash") != normalized_text_sha256(rewritten):
            raise ValueError("reused RA candidate text/hash changed")
        return dict(payload)
    rewritten = generator(
        prompt,
        model_name=AUTOGEO_REWRITE_MODEL,
        temperature=REWRITE_TEMPERATURE,
    )
    if not isinstance(rewritten, str) or not rewritten.strip():
        raise ValueError("RA backend returned empty candidate")
    payload = {
        "schema_version": "e7_test50_ra_raw_candidate_v1",
        "query_id": record["query_id"],
        "target_document_id": record["target_document_id"],
        "original_target_hash": record["target_text_hash"],
        "round": round_index,
        "candidate_id": candidate_id,
        "parent_candidate_id": parent.source_id,
        "rewritten_text": rewritten,
        "rewrite_hash": normalized_text_sha256(rewritten),
        "rewrite_raw_text_sha256": sha256_text(rewritten),
        "prompt_sha256": prompt_hash,
        "rewrite_model": AUTOGEO_REWRITE_MODEL,
        "generation_config": {
            "temperature": REWRITE_TEMPERATURE,
            "top_p": None,
            "seed": None,
            "max_tokens": None,
        },
        "filtered_rules_sha256": EXPECTED_FILTERED_RULES_SHA256,
        "retrieval_feedback": {
            "source": parent.source_id,
            "rank": parent.rank,
            "score": parent.score,
            "rank_improvement_vs_original": original.rank - parent.rank,
            "rank_degradation_vs_original": max(0, parent.rank - original.rank),
        },
        "original_document_is_factual_ground_truth": True,
        "geo_geu_used": False,
    }
    _persist(path, payload, config)
    return payload


def _rank_ra(
    record: Mapping[str, Any],
    raw: Mapping[str, Any],
    parent: RankedOption,
    retriever: Any,
    inputs: TestInputs,
    config: E7Config,
) -> RankedOption:
    result = retriever.rerank_with_rewritten_target(
        record["query"],
        inputs.corpus,
        record["target_document_id"],
        raw["rewritten_text"],
        top_k=config.ge_top_k,
    )
    if result.original_target_rank != record["initial_r1_rank"]:
        raise AssertionError("RA changed frozen original target rank")
    if result.reencoded_document_ids != (record["target_document_id"],):
        raise AssertionError("RA encoded more than the frozen target")
    if not result.competitor_invariant_verified:
        raise AssertionError("RA changed an N-1 competitor")
    return RankedOption(
        source_id=raw["candidate_id"],
        text=raw["rewritten_text"],
        rewrite_hash=raw["rewrite_hash"],
        rank=result.new_target_rank,
        score=result.new_target_score,
        top5_document_ids=tuple(item.document_id for item in result.new_top_k),
        parent_candidate_id=parent.source_id,
    )


def _ra_original(
    record: Mapping[str, Any],
    baseline: Mapping[str, Any],
    inputs: TestInputs,
) -> RankedOption:
    result = baseline["methods"]["original_e2e"]
    return RankedOption(
        source_id="original",
        text=_target_text(inputs, record),
        rewrite_hash=record["target_text_hash"],
        rank=result["final_rank"],
        score=result["final_score"],
        top5_document_ids=tuple(result["top5_document_ids"]),
        parent_candidate_id=None,
    )


def _run_ra_query(
    config: E7Config,
    inputs: TestInputs,
    record: Mapping[str, Any],
    baseline: Mapping[str, Any],
    retriever: Any,
    *,
    generator: Callable[..., str] = call_gemini,
) -> dict[str, Any]:
    query_id = str(record["query_id"])
    output = _ra_result_path(config, query_id)
    if output.exists():
        return read_json(output)
    original = _ra_original(record, baseline, inputs)
    raw_by_id: dict[str, dict[str, Any]] = {}

    def provider(round_index: int, parent: RankedOption) -> Sequence[RankedOption]:
        ranked: list[RankedOption] = []
        for candidate_id in ROUND_CANDIDATES[round_index]:
            raw = _ra_raw(
                config,
                inputs,
                record,
                original,
                parent,
                round_index,
                candidate_id,
                generator=generator,
            )
            raw_by_id[candidate_id] = raw
            ranked.append(_rank_ra(record, raw, parent, retriever, inputs, config))
        return ranked

    policy = execute_ra_policy(
        original,
        provider,
        ge_top_k=config.ge_top_k,
        max_rounds=config.max_rounds,
    )
    candidates: list[dict[str, Any]] = []
    for round_payload in policy.rounds:
        round_index = round_payload["round"]
        selected_source = round_payload["selected_source"]
        round_candidates = [
            option for option in policy.candidates
            if option.source_id in ROUND_CANDIDATES[round_index]
        ]
        detail = []
        for option in round_candidates:
            raw_path = _ra_root(config, query_id) / "raw_candidates" / f"{option.source_id}.json"
            item = {
                "round": round_index,
                "candidate_id": option.source_id,
                "parent_candidate_id": option.parent_candidate_id,
                "raw_artifact_sha256": sha256_file(raw_path),
                "rewrite_hash": option.rewrite_hash,
                "rank": option.rank,
                "score": option.score,
                "rank_gain_vs_original": original.rank - option.rank,
                "selected": option.source_id == selected_source,
                "entered_top5": option.rank <= config.ge_top_k,
            }
            detail.append(item)
            candidates.append(item)
        persisted_round = {**round_payload, "candidates": detail}
        _persist(
            _ra_root(config, query_id) / f"round_{round_index}.json",
            persisted_round,
            config,
        )
    selected = policy.selected
    payload = {
        "schema_version": "e7_test50_ra_retrieval_result_v1",
        "query_id": query_id,
        "query": record["query"],
        "eligible": True,
        "target_document_id": record["target_document_id"],
        "initial_rank": original.rank,
        "final_rank": selected.rank,
        "initial_score": original.score,
        "final_score": selected.score,
        "rank_gain": original.rank - selected.rank,
        "entered_top5": selected.rank <= config.ge_top_k,
        "rounds_used": len(policy.rounds),
        "candidate_count": len(policy.candidates),
        "early_stop": policy.early_stop,
        "selected_source": selected.source_id,
        "stop_reason": "entered_top5" if selected.rank <= config.ge_top_k else "max_rounds_reached",
        "original_factual_document_preserved": True,
        "current_best_preserved": True,
        "selection_policy": ["minimum_rank", "maximum_score", "candidate_id_ascending"],
        "rounds": list(policy.rounds),
        "candidates": candidates,
        "monotonic_non_worsening_verified": (
            selected.rank < original.rank
            or (selected.rank == original.rank and selected.score >= original.score)
        ),
        "geo_geu_used_for_selection": False,
    }
    _persist(output, payload, config)
    return payload


def run_ra(
    config: E7Config | None = None,
    *,
    _inputs: TestInputs | None = None,
    _retriever: Any | None = None,
    _generator: Callable[..., str] = call_gemini,
) -> dict[str, Any]:
    config = config or E7Config()
    inputs = _inputs or _load_test_inputs(config)
    baseline_manifest = _root(config) / "baseline_evaluation_manifest.json"
    if not baseline_manifest.is_file():
        raise FileNotFoundError("baseline retrieval evaluation must finish before RA")
    destination = _root(config) / "ra_evaluation_manifest.json"
    if destination.exists():
        return {"stage": "run_ra", "action": "reused_identical", "result": read_json(destination)}
    embedding_path, metadata_path = _test_cache_paths(config)
    cache_before = (sha256_file(embedding_path), sha256_file(metadata_path))
    retriever = _retriever or _load_test_retriever(inputs, config)
    hashes: dict[str, str] = {}
    for record in inputs.records:
        query_id = str(record["query_id"])
        if record["eligible"] is not True:
            payload = {
                "schema_version": "e7_test50_ra_retrieval_result_v1",
                "query_id": query_id,
                "query": record["query"],
                "eligible": False,
                "status": "ineligible",
                "reason": record["selection_reason"],
            }
            _persist(_ra_result_path(config, query_id), payload, config)
        else:
            baseline = read_json(_baseline_result_path(config, query_id))
            _run_ra_query(
                config,
                inputs,
                record,
                baseline,
                retriever,
                generator=_generator,
            )
        hashes[query_id] = sha256_file(_ra_result_path(config, query_id))
    if (sha256_file(embedding_path), sha256_file(metadata_path)) != cache_before:
        raise AssertionError("RA modified frozen TEST cache")
    payload = {
        "schema_version": "e7_test50_ra_evaluation_manifest_v1",
        "dataset_name": "TEST50",
        "result_sha256": hashes,
        "eligible_queries": len(inputs.eligible_records),
        "ineligible_queries": len(inputs.records) - len(inputs.eligible_records),
        "num_candidates_per_round": 3,
        "max_rounds": 2,
        "strict_top5_early_stop": True,
        "ge_geo_geu_used_for_selection": False,
        "test50_processed": True,
    }
    action, digest = _persist(destination, payload, config)
    return {"stage": "run_ra", "action": action, "output_sha256": digest, "result": payload}


def _method_metrics(records: Sequence[Mapping[str, Any]], method: str) -> dict[str, Any]:
    results = [record["methods"][method] for record in records]
    if not results:
        return {
            "n": 0,
            "hit_at_5_count": 0,
            "hit_at_5": None,
            "mean_final_rank": None,
            "median_final_rank": None,
            "mean_rank_gain": None,
            "median_rank_gain": None,
            "target_mrr": None,
        }
    ranks = [int(result["final_rank"]) for result in results]
    gains = [int(result["rank_gain"]) for result in results]
    hits = sum(bool(result["entered_top5"]) for result in results)
    return {
        "n": len(results),
        "hit_at_5_count": hits,
        "hit_at_5": hits / len(results),
        "mean_final_rank": statistics.fmean(ranks),
        "median_final_rank": statistics.median(ranks),
        "mean_rank_gain": statistics.fmean(gains),
        "median_rank_gain": statistics.median(gains),
        "target_mrr": statistics.fmean(1.0 / rank for rank in ranks),
    }


def paired_comparison(
    records: Sequence[Mapping[str, Any]],
    baseline_method: str,
    treatment_method: str,
) -> dict[str, Any]:
    pairs = [
        (record["methods"][baseline_method], record["methods"][treatment_method])
        for record in records
    ]
    deltas = [int(left["final_rank"]) - int(right["final_rank"]) for left, right in pairs]
    wins = sum(delta > 0 for delta in deltas)
    losses = sum(delta < 0 for delta in deltas)
    return {
        "baseline": baseline_method,
        "treatment": treatment_method,
        "positive_rank_delta_favors_treatment": True,
        "treatment_rank_wins": wins,
        "rank_ties": len(deltas) - wins - losses,
        "treatment_rank_losses": losses,
        "mean_paired_final_rank_improvement": statistics.fmean(deltas) if deltas else None,
        "median_paired_final_rank_improvement": statistics.median(deltas) if deltas else None,
        "treatment_hit5_gains": sum(
            not left["entered_top5"] and right["entered_top5"] for left, right in pairs
        ),
        "treatment_hit5_losses": sum(
            left["entered_top5"] and not right["entered_top5"] for left, right in pairs
        ),
        "per_query_rank_delta": {
            record["query_id"]: delta for record, delta in zip(records, deltas)
        },
    }


def summarize_records(
    combined: Sequence[Mapping[str, Any]],
    *,
    retriever_config_sha256: str,
    corpus_manifest_sha256: str,
    target_manifest_sha256: str,
) -> dict[str, Any]:
    if len(combined) != EXPECTED_TEST_QUERY_COUNT:
        raise ValueError("TEST50 summary must preserve all 50 queries")
    if len({str(record.get("query_id")) for record in combined}) != len(combined):
        raise ValueError("TEST50 summary query IDs are not unique")
    eligible = [record for record in combined if record.get("eligible") is True]
    ineligible = [record for record in combined if record.get("eligible") is False]
    if any(record.get("methods") is not None for record in ineligible):
        raise ValueError("ineligible TEST50 rows cannot have promotion metrics")
    method_statistics = {
        method: _method_metrics(eligible, method) for method in METHOD_IDS
    }
    comparisons = {
        "A_autogeo_vs_query_aware": paired_comparison(
            eligible, "autogeo_api_e2e", "query_aware_autogeo"
        ),
        "B_query_aware_vs_ra": paired_comparison(
            eligible, "query_aware_autogeo", "ra_autogeo_e2e"
        ),
        "C_no_feedback_c1_vs_ra": paired_comparison(
            eligible, "no_feedback_multi_sample", "ra_autogeo_e2e"
        ),
    }
    return {
        "schema_version": "e7_test50_retrieval_summary_v1",
        "dataset_name": "TEST50",
        "total_queries": len(combined),
        "eligible_queries": len(eligible),
        "ineligible_queries": len(ineligible),
        "eligibility_rate": len(eligible) / len(combined),
        "ineligible_counted_as_retrieval_failure": False,
        "method_statistics": method_statistics,
        "paired_comparisons": comparisons,
        "formal_no_feedback_output": "C1",
        "no_feedback_c2_c3_role": "posthoc_diagnostic_only",
        "retriever_id": "r1_independent_dense",
        "retriever_config_sha256": retriever_config_sha256,
        "corpus_manifest_sha256": corpus_manifest_sha256,
        "target_manifest_sha256": target_manifest_sha256,
        "filtered_rules_sha256": EXPECTED_FILTERED_RULES_SHA256,
        "ge_geo_geu_executed": False,
        "test50_processed": True,
    }


def _report(summary: Mapping[str, Any]) -> str:
    eligible = int(summary["eligible_queries"])
    lines = [
        "# E7 TEST50 Retrieval-Side Evaluation",
        "",
        f"- Total queries: {summary['total_queries']}",
        f"- Eligible frozen targets: {eligible}",
        f"- Ineligible queries retained: {summary['ineligible_queries']}",
        f"- Eligibility rate: {summary['eligibility_rate']:.6f}",
        "- Ineligible queries are not counted as promotion failures.",
        "- GE/GEO/GEU executed: no",
        "",
        "## Eligible-target retrieval metrics",
        "",
        "| Method | Hit@5 | Mean rank | Median rank | Mean gain | Median gain | Target MRR |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    labels = {
        "original_e2e": "Original-E2E",
        "autogeo_api_e2e": "AutoGEO_API-E2E",
        "query_aware_autogeo": "Query-Aware AutoGEO",
        "no_feedback_multi_sample": "No-Feedback C1",
        "ra_autogeo_e2e": "RA-AutoGEO",
    }
    for method in METHOD_IDS:
        item = summary["method_statistics"][method]
        if eligible == 0:
            lines.append(f"| {labels[method]} | 0/0 | null | null | null | null | null |")
        else:
            lines.append(
                f"| {labels[method]} | {item['hit_at_5_count']}/{eligible} ({item['hit_at_5']:.3f}) | "
                f"{item['mean_final_rank']:.3f} | {item['median_final_rank']:.3f} | "
                f"{item['mean_rank_gain']:.3f} | {item['median_rank_gain']:.3f} | "
                f"{item['target_mrr']:.6f} |"
            )
    lines.extend(["", "## Frozen paired comparisons", ""])
    for key, item in summary["paired_comparisons"].items():
        lines.append(
            f"- {key}: treatment rank wins/ties/losses = "
            f"{item['treatment_rank_wins']}/{item['rank_ties']}/{item['treatment_rank_losses']}; "
            f"Hit@5 gains/losses = {item['treatment_hit5_gains']}/{item['treatment_hit5_losses']}."
        )
    lines.extend([
        "",
        "C2/C3 are post-hoc diagnostics only; formal No-Feedback is precommitted C1.",
        "",
    ])
    return "\n".join(lines)


def run_summary(
    config: E7Config | None = None,
    *,
    _inputs: TestInputs | None = None,
) -> dict[str, Any]:
    config = config or E7Config()
    inputs = _inputs or _load_test_inputs(config)
    if not (_root(config) / "ra_evaluation_manifest.json").is_file():
        raise FileNotFoundError("RA evaluation must finish before TEST50 summary")
    combined: list[dict[str, Any]] = []
    for record in inputs.records:
        query_id = str(record["query_id"])
        baseline = read_json(_baseline_result_path(config, query_id))
        ra = read_json(_ra_result_path(config, query_id))
        if record["eligible"] is not True:
            item = {
                "query_id": query_id,
                "query": record["query"],
                "eligible": False,
                "status": "ineligible",
                "target_document_id": None,
                "methods": None,
                "selection_reason": record["selection_reason"],
            }
        else:
            methods = dict(baseline["methods"])
            methods["ra_autogeo_e2e"] = {
                key: ra[key]
                for key in (
                    "initial_rank",
                    "final_rank",
                    "initial_score",
                    "final_score",
                    "rank_gain",
                    "entered_top5",
                    "rounds_used",
                    "candidate_count",
                    "early_stop",
                    "selected_source",
                )
            }
            item = {
                "query_id": query_id,
                "query": record["query"],
                "eligible": True,
                "status": "evaluated",
                "target_document_id": record["target_document_id"],
                "target_source": record["target_source"],
                "initial_rank": record["initial_r1_rank"],
                "initial_score": record["initial_r1_score"],
                "methods": methods,
            }
        combined.append(item)
    summary = summarize_records(
        combined,
        retriever_config_sha256=FROZEN_R1_CONFIG_SHA256,
        corpus_manifest_sha256=inputs.corpus_manifest_sha256,
        target_manifest_sha256=inputs.target_manifest_sha256,
    )
    jsonl = "".join(
        json.dumps(item, ensure_ascii=False, sort_keys=True) + "\n"
        for item in combined
    )
    jsonl_action, jsonl_hash = _write_text_once(
        _root(config) / "test50_retrieval_per_query.jsonl", jsonl, config
    )
    summary_action, summary_hash = _persist(
        _root(config) / "test50_retrieval_summary.json", summary, config
    )
    report_action, report_hash = _write_text_once(
        _root(config) / "test50_retrieval_report.md", _report(summary), config
    )
    return {
        "stage": "summary",
        "actions": {
            "jsonl": jsonl_action,
            "summary": summary_action,
            "report": report_action,
        },
        "sha256": {
            "jsonl": jsonl_hash,
            "summary": summary_hash,
            "report": report_hash,
        },
        "result": summary,
    }


def run_validate(config: E7Config | None = None) -> dict[str, Any]:
    config = config or E7Config()
    inputs = _load_test_inputs(config)
    return {
        "stage": "validate",
        "dataset_name": "TEST50",
        "total": len(inputs.records),
        "eligible": len(inputs.eligible_records),
        "ineligible": len(inputs.records) - len(inputs.eligible_records),
        "candidate_count": 3,
        "max_rounds": 2,
        "protocol_contract_sha256": inputs.protocol_contract["contract_sha256"],
        "external_calls_made": False,
        "retrieval_run": False,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--stage",
        required=True,
        choices=(
            "validate",
            "generate_baselines",
            "evaluate_baselines",
            "run_ra",
            "summary",
        ),
    )
    args = parser.parse_args()
    runners = {
        "validate": run_validate,
        "generate_baselines": run_generate_baselines,
        "evaluate_baselines": run_evaluate_baselines,
        "run_ra": run_ra,
        "summary": run_summary,
    }
    print(json.dumps(runners[args.stage](), ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
