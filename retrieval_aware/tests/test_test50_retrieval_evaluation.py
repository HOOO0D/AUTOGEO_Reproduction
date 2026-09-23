from __future__ import annotations

import json
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path

from retrieval_aware.config import E7Config
from retrieval_aware.frozen_local_retriever import FrozenLocalCorpus, LocalDocument
from retrieval_aware.ra_e2e_smoke import RankedOption
from retrieval_aware.test50_retrieval_evaluation import (
    EXPECTED_PROMPT_PROBE_SHA256,
    FROZEN_R1_CONFIG_SHA256,
    TestInputs,
    _create_rewrite,
    _freeze_c1,
    _nf_candidate_path,
    _nf_selection_path,
    _persist,
    _ra_raw,
    execute_ra_policy,
    formal_no_feedback_result,
    run_generate_baselines,
    summarize_records,
    validate_prompt_contract,
    validate_protocol_scalars,
)
from retrieval_aware.utils import normalized_text_sha256, read_json


def _config(root: Path) -> E7Config:
    frozen = root / "experiments" / "frozen"
    return replace(
        E7Config(),
        project_root=root,
        data_root=root / "data",
        frozen_root=frozen,
        rule_pool_query_dir=frozen / "rule_pool",
        dev_query_dir=frozen / "dev20",
        test_query_dir=frozen / "test50",
        frozen_query_dir=frozen / "test50",
        output_root=root / "outputs" / "e7_retrieval_aware",
    )


def _inputs(root: Path, *, eligible_count: int = 1) -> TestInputs:
    config = _config(root)
    text = "Original factual document."
    text_hash = normalized_text_sha256(text)
    target_id = f"e7doc_sha256_{text_hash}"
    corpus = FrozenLocalCorpus(
        corpus_id="synthetic-test-pool-v2",
        corpus_split="test",
        manifest_path=root / "synthetic_manifest.json",
        manifest_sha256="corpus-hash",
        documents=(LocalDocument(target_id, text, text_hash),),
    )
    records = []
    for index in range(50):
        eligible = index < eligible_count
        records.append(
            {
                "query_id": f"q{index:02d}",
                "query": f"query {index}",
                "eligible": eligible,
                "status": "selected" if eligible else "ineligible",
                "target_document_id": target_id if eligible else None,
                "target_text_hash": text_hash if eligible else None,
                "target_source": "original_target_id" if eligible else None,
                "initial_r1_rank": 7 if eligible else None,
                "initial_r1_score": 0.5 if eligible else None,
                "selection_reason": "synthetic eligible" if eligible else "synthetic ineligible",
            }
        )
    rules_path = root / "rules.json"
    rules_path.write_text(json.dumps({"filtered_rules": ["rule one"]}), encoding="utf-8")
    return TestInputs(
        records=tuple(records),
        eligible_records=tuple(records[:eligible_count]),
        corpus=corpus,
        target_manifest_path=root / "target.json",
        corpus_manifest_path=root / "corpus.json",
        target_manifest_sha256="target-hash",
        corpus_manifest_sha256="corpus-hash",
        rules_path=rules_path,
        filtered_rules=("rule one",),
        protocol_contract={"contract_sha256": "contract-hash"},
    )


def _option(
    source: str,
    rank: int,
    score: float,
    parent: str | None,
    *,
    text: str | None = None,
) -> RankedOption:
    return RankedOption(
        source_id=source,
        text=text or source,
        rewrite_hash=f"hash-{source}",
        rank=rank,
        score=score,
        top5_document_ids=("a", "b", "c", "d", "e"),
        parent_candidate_id=parent,
    )


def _method(rank: int, initial: int = 7):
    return {
        "initial_rank": initial,
        "final_rank": rank,
        "initial_score": 0.5,
        "final_score": 1.0 / rank,
        "rank_gain": initial - rank,
        "entered_top5": rank <= 5,
    }


class Test50RetrievalEvaluationTests(unittest.TestCase):
    def test_prompt_and_config_hash_mismatch_hard_fail(self) -> None:
        self.assertEqual(validate_prompt_contract(), EXPECTED_PROMPT_PROBE_SHA256)
        changed = dict(EXPECTED_PROMPT_PROBE_SHA256)
        changed["ra_round_1"] = "changed"
        with self.assertRaisesRegex(ValueError, "prompt behavior changed"):
            validate_prompt_contract(changed)
        with self.assertRaisesRegex(ValueError, "R1 config hash"):
            validate_protocol_scalars(
                ge_top_k=5,
                candidate_count=3,
                max_rounds=2,
                r1_config_sha256="changed",
            )
        with self.assertRaisesRegex(ValueError, "candidate/round"):
            validate_protocol_scalars(
                ge_top_k=5,
                candidate_count=4,
                max_rounds=2,
                r1_config_sha256=FROZEN_R1_CONFIG_SHA256,
            )

    def test_generate_freezes_c1_before_any_r1_and_preserves_50_records(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            config = _config(root)
            inputs = _inputs(root)
            generated = []

            def generator(prompt, **kwargs):
                generated.append((prompt, kwargs))
                return f"generated {len(generated)}"

            def stock(document, **kwargs):
                return f"stock: {document}"

            result = run_generate_baselines(
                config,
                _inputs=inputs,
                _generator=generator,
                _stock_rewriter=stock,
            )
            self.assertEqual(result["result"]["total_queries"], 50)
            self.assertEqual(result["result"]["eligible_queries"], 1)
            self.assertFalse(result["result"]["r1_loaded_or_evaluated"])
            selection = read_json(_nf_selection_path(config, "q00"))
            self.assertEqual(selection["selected_candidate_id"], "C1")
            self.assertTrue(selection["selected_before_r1_evaluation"])
            self.assertFalse(selection["r1_rank_or_score_available_at_selection"])
            self.assertEqual(tuple(selection["candidate_ids"]), ("C1", "C2", "C3"))
            self.assertEqual(len(generated), 4)  # QA1 plus C1/C2/C3; stock is separate.

    def test_resume_generates_only_missing_candidate(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            config = _config(root)
            inputs = _inputs(root)
            record = inputs.eligible_records[0]
            prompt = __import__(
                "retrieval_aware.retrieval_ablation_smoke",
                fromlist=["build_query_aware_no_feedback_prompt"],
            ).build_query_aware_no_feedback_prompt(
                query=record["query"],
                original_document="Original factual document.",
                filtered_rules=inputs.filtered_rules,
            )
            calls = []

            def generator(prompt_text, **kwargs):
                calls.append(prompt_text)
                return f"candidate {len(calls)}"

            def stock(document, **kwargs):
                return "stock rewrite"

            _create_rewrite(
                config, inputs, record, "autogeo_api_e2e", "AUTO1", None,
                generator=generator, stock_rewriter=stock,
            )
            _create_rewrite(
                config, inputs, record, "query_aware_autogeo", "QA1", prompt,
                generator=generator, stock_rewriter=stock,
            )
            for candidate_id in ("C1", "C2"):
                _create_rewrite(
                    config, inputs, record, "no_feedback_multi_sample", candidate_id, prompt,
                    generator=generator, stock_rewriter=stock,
                )
            calls.clear()
            run_generate_baselines(
                config,
                _inputs=inputs,
                _generator=generator,
                _stock_rewriter=stock,
            )
            self.assertEqual(len(calls), 1)
            self.assertTrue(_nf_candidate_path(config, "q00", "C3").is_file())
            self.assertTrue(_nf_selection_path(config, "q00").is_file())

    def test_reused_prompt_mismatch_fails_without_resampling(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            config = _config(root)
            inputs = _inputs(root)
            record = inputs.eligible_records[0]
            path = _nf_candidate_path(config, "q00", "C1")
            payload = {
                "schema_version": "e7_test50_retrieval_rewrite_v1",
                "method_id": "no_feedback_multi_sample",
                "candidate_id": "C1",
                "query_id": "q00",
                "query": "query 0",
                "target_document_id": record["target_document_id"],
                "original_target_hash": record["target_text_hash"],
                "rewritten_text": "existing",
                "rewrite_hash": normalized_text_sha256("existing"),
                "rewrite_raw_text_sha256": __import__("hashlib").sha256(b"existing").hexdigest(),
                "prompt_sha256": "wrong",
                "rewrite_model": "gemini-2.5-pro",
                "generation_config": {"temperature": 0.7, "top_p": None, "seed": None, "max_tokens": None},
                "rules_file_sha256": "7aab609ca6be500e16d882c326c8ced3dd446dc9c1b9f4746b83339d3ab9df94",
                "filtered_rules_sha256": "1b1d918ce956c07f3eb142b9465e5d5cf59786135a7a1c287aae3e58a091aff0",
                "target_manifest_sha256": "target-hash",
                "corpus_manifest_sha256": "corpus-hash",
                "protocol_contract_sha256": "contract-hash",
                "r1_rank_provided": False,
                "r1_score_provided": False,
                "retrieval_feedback_provided": False,
                "iterative_refinement_used": False,
                "r1_evaluation_performed_before_artifact_freeze": False,
            }
            _persist(path, payload, config)
            calls = []
            with self.assertRaisesRegex(ValueError, "prompt/config changed"):
                _create_rewrite(
                    config,
                    inputs,
                    record,
                    "no_feedback_multi_sample",
                    "C1",
                    "correct prompt",
                    generator=lambda *args, **kwargs: calls.append(1) or "new",
                )
            self.assertEqual(calls, [])

    def test_c2_c3_cannot_replace_formal_c1(self) -> None:
        selection = {
            "selected_candidate_id": "C1",
            "selected_before_r1_evaluation": True,
            "r1_rank_or_score_available_at_selection": False,
        }
        artifacts = {
            "C1": {"rank": 20},
            "C2": {"rank": 1},
            "C3": {"rank": 2},
        }
        result = formal_no_feedback_result(
            selection,
            artifacts,
            lambda artifact: {
                "final_rank": artifact["rank"],
                "rank_gain": 7 - artifact["rank"],
                "entered_top5": artifact["rank"] <= 5,
            },
        )
        self.assertEqual(result["formal_selected_candidate"], "C1")
        self.assertEqual(result["final_rank"], 20)
        self.assertFalse(result["diagnostics_used_for_selection"])

    def test_ra_early_stop_and_original_preservation(self) -> None:
        original = _option("original", 7, 0.5, None, text="ORIGINAL FACTS")
        rounds_called = []

        def provider(round_index, parent):
            rounds_called.append(round_index)
            return (
                _option("C1", 2, 0.7, parent.source_id),
                _option("C2", 4, 0.6, parent.source_id),
                _option("C3", 6, 0.8, parent.source_id),
            )

        result = execute_ra_policy(original, provider)
        self.assertEqual(rounds_called, [1])
        self.assertTrue(result.early_stop)
        self.assertEqual(result.selected.source_id, "C1")

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            config = _config(root)
            inputs = _inputs(root)
            record = inputs.eligible_records[0]
            captured = []
            parent = _option("C1", 6, 0.6, "original", text="PREVIOUS BEST")
            payload = _ra_raw(
                config,
                inputs,
                record,
                original,
                parent,
                2,
                "C4",
                generator=lambda prompt, **kwargs: captured.append(prompt) or "rewritten",
            )
            self.assertIn("ORIGINAL FACTS", captured[0])
            self.assertIn("PREVIOUS BEST", captured[0])
            self.assertTrue(payload["original_document_is_factual_ground_truth"])
            self.assertEqual(payload["retrieval_feedback"]["rank_improvement_vs_original"], 1)

    def test_ra_second_round_and_regression_protection(self) -> None:
        original = _option("original", 7, 0.5, None)
        rounds_called = []

        def provider(round_index, parent):
            rounds_called.append(round_index)
            if round_index == 1:
                return tuple(
                    _option(candidate_id, rank, score, parent.source_id)
                    for candidate_id, rank, score in (
                        ("C1", 6, 0.6), ("C2", 8, 0.8), ("C3", 9, 0.9)
                    )
                )
            return tuple(
                _option(candidate_id, rank, score, parent.source_id)
                for candidate_id, rank, score in (
                    ("C4", 4, 0.7), ("C5", 6, 0.9), ("C6", 10, 0.9)
                )
            )

        result = execute_ra_policy(original, provider)
        self.assertEqual(rounds_called, [1, 2])
        self.assertEqual(result.selected.source_id, "C4")
        self.assertEqual(len(result.rounds), 2)

        def all_worse(round_index, parent):
            ids = ("C1", "C2", "C3") if round_index == 1 else ("C4", "C5", "C6")
            return tuple(_option(candidate_id, 8 + i, 0.9, parent.source_id) for i, candidate_id in enumerate(ids))

        protected = execute_ra_policy(original, all_worse)
        self.assertEqual(protected.selected.source_id, "original")
        self.assertEqual(protected.selected.rank, 7)
        self.assertTrue(all(row["selection_pool"][0] == "original" for row in protected.rounds))

    def test_summary_preserves_50_and_excludes_ineligible_from_metrics(self) -> None:
        combined = []
        for index in range(50):
            if index < 3:
                methods = {
                    "original_e2e": _method(7),
                    "autogeo_api_e2e": _method(8),
                    "query_aware_autogeo": _method(5),
                    "no_feedback_multi_sample": _method(6),
                    "ra_autogeo_e2e": _method(4),
                }
                combined.append({"query_id": f"q{index}", "eligible": True, "methods": methods})
            else:
                combined.append({"query_id": f"q{index}", "eligible": False, "methods": None})
        summary = summarize_records(
            combined,
            retriever_config_sha256=FROZEN_R1_CONFIG_SHA256,
            corpus_manifest_sha256="corpus",
            target_manifest_sha256="target",
        )
        self.assertEqual(summary["total_queries"], 50)
        self.assertEqual(summary["eligible_queries"], 3)
        self.assertEqual(summary["ineligible_queries"], 47)
        self.assertEqual(summary["method_statistics"]["ra_autogeo_e2e"]["n"], 3)
        self.assertFalse(summary["ineligible_counted_as_retrieval_failure"])

    def test_immutable_artifact_reused_and_changed_hard_fails(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            config = _config(Path(temporary))
            path = config.output_root / "evaluation" / "test50_retrieval" / "x.json"
            self.assertEqual(_persist(path, {"value": 1}, config)[0], "created")
            self.assertEqual(_persist(path, {"value": 1}, config)[0], "reused_identical")
            self.assertFalse(path.stat().st_mode & 0o222)
            with self.assertRaisesRegex(FileExistsError, "changed TEST50 artifact"):
                _persist(path, {"value": 2}, config)


if __name__ == "__main__":
    unittest.main()
