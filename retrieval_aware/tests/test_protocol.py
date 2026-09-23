from __future__ import annotations

import unittest
from dataclasses import replace

from retrieval_aware.config import E7Config
from retrieval_aware.e2e_evaluator import (
    E2EEvaluationResult,
    build_method_specific_ge_context,
)
from retrieval_aware.local_ranker import CounterfactualRankResult
from retrieval_aware.pool_builder import RetrievalDocument, RetrievalPool
from retrieval_aware.protocol import (
    AUTOGEO_FILTERED_RULES,
    PROTOCOL_SPECS,
    TRACK_B_PROTOCOLS,
    ProtocolId,
    validate_protocol_registry,
    validate_track_b_paired_experiment,
)
from retrieval_aware.ra_rewriter import (
    RetrieverFeedback,
    RewriteCandidate,
    RewriteRequest,
    RewriteResult,
    validate_track_b_sibling_results,
    validate_rewrite_request,
)
from retrieval_aware.target_selector import TargetMetadata
from retrieval_aware.utils import E7SafetyError, assert_frozen_input_is_read_only


class ProtocolTests(unittest.TestCase):
    def setUp(self) -> None:
        self.config = E7Config()

    def test_config_is_frozen_to_v1_values(self) -> None:
        self.config.validate()
        self.assertEqual(self.config.pool_size, 100)
        self.assertEqual(self.config.ge_top_k, 5)
        self.assertEqual(self.config.preferred_target_rank_min, 6)
        self.assertEqual(self.config.preferred_target_rank_max, 20)
        self.assertEqual(self.config.max_target_rank, 100)
        self.assertEqual(self.config.num_candidates_per_round, 3)
        self.assertEqual(self.config.max_rounds, 2)
        self.assertEqual(self.config.retrieval_backend, "researchy_pooled_local")
        self.assertEqual(
            self.config.retrieval_model,
            "openbmb/MiniCPM-Embedding-Light",
        )
        self.assertEqual(self.config.retrieval_query_instruction, "Query:")
        self.assertIsNone(self.config.retrieval_document_instruction)
        self.assertEqual(self.config.embedding_dimension, 1024)
        self.assertEqual(self.config.retrieval_max_input_tokens, 8192)
        self.assertEqual(self.config.embedding_normalization, "l2")
        self.assertEqual(self.config.similarity_function, "dot_product")
        self.assertEqual(self.config.dev_corpus_sources, ("rule_pool", "dev"))
        self.assertEqual(self.config.test_corpus_sources, ("rule_pool", "test"))

    def test_registry_and_ra_rules_are_frozen(self) -> None:
        validate_protocol_registry()
        self.assertEqual(set(PROTOCOL_SPECS), set(ProtocolId))
        self.assertEqual(
            PROTOCOL_SPECS[ProtocolId.AUTOGEO_API_E2E].ruleset_id,
            AUTOGEO_FILTERED_RULES,
        )
        self.assertEqual(
            PROTOCOL_SPECS[ProtocolId.RA_AUTOGEO_E2E].ruleset_id,
            AUTOGEO_FILTERED_RULES,
        )

    def test_only_all_three_track_b_arms_form_primary_pairing(self) -> None:
        validate_track_b_paired_experiment(TRACK_B_PROTOCOLS)
        with self.assertRaises(ValueError):
            validate_track_b_paired_experiment(
                [
                    ProtocolId.VANILLA_ORIGINAL_PROTOCOL,
                    ProtocolId.ORIGINAL_E2E,
                ]
            )

    def test_autogeo_e2e_rejects_retrieval_feedback(self) -> None:
        document = RetrievalDocument(
            document_id="target",
            rank=6,
            score=0.5,
            text="factual ground truth",
            url="https://example.com/target",
            clueweb_url_hash="HASH-TARGET",
            language="en",
        )
        pool = RetrievalPool(
            schema_version="e7_retrieval_pool_v1",
            pool_id="pool-q1",
            e7_query_id="q1",
            autogeo_question_id="local-q1",
            researchy_question_id="researchy-q1",
            query="query",
            corpus="clueweb22-b",
            retriever="fixed",
            requested_top_n=1,
            documents=(document,),
        )
        target = TargetMetadata(
            query_id="q1",
            document_id="target",
            original_rank=6,
            original_retrieval_score=0.5,
            relevance_source="clicked",
            relevance_identifier="clicked-id",
            selection_band="preferred",
        )
        request = RewriteRequest(
            protocol_id=ProtocolId.AUTOGEO_API_E2E,
            query_id="q1",
            query="query",
            target=target,
            original_target_text=document.text,
            pool=pool,
            ruleset_id=AUTOGEO_FILTERED_RULES,
            feedback_history=(
                RetrieverFeedback(
                    round_index=1,
                    candidate_id="forbidden",
                    target_rank=5,
                    target_score=0.6,
                    rank_delta_from_original=1,
                ),
            ),
        )
        with self.assertRaises(ValueError):
            validate_rewrite_request(request)

    def test_miss_still_has_real_top5_and_requires_geu(self) -> None:
        rewrite = RewriteResult(
            protocol_id=ProtocolId.AUTOGEO_API_E2E,
            query_id="q1",
            target_document_id="d6",
            pool_id="pool-q1",
            original_text="original",
            rewritten_text="rewrite",
            model_name="model",
            prompt_version="unchanged-autogeo",
            ruleset_id=AUTOGEO_FILTERED_RULES,
            rules_sha256="a" * 64,
        )
        rank = CounterfactualRankResult(
            query_id="q1",
            target_document_id="d6",
            original_rank=6,
            counterfactual_rank=6,
            target_score=0.1,
            ranked_document_ids=tuple(f"d{i}" for i in range(1, 101)),
        )
        context = build_method_specific_ge_context(rewrite, rank, self.config)
        self.assertEqual(context.document_ids, ("d1", "d2", "d3", "d4", "d5"))
        self.assertIsNone(context.target_source_index)

        completed = E2EEvaluationResult(
            query_id="q1",
            target_document_id="d6",
            protocol_id=ProtocolId.AUTOGEO_API_E2E,
            original_rank=6,
            counterfactual_rank=6,
            method_top5_document_ids=context.document_ids,
            target_source_index=None,
            target_geo_score=None,
            target_e2e_visibility=0.0,
            geu_score={"KPR": 0.5, "KPC": 0.0},
            response="Answer generated from the actual Top-5.",
        )
        completed.validate(self.config)
        with self.assertRaises(ValueError):
            replace(completed, geu_score={}).validate(self.config)

    def test_hit_uses_dynamic_zero_based_source_index(self) -> None:
        rewrite = RewriteResult(
            protocol_id=ProtocolId.RA_AUTOGEO_E2E,
            query_id="q1",
            target_document_id="target",
            pool_id="pool-q1",
            original_text="original",
            rewritten_text="candidate",
            model_name="model",
            prompt_version="ra-v1",
            ruleset_id=AUTOGEO_FILTERED_RULES,
            rules_sha256="a" * 64,
            selected_round=1,
            selected_candidate_id="r1-c1",
            candidate_history=(
                RewriteCandidate(
                    candidate_id="r1-c1",
                    round_index=1,
                    candidate_index=1,
                    rewritten_text="candidate",
                    target_rank=3,
                    target_score=0.9,
                ),
            ),
        )
        rank = CounterfactualRankResult(
            query_id="q1",
            target_document_id="target",
            original_rank=10,
            counterfactual_rank=3,
            target_score=0.9,
            ranked_document_ids=("d1", "d2", "target", "d4", "d5", "d6"),
        )
        context = build_method_specific_ge_context(rewrite, rank, self.config)
        self.assertEqual(context.target_source_index, 2)

    def test_track_b_siblings_share_original_target_pool_and_rules(self) -> None:
        common = {
            "query_id": "q1",
            "target_document_id": "target",
            "pool_id": "pool-q1",
            "original_text": "factual ground truth",
            "model_name": "model",
        }
        original = RewriteResult(
            protocol_id=ProtocolId.ORIGINAL_E2E,
            rewritten_text="factual ground truth",
            prompt_version=None,
            ruleset_id=None,
            rules_sha256=None,
            **common,
        )
        autogeo = RewriteResult(
            protocol_id=ProtocolId.AUTOGEO_API_E2E,
            rewritten_text="autogeo candidate",
            prompt_version="unchanged-autogeo",
            ruleset_id=AUTOGEO_FILTERED_RULES,
            rules_sha256="a" * 64,
            **common,
        )
        ra = RewriteResult(
            protocol_id=ProtocolId.RA_AUTOGEO_E2E,
            rewritten_text="ra candidate",
            prompt_version="ra-v1",
            ruleset_id=AUTOGEO_FILTERED_RULES,
            rules_sha256="a" * 64,
            selected_round=1,
            selected_candidate_id="r1-c1",
            candidate_history=(
                RewriteCandidate(
                    candidate_id="r1-c1",
                    round_index=1,
                    candidate_index=1,
                    rewritten_text="ra candidate",
                    target_rank=4,
                    target_score=0.8,
                ),
            ),
            **common,
        )
        siblings = (original, autogeo, ra)
        validate_track_b_sibling_results(siblings, self.config)
        with self.assertRaises(ValueError):
            validate_track_b_sibling_results(
                (original, autogeo, replace(ra, rules_sha256="b" * 64)),
                self.config,
            )

    def test_data_directory_cannot_be_an_output_root(self) -> None:
        with self.assertRaises(E7SafetyError):
            assert_frozen_input_is_read_only(
                replace(self.config, output_root=self.config.data_root / "e7")
            )

    def test_frozen_directory_cannot_be_an_output_root(self) -> None:
        with self.assertRaises(E7SafetyError):
            assert_frozen_input_is_read_only(
                replace(self.config, output_root=self.config.frozen_root / "e7")
            )


if __name__ == "__main__":
    unittest.main()
