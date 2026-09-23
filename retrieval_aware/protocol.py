"""Frozen experiment protocol identifiers and invariants for E7."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Iterable


class ExperimentTrack(str, Enum):
    """The two populations/protocol families must not be paired together."""

    TRACK_A_ORIGINAL = "track_a_original_protocol"
    TRACK_B_RETRIEVAL_AWARE = "track_b_retrieval_aware_protocol"


class ProtocolId(str, Enum):
    """Canonical, non-ambiguous identifiers used in every E7 artifact."""

    VANILLA_ORIGINAL_PROTOCOL = "vanilla_original_protocol"
    AUTOGEO_API_ORIGINAL_PROTOCOL = "autogeo_api_original_protocol"
    ORIGINAL_E2E = "original_e2e"
    AUTOGEO_API_E2E = "autogeo_api_e2e"
    RA_AUTOGEO_E2E = "ra_autogeo_e2e"


AUTOGEO_FILTERED_RULES = "autogeo_filtered_rules"
TRACK_B_PAIRED_GROUP = "e7_same_query_pool_target"
TRACK_B_RETRIEVAL_BACKEND = "researchy_pooled_local"


@dataclass(frozen=True)
class ProtocolSpec:
    protocol_id: ProtocolId
    track: ExperimentTrack
    uses_fixed_researchy_top5: bool
    uses_e7_retrieval_pool: bool
    retrieval_backend: str | None
    rewrites_target: bool
    uses_unchanged_autogeo_api: bool
    ruleset_id: str | None
    allows_retriever_feedback: bool
    iterative_candidate_selection: bool
    performs_local_reindex: bool
    paired_group: str | None


PROTOCOL_SPECS: dict[ProtocolId, ProtocolSpec] = {
    ProtocolId.VANILLA_ORIGINAL_PROTOCOL: ProtocolSpec(
        protocol_id=ProtocolId.VANILLA_ORIGINAL_PROTOCOL,
        track=ExperimentTrack.TRACK_A_ORIGINAL,
        uses_fixed_researchy_top5=True,
        uses_e7_retrieval_pool=False,
        retrieval_backend=None,
        rewrites_target=False,
        uses_unchanged_autogeo_api=False,
        ruleset_id=None,
        allows_retriever_feedback=False,
        iterative_candidate_selection=False,
        performs_local_reindex=False,
        paired_group=None,
    ),
    ProtocolId.AUTOGEO_API_ORIGINAL_PROTOCOL: ProtocolSpec(
        protocol_id=ProtocolId.AUTOGEO_API_ORIGINAL_PROTOCOL,
        track=ExperimentTrack.TRACK_A_ORIGINAL,
        uses_fixed_researchy_top5=True,
        uses_e7_retrieval_pool=False,
        retrieval_backend=None,
        rewrites_target=True,
        uses_unchanged_autogeo_api=True,
        ruleset_id=AUTOGEO_FILTERED_RULES,
        allows_retriever_feedback=False,
        iterative_candidate_selection=False,
        performs_local_reindex=False,
        paired_group=None,
    ),
    ProtocolId.ORIGINAL_E2E: ProtocolSpec(
        protocol_id=ProtocolId.ORIGINAL_E2E,
        track=ExperimentTrack.TRACK_B_RETRIEVAL_AWARE,
        uses_fixed_researchy_top5=False,
        uses_e7_retrieval_pool=True,
        retrieval_backend=TRACK_B_RETRIEVAL_BACKEND,
        rewrites_target=False,
        uses_unchanged_autogeo_api=False,
        ruleset_id=None,
        allows_retriever_feedback=False,
        iterative_candidate_selection=False,
        performs_local_reindex=True,
        paired_group=TRACK_B_PAIRED_GROUP,
    ),
    ProtocolId.AUTOGEO_API_E2E: ProtocolSpec(
        protocol_id=ProtocolId.AUTOGEO_API_E2E,
        track=ExperimentTrack.TRACK_B_RETRIEVAL_AWARE,
        uses_fixed_researchy_top5=False,
        uses_e7_retrieval_pool=True,
        retrieval_backend=TRACK_B_RETRIEVAL_BACKEND,
        rewrites_target=True,
        uses_unchanged_autogeo_api=True,
        ruleset_id=AUTOGEO_FILTERED_RULES,
        allows_retriever_feedback=False,
        iterative_candidate_selection=False,
        performs_local_reindex=True,
        paired_group=TRACK_B_PAIRED_GROUP,
    ),
    ProtocolId.RA_AUTOGEO_E2E: ProtocolSpec(
        protocol_id=ProtocolId.RA_AUTOGEO_E2E,
        track=ExperimentTrack.TRACK_B_RETRIEVAL_AWARE,
        uses_fixed_researchy_top5=False,
        uses_e7_retrieval_pool=True,
        retrieval_backend=TRACK_B_RETRIEVAL_BACKEND,
        rewrites_target=True,
        uses_unchanged_autogeo_api=False,
        ruleset_id=AUTOGEO_FILTERED_RULES,
        allows_retriever_feedback=True,
        iterative_candidate_selection=True,
        performs_local_reindex=True,
        paired_group=TRACK_B_PAIRED_GROUP,
    ),
}

TRACK_A_PROTOCOLS = frozenset(
    {
        ProtocolId.VANILLA_ORIGINAL_PROTOCOL,
        ProtocolId.AUTOGEO_API_ORIGINAL_PROTOCOL,
    }
)
TRACK_B_PROTOCOLS = frozenset(
    {
        ProtocolId.ORIGINAL_E2E,
        ProtocolId.AUTOGEO_API_E2E,
        ProtocolId.RA_AUTOGEO_E2E,
    }
)


def validate_protocol_registry() -> None:
    """Fail fast if the frozen E7 protocol definitions become inconsistent."""
    if set(PROTOCOL_SPECS) != set(ProtocolId):
        raise ValueError("every ProtocolId must have exactly one ProtocolSpec")

    for protocol_id, spec in PROTOCOL_SPECS.items():
        if spec.protocol_id is not protocol_id:
            raise ValueError(f"registry key/spec mismatch for {protocol_id.value}")
        if spec.track is ExperimentTrack.TRACK_A_ORIGINAL:
            if not spec.uses_fixed_researchy_top5 or spec.uses_e7_retrieval_pool:
                raise ValueError("Track A must use only the original fixed Top-5")
            if spec.performs_local_reindex or spec.paired_group is not None:
                raise ValueError("Track A cannot be locally re-indexed or E7-paired")
            if spec.retrieval_backend is not None:
                raise ValueError("Track A does not use an E7 retrieval backend")
        else:
            if spec.uses_fixed_researchy_top5 or not spec.uses_e7_retrieval_pool:
                raise ValueError("Track B must use only the E7 retrieval pool")
            if not spec.performs_local_reindex:
                raise ValueError("every Track B arm must locally re-index")
            if spec.paired_group != TRACK_B_PAIRED_GROUP:
                raise ValueError("all Track B arms must share one paired group")
            if spec.retrieval_backend != TRACK_B_RETRIEVAL_BACKEND:
                raise ValueError("Track B must use Researchy-GEO pooled local retrieval")

    for protocol_id in (
        ProtocolId.AUTOGEO_API_ORIGINAL_PROTOCOL,
        ProtocolId.AUTOGEO_API_E2E,
    ):
        spec = PROTOCOL_SPECS[protocol_id]
        if spec.allows_retriever_feedback or spec.iterative_candidate_selection:
            raise ValueError("unchanged AutoGEO_API cannot receive retrieval feedback")

    autogeo_rules = PROTOCOL_SPECS[ProtocolId.AUTOGEO_API_E2E].ruleset_id
    ra_rules = PROTOCOL_SPECS[ProtocolId.RA_AUTOGEO_E2E].ruleset_id
    if autogeo_rules != AUTOGEO_FILTERED_RULES or ra_rules != autogeo_rules:
        raise ValueError("Rules_RA_v1 must equal Rules_AutoGEO")

    ra_spec = PROTOCOL_SPECS[ProtocolId.RA_AUTOGEO_E2E]
    if not ra_spec.allows_retriever_feedback or not ra_spec.iterative_candidate_selection:
        raise ValueError("RA-AutoGEO must be retriever-in-the-loop and iterative")


def validate_track_b_paired_experiment(protocol_ids: Iterable[ProtocolId]) -> None:
    """Require exactly the three same-query/same-pool/same-target Track B arms."""
    selected = tuple(protocol_ids)
    if len(selected) != len(set(selected)):
        raise ValueError("paired protocol identifiers must be unique")
    if frozenset(selected) != TRACK_B_PROTOCOLS:
        raise ValueError(
            "the paired experiment is exactly original_e2e, autogeo_api_e2e, "
            "and ra_autogeo_e2e; Track A must be analyzed separately"
        )


validate_protocol_registry()
