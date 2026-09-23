"""Read-only audit of Researchy-GEO document-bearing repository artifacts."""

from __future__ import annotations

import json
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable

from .config import E7Config
from .utils import (
    canonical_json_sha256,
    normalized_text_sha256,
    read_json,
    sha256_file,
    write_e7_json_once,
)


@dataclass(frozen=True)
class DocumentSourceAudit:
    source_name: str
    location: str
    artifact_role: str
    split_semantics: str
    number_of_queries: int
    document_slots: int
    already_in_pool_v1: str
    dev_background_decision: str
    decision_reason: str

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def _chunk_rows(directory: Path) -> list[tuple[str, dict[str, object]]]:
    rows: list[tuple[str, dict[str, object]]] = []
    for path in sorted(directory.glob("datachunk_*.json")):
        raw = read_json(path)
        if not isinstance(raw, dict):
            raise ValueError(f"Researchy chunk must be an object: {path}")
        for question_id, row in raw.items():
            if not isinstance(row, dict):
                raise ValueError(f"Researchy row must be an object: {path}/{question_id}")
            rows.append((str(question_id), row))
    return rows


def _structured_counts(directory: Path) -> tuple[int, int, Counter[int]]:
    rows = _chunk_rows(directory)
    lengths: Counter[int] = Counter()
    for question_id, row in rows:
        query = row.get("query")
        documents = row.get("text_list")
        if not isinstance(query, str) or not query.strip():
            raise ValueError(f"missing query: {directory}/{question_id}")
        if not isinstance(documents, list) or not documents:
            raise ValueError(f"missing source documents: {directory}/{question_id}")
        if not all(isinstance(text, str) and text.strip() for text in documents):
            raise ValueError(f"invalid source document: {directory}/{question_id}")
        lengths[len(documents)] += 1
    ids = [question_id for question_id, _ in rows]
    if len(ids) != len(set(ids)):
        raise ValueError(f"duplicate query ID across chunks: {directory}")
    return len(rows), sum(length * count for length, count in lengths.items()), lengths


def _relative(path: Path, config: E7Config) -> str:
    return str(path.resolve().relative_to(config.project_root.resolve()))


def _entry(
    config: E7Config,
    source_name: str,
    location: Path,
    artifact_role: str,
    split_semantics: str,
    number_of_queries: int,
    document_slots: int,
    already_in_pool_v1: str,
    decision: str,
    reason: str,
) -> DocumentSourceAudit:
    return DocumentSourceAudit(
        source_name=source_name,
        location=_relative(location, config),
        artifact_role=artifact_role,
        split_semantics=split_semantics,
        number_of_queries=number_of_queries,
        document_slots=document_slots,
        already_in_pool_v1=already_in_pool_v1,
        dev_background_decision=decision,
        decision_reason=reason,
    )


def _hashes_from_rows(rows: Iterable[tuple[str, dict[str, object]]]) -> set[str]:
    return {
        normalized_text_sha256(text)
        for _, row in rows
        for text in row["text_list"]
    }


def build_document_source_audit(config: E7Config) -> dict[str, object]:
    """Freeze source decisions without consulting any retrieval result."""
    config.validate()
    data_root = config.data_root / "Researchy-GEO"
    train_dir = data_root / "train"
    test_dir = data_root / "test"
    train_count, train_slots, train_lengths = _structured_counts(train_dir)
    test_count, test_slots, _ = _structured_counts(test_dir)
    rule_count, rule_slots, _ = _structured_counts(config.rule_pool_query_dir)
    dev_count, dev_slots, _ = _structured_counts(config.dev_query_dir)
    test50_count, test50_slots, _ = _structured_counts(config.test_query_dir)

    if (train_count, train_slots, dict(train_lengths)) != (9998, 49995, {5: 9997, 10: 1}):
        raise ValueError("unexpected canonical Researchy train composition")
    if (test_count, test_slots) != (1000, 5000):
        raise ValueError("unexpected canonical Researchy test composition")

    keypoint_v2 = {
        name: config.frozen_root / directory
        for name, directory in {
            "researchy_rulepool100_keypoint_v2": "researchy_rulepool100_keypoint_v2",
            "researchy_dev20_keypoint_v2": "researchy_dev20_keypoint_v2",
            "researchy_test50_keypoint_v2": "researchy_test50_keypoint_v2",
        }.items()
    }
    kp_counts = {
        name: _structured_counts(path)[:2]
        for name, path in keypoint_v2.items()
    }
    e0_count, e0_slots, _ = _structured_counts(data_root / "test_dev20_e0")
    e1_count, e1_slots, _ = _structured_counts(data_root / "test_dev20_e1")

    grpo_eval = read_json(data_root / "RL" / "grpo_eval.json")
    rule_candidate = read_json(data_root / "RL" / "rule_candidate.json")
    grpo_input = read_json(data_root / "RL" / "grpo_input.json")
    finetune = read_json(data_root / "RL" / "finetune.json")
    inference = read_json(data_root / "RL" / "inference.json")
    if not all(isinstance(value, dict) for value in (grpo_eval, rule_candidate)):
        raise ValueError("unexpected Researchy RL mapping schema")

    entries = [
        _entry(config, "researchy_geo_train_original", train_dir, "canonical split", "train", train_count, train_slots, "partial: RULE_POOL100 and DEV20 subsets", "include", "canonical complete non-test training/background source"),
        _entry(config, "researchy_rulepool100_v1", config.rule_pool_query_dir, "frozen E7 source", "train/background", rule_count, rule_slots, "yes: complete", "include", "explicit fixed shared background required by Pool V2 protocol"),
        _entry(config, "researchy_dev20_v1", config.dev_query_dir, "frozen E7 source", "train/development", dev_count, dev_slots, "yes: complete", "include", "fixed DEV query candidates required by Pool V2 protocol"),
        _entry(config, "researchy_geo_test_original", test_dir, "canonical split", "test-only", test_count, test_slots, "partial only in separate V1 TEST corpus", "exclude", "test-only source"),
        _entry(config, "researchy_test50_v1", config.test_query_dir, "frozen E7 source", "test-only", test50_count, test50_slots, "only in separate V1 TEST corpus", "exclude", "held-out TEST50 is forbidden in DEV"),
        _entry(config, "researchy_rulepool100_keypoint_v2", keypoint_v2["researchy_rulepool100_keypoint_v2"], "frozen derived subset", "train/background", *kp_counts["researchy_rulepool100_keypoint_v2"], "no", "exclude_redundant", "exact subset of canonical train; contributes no new physical documents"),
        _entry(config, "researchy_dev20_keypoint_v2", keypoint_v2["researchy_dev20_keypoint_v2"], "frozen derived subset", "test-derived development", *kp_counts["researchy_dev20_keypoint_v2"], "no", "exclude", "drawn from canonical test split"),
        _entry(config, "researchy_test50_keypoint_v2", keypoint_v2["researchy_test50_keypoint_v2"], "frozen derived subset", "test-only", *kp_counts["researchy_test50_keypoint_v2"], "no", "exclude", "held-out test-only source"),
        _entry(config, "researchy_test_dev20_e0", data_root / "test_dev20_e0", "evaluation result artifact", "test-derived", e0_count, e0_slots, "no", "exclude", "test-derived and contains generated evaluation fields"),
        _entry(config, "researchy_test_dev20_e1", data_root / "test_dev20_e1", "attack result artifact", "test-derived", e1_count, e1_slots, "no", "exclude", "test-derived and contains Hijack/Poison result fields"),
        _entry(config, "researchy_rl_grpo_eval", data_root / "RL" / "grpo_eval.json", "RL derived replica", "train-derived", len(grpo_eval), sum(len(row["text_list"]) for row in grpo_eval.values()), "no", "exclude_redundant", "exact row-level replica of canonical train with added RL metadata"),
        _entry(config, "researchy_rl_rule_candidate", data_root / "RL" / "rule_candidate.json", "RL pair projection", "train-derived", len(rule_candidate), 2 * len(rule_candidate), "no", "exclude_redundant", "good/bad documents are exact selections from canonical train text_list"),
        _entry(config, "researchy_rl_grpo_input", data_root / "RL" / "grpo_input.json", "serialized RL prompt", "train-derived", len(grpo_input), len(grpo_input), "no", "exclude", "prompt serialization is not a canonical Researchy document source"),
        _entry(config, "researchy_rl_finetune", data_root / "RL" / "finetune.json", "generated rewrite training artifact", "train-derived synthetic", len(finetune), 2 * len(finetune), "no", "exclude", "contains generated rewritten outputs"),
        _entry(config, "researchy_rl_inference", data_root / "RL" / "inference.json", "inference prompt artifact", "test-derived", len(inference), len(inference), "no", "exclude", "test-derived prompt serialization"),
        _entry(config, "researchy_keypoint_annotations", data_root / "key_point", "question annotation", "test annotation", 1000, 0, "no", "exclude", "contains questions/keypoints but no source documents"),
        _entry(config, "researchy_rule_extraction_pairs", data_root / "rule_sets" / "gemini-2.5-flash-lite" / "individual_results", "rule-analysis artifact", "train-derived", 10, 20, "no", "exclude_redundant", "document pairs are exact projections of canonical train and explanations are generated"),
    ]

    train_rows = _chunk_rows(train_dir)
    test50_rows = _chunk_rows(config.test_query_dir)
    train_by_id = {question_id: row for question_id, row in train_rows}
    test_by_id = {question_id: row for question_id, row in _chunk_rows(test_dir)}
    for question_id, row in grpo_eval.items():
        original = train_by_id.get(str(question_id))
        if original is None or any(
            row.get(key) != original.get(key)
            for key in ("query", "target_id", "text_list")
        ):
            raise ValueError("RL grpo_eval is not an exact canonical-train replica")
    for question_id, row in rule_candidate.items():
        original = train_by_id.get(str(question_id))
        if original is None or row.get("query") != original.get("query"):
            raise ValueError("RL rule candidate query is not from canonical train")
        if row.get("good_document") not in original["text_list"]:
            raise ValueError("RL good document is not from canonical train")
        if row.get("bad_document") not in original["text_list"]:
            raise ValueError("RL bad document is not from canonical train")
    for name, directory in keypoint_v2.items():
        reference = train_by_id if "rulepool" in name else test_by_id
        for question_id, row in _chunk_rows(directory):
            original = reference.get(question_id)
            if original is None or any(
                row.get(key) != original.get(key)
                for key in ("query", "target_id", "text_list")
            ):
                raise ValueError(f"derived split provenance mismatch: {name}")
    train_hashes = _hashes_from_rows(train_rows)
    test50_hashes = _hashes_from_rows(test50_rows)
    collisions = train_hashes & test50_hashes
    excluded_train_slots = sum(
        normalized_text_sha256(text) in collisions
        for _, row in train_rows
        for text in row["text_list"]
    )
    if len(collisions) != 52 or excluded_train_slots != 87:
        raise ValueError("unexpected canonical train/TEST50 physical overlap")

    source_files = sorted(train_dir.glob("datachunk_*.json"))
    source_files += [
        config.rule_pool_query_dir / "datachunk_0.json",
        config.dev_query_dir / "datachunk_0.json",
    ]
    selection_contract = {
        "eligible_source_set": [
            "researchy_geo_train_original",
            "researchy_rulepool100_v1",
            "researchy_dev20_v1",
        ],
        "source_names_in_manifest": list(config.pool_v2_corpus_sources),
        "test50_content_isolation": {
            "identity": "normalized_text_sha256",
            "test50_physical_hash_count": len(test50_hashes),
            "train_test50_collision_hash_count": len(collisions),
            "excluded_train_slot_count": excluded_train_slots,
            "policy": "exclude every physical identity present in TEST50",
        },
        "selection_basis": "split provenance and held-out content isolation only",
        "retrieval_or_eligibility_used": False,
        "incremental_corpus_size_search_used": False,
        "synthetic_documents_allowed": False,
    }
    payload: dict[str, object] = {
        "schema_version": "e7_researchy_document_source_audit_v1",
        "sources": [entry.to_dict() for entry in entries],
        "selection_contract": selection_contract,
        "selected_source_files": [
            {
                "path": _relative(path, config),
                "sha256": sha256_file(path),
            }
            for path in source_files
        ],
    }
    payload["audit_hash"] = canonical_json_sha256(payload)
    return payload


def persist_document_source_audit(
    audit: dict[str, object],
    config: E7Config,
) -> tuple[Path, str]:
    destination = (
        config.output_paths["corpus"] / "pool_v2" / "source_audit.json"
    )
    if destination.exists():
        if read_json(destination) != audit:
            raise FileExistsError(f"refusing to replace changed source audit: {destination}")
        return destination, "reused_identical"
    write_e7_json_once(destination, audit, config)
    destination.chmod(0o444)
    return destination, "created"


def main() -> None:
    config = E7Config()
    audit = build_document_source_audit(config)
    path, action = persist_document_source_audit(audit, config)
    print(json.dumps({"path": str(path), "action": action, **audit}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
