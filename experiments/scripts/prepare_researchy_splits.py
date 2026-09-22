import json
import random
import hashlib
from pathlib import Path

from datasets import load_dataset


# ============================================================
# Experiment configuration
# ============================================================

DATASET_REPO = "cx-cmu/Researchy-GEO"
DATASET_CONFIG = "main"

SEED = 20260922

DEV_SIZE = 20 # 开发
RULE_POOL_SIZE = 100 # 规则池
TEST_SIZE = 50 # 测试
QUALITY_SIZE = 10 # 质量


ROOT = Path("experiments")

SPLIT_DIR = ROOT / "splits"
FROZEN_DIR = ROOT / "frozen"
CONFIG_DIR = ROOT / "config"


# ============================================================
# Helpers
# ============================================================

def save_json(path: Path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)

    with open(path, "w", encoding="utf-8") as f:
        json.dump(
            obj,
            f,
            ensure_ascii=False,
            indent=2,
        )


def record_to_json(row):
    """
    Convert one Hugging Face Dataset record into a plain Python dict.
    """
    return {
        key: row[key]
        for key in row.keys()
    }


def build_chunk(split, indices):
    """
    Convert selected records into AutoGEO's expected format:

    {
        "query_id_1": {...},
        "query_id_2": {...}
    }
    """

    result = {}

    for idx in indices:
        row = record_to_json(split[idx])

        query_id = str(row.pop("query_id"))

        if query_id in result:
            raise RuntimeError(
                f"Duplicate query_id detected: {query_id}"
            )

        result[query_id] = row

    return result


def sha256_json(obj):
    raw = json.dumps(
        obj,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")

    return hashlib.sha256(raw).hexdigest()


def ensure_output_not_exists():
    """
    Refuse to silently overwrite frozen experimental data.
    """

    important_paths = [
        FROZEN_DIR / "researchy_dev20",
        FROZEN_DIR / "researchy_rulepool100",
        FROZEN_DIR / "researchy_test50",
        SPLIT_DIR / "researchy_dev20_ids.json",
        SPLIT_DIR / "researchy_test50_ids.json",
    ]

    existing = [
        str(p)
        for p in important_paths
        if p.exists()
    ]

    if existing:
        print("ERROR: Frozen experiment files already exist:")
        for p in existing:
            print("  ", p)

        print()
        print(
            "Refusing to overwrite them automatically. "
            "Delete them manually only if you intentionally "
            "want to create a completely new split."
        )

        raise SystemExit(1)


# ============================================================
# Main
# ============================================================

def main():

    ensure_output_not_exists()

    SPLIT_DIR.mkdir(parents=True, exist_ok=True)
    FROZEN_DIR.mkdir(parents=True, exist_ok=True)
    CONFIG_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 80)
    print("Loading Researchy-GEO")
    print("=" * 80)

    print("\nLoading train split...")

    train = load_dataset(
        DATASET_REPO,
        DATASET_CONFIG,
        split="train",
    )

    print("Loading test split...")

    test = load_dataset(
        DATASET_REPO,
        DATASET_CONFIG,
        split="test",
    )

    print()
    print(f"Train size: {len(train)}")
    print(f"Test size : {len(test)}")

    if len(train) < DEV_SIZE + RULE_POOL_SIZE:
        raise RuntimeError(
            "Train split is too small."
        )

    if len(test) < TEST_SIZE:
        raise RuntimeError(
            "Test split is too small."
        )

    # ========================================================
    # Dev + Rule Pool
    # ========================================================

    train_rng = random.Random(SEED)

    selected_train_indices = train_rng.sample(
        range(len(train)),
        DEV_SIZE + RULE_POOL_SIZE,
    )

    dev_indices = sorted(
        selected_train_indices[:DEV_SIZE]
    )

    rule_pool_indices = sorted(
        selected_train_indices[DEV_SIZE:]
    )

    # ========================================================
    # Test
    # ========================================================

    test_rng = random.Random(SEED + 1)

    test_indices = sorted(
        test_rng.sample(
            range(len(test)),
            TEST_SIZE,
        )
    )

    # ========================================================
    # Build datasets
    # ========================================================

    dev_data = build_chunk(
        train,
        dev_indices,
    )

    rule_pool_data = build_chunk(
        train,
        rule_pool_indices,
    )

    test_data = build_chunk(
        test,
        test_indices,
    )

    dev_ids = list(dev_data.keys())
    rule_pool_ids = list(rule_pool_data.keys())
    test_ids = list(test_data.keys())

    # ========================================================
    # Quality10
    # ========================================================

    quality_rng = random.Random(SEED + 2)

    quality_ids = sorted(
        quality_rng.sample(
            test_ids,
            QUALITY_SIZE,
        )
    )

    # ========================================================
    # Sanity checks
    # ========================================================

    assert len(dev_ids) == DEV_SIZE
    assert len(rule_pool_ids) == RULE_POOL_SIZE
    assert len(test_ids) == TEST_SIZE
    assert len(quality_ids) == QUALITY_SIZE

    assert set(dev_ids).isdisjoint(rule_pool_ids)
    assert set(dev_ids).isdisjoint(test_ids)
    assert set(rule_pool_ids).isdisjoint(test_ids)

    assert set(quality_ids).issubset(test_ids)

    # Check required fields
    all_datasets = {
        "Dev20": dev_data,
        "RulePool100": rule_pool_data,
        "Test50": test_data,
    }

    for dataset_name, dataset in all_datasets.items():

        for qid, item in dataset.items():

            required_fields = [
                "query",
                "text_list",
                "target_id",
            ]

            for field in required_fields:
                if field not in item:
                    raise RuntimeError(
                        f"{dataset_name}/{qid}: "
                        f"missing field '{field}'"
                    )

            target_id = item["target_id"]
            text_list = item["text_list"]

            if not (
                0 <= target_id < len(text_list)
            ):
                raise RuntimeError(
                    f"{dataset_name}/{qid}: "
                    f"invalid target_id={target_id}"
                )

    # ========================================================
    # Save IDs
    # ========================================================

    save_json(
        SPLIT_DIR / "researchy_dev20_ids.json",
        dev_ids,
    )

    save_json(
        SPLIT_DIR / "researchy_rulepool100_ids.json",
        rule_pool_ids,
    )

    save_json(
        SPLIT_DIR / "researchy_test50_ids.json",
        test_ids,
    )

    save_json(
        SPLIT_DIR / "researchy_quality10_ids.json",
        quality_ids,
    )

    # ========================================================
    # Save frozen data
    # ========================================================

    save_json(
        FROZEN_DIR
        / "researchy_dev20"
        / "datachunk_0.json",
        dev_data,
    )

    save_json(
        FROZEN_DIR
        / "researchy_rulepool100"
        / "datachunk_0.json",
        rule_pool_data,
    )

    save_json(
        FROZEN_DIR
        / "researchy_test50"
        / "datachunk_0.json",
        test_data,
    )

    # ========================================================
    # Metadata / hashes
    # ========================================================

    metadata = {
        "dataset_repo": DATASET_REPO,
        "dataset_config": DATASET_CONFIG,

        "seed": SEED,

        "sizes": {
            "dev": DEV_SIZE,
            "rule_pool": RULE_POOL_SIZE,
            "test": TEST_SIZE,
            "quality": QUALITY_SIZE,
        },

        "source_splits": {
            "dev": "main/train",
            "rule_pool": "main/train",
            "test": "main/test",
            "quality": "subset of Test50",
        },

        "hashes": {
            "dev20_sha256":
                sha256_json(dev_data),

            "rulepool100_sha256":
                sha256_json(rule_pool_data),

            "test50_sha256":
                sha256_json(test_data),
        },
    }

    save_json(
        CONFIG_DIR / "researchy_split_metadata.json",
        metadata,
    )

    # ========================================================
    # Summary
    # ========================================================

    print()
    print("=" * 80)
    print("Researchy-GEO split successfully frozen")
    print("=" * 80)

    print(f"Dev20       : {len(dev_ids)}")
    print(f"RulePool100 : {len(rule_pool_ids)}")
    print(f"Test50      : {len(test_ids)}")
    print(f"Quality10   : {len(quality_ids)}")

    print()

    print(
        "Dev/Test overlap:",
        len(set(dev_ids) & set(test_ids)),
    )

    print(
        "Rule/Test overlap:",
        len(set(rule_pool_ids) & set(test_ids)),
    )

    print(
        "Dev/Rule overlap:",
        len(set(dev_ids) & set(rule_pool_ids)),
    )

    print()

    print("Frozen test hash:")
    print(metadata["hashes"]["test50_sha256"])


if __name__ == "__main__":
    main()