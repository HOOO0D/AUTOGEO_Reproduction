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

# 固定随机种子
SEED = 20260922

DEV_SIZE = 20
RULE_POOL_SIZE = 100
TEST_SIZE = 50
QUALITY_SIZE = 10

# 实验版本
SPLIT_VERSION = "researchy_keypoint_v2"


# ============================================================
# Paths
# ============================================================

ROOT = Path("experiments")

SPLIT_DIR = ROOT / "splits"
FROZEN_DIR = ROOT / "frozen"
CONFIG_DIR = ROOT / "config"

KEYPOINT_DIR = Path(
    "data/Researchy-GEO/key_point"
)

DEV_OUTPUT_DIR = (
    FROZEN_DIR
    / "researchy_dev20_keypoint_v2"
)

RULE_OUTPUT_DIR = (
    FROZEN_DIR
    / "researchy_rulepool100_keypoint_v2"
)

TEST_OUTPUT_DIR = (
    FROZEN_DIR
    / "researchy_test50_keypoint_v2"
)


# ============================================================
# Helpers
# ============================================================

def save_json(path: Path, obj):
    """
    Save JSON with UTF-8 encoding.
    """

    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with open(
        path,
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(
            obj,
            f,
            ensure_ascii=False,
            indent=2,
        )


def load_json(path: Path):
    with open(
        path,
        "r",
        encoding="utf-8",
    ) as f:
        return json.load(f)


def record_to_json(row):
    """
    Convert HuggingFace Dataset record
    into a normal Python dictionary.
    """

    return {
        key: row[key]
        for key in row.keys()
    }


def build_chunk(split, indices):
    """
    Convert selected HF rows into AutoGEO format:

    {
        "query_id": {
            "query": ...,
            "text_list": ...,
            "target_id": ...
        }
    }
    """

    result = {}

    for idx in indices:

        row = record_to_json(
            split[idx]
        )

        query_id = str(
            row.pop("query_id")
        )

        if query_id in result:
            raise RuntimeError(
                f"Duplicate query_id: "
                f"{query_id}"
            )

        result[query_id] = row

    return result


def sha256_json(obj):
    """
    Stable SHA256 hash for a JSON object.
    """

    raw = json.dumps(
        obj,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")

    return hashlib.sha256(
        raw
    ).hexdigest()


def get_keypoint_path(qid):
    return (
        KEYPOINT_DIR
        / f"{qid}_aggregated.json"
    )


def has_valid_keypoints(qid):
    """
    Check that:
    1. aggregated keypoint file exists
    2. contains non-empty key_points
    3. every point has required fields
    """

    path = get_keypoint_path(qid)

    if not path.exists():
        return False

    try:
        data = load_json(path)
    except Exception:
        return False

    key_points = data.get(
        "key_points"
    )

    if not key_points:
        return False

    for point in key_points:

        if (
            "point_number"
            not in point
        ):
            return False

        if (
            "point_content"
            not in point
        ):
            return False

        if not point["point_content"]:
            return False

    return True


def verify_keypoints(qids):
    """
    Fail immediately if any query does not have
    valid official aggregated keypoints.
    """

    missing = []

    invalid = []

    for qid in qids:

        path = get_keypoint_path(
            qid
        )

        if not path.exists():
            missing.append(qid)

        elif not has_valid_keypoints(qid):
            invalid.append(qid)

    if missing:
        raise RuntimeError(
            "Missing keypoint files:\n"
            + "\n".join(missing)
        )

    if invalid:
        raise RuntimeError(
            "Invalid / empty keypoints:\n"
            + "\n".join(invalid)
        )


def sha256_keypoints(qids):
    """
    Hash the complete aggregated keypoint
    annotations for selected queries.
    """

    all_keypoints = {}

    for qid in sorted(qids):

        path = get_keypoint_path(
            qid
        )

        all_keypoints[qid] = (
            load_json(path)
        )

    return sha256_json(
        all_keypoints
    )


def validate_dataset(
    name,
    dataset,
):
    """
    Validate AutoGEO sample format.
    """

    for qid, item in dataset.items():

        required_fields = [
            "query",
            "text_list",
            "target_id",
        ]

        for field in required_fields:

            if field not in item:
                raise RuntimeError(
                    f"{name}/{qid}: "
                    f"missing field "
                    f"'{field}'"
                )

        text_list = item[
            "text_list"
        ]

        target_id = item[
            "target_id"
        ]

        # Researchy-GEO should use
        # exactly five candidate documents
        if len(text_list) != 5:
            raise RuntimeError(
                f"{name}/{qid}: "
                f"expected 5 candidates, "
                f"got {len(text_list)}"
            )

        if not (
            0
            <= target_id
            < len(text_list)
        ):
            raise RuntimeError(
                f"{name}/{qid}: "
                f"invalid target_id="
                f"{target_id}"
            )

        if not item["query"]:
            raise RuntimeError(
                f"{name}/{qid}: "
                "empty query"
            )


def ensure_output_not_exists():
    """
    Protect frozen v2 data from accidental overwrite.
    """

    paths = [
        DEV_OUTPUT_DIR,
        RULE_OUTPUT_DIR,
        TEST_OUTPUT_DIR,

        SPLIT_DIR
        / "researchy_dev20_keypoint_v2_ids.json",

        SPLIT_DIR
        / "researchy_rulepool100_keypoint_v2_ids.json",

        SPLIT_DIR
        / "researchy_test50_keypoint_v2_ids.json",

        SPLIT_DIR
        / "researchy_quality10_keypoint_v2_ids.json",

        CONFIG_DIR
        / "researchy_keypoint_v2_metadata.json",
    ]

    existing = [
        p
        for p in paths
        if p.exists()
    ]

    if existing:

        print(
            "ERROR: keypoint_v2 "
            "experiment files already exist:"
        )

        for p in existing:
            print(" ", p)

        print()
        print(
            "Refusing to overwrite "
            "frozen experimental data."
        )

        raise SystemExit(1)


# ============================================================
# Main
# ============================================================

def main():

    ensure_output_not_exists()

    if not KEYPOINT_DIR.exists():

        raise RuntimeError(
            "Keypoint directory does "
            f"not exist: {KEYPOINT_DIR}"
        )

    SPLIT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    FROZEN_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    CONFIG_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )


    # ========================================================
    # Load dataset
    # ========================================================

    print("=" * 80)
    print(
        "Loading Researchy-GEO"
    )
    print("=" * 80)

    print(
        "\nLoading train split..."
    )

    train = load_dataset(
        DATASET_REPO,
        DATASET_CONFIG,
        split="train",
    )

    print(
        "Loading test split..."
    )

    test = load_dataset(
        DATASET_REPO,
        DATASET_CONFIG,
        split="test",
    )

    print()
    print(
        "Train size:",
        len(train),
    )

    print(
        "Test size :",
        len(test),
    )


    # ========================================================
    # Confirm keypoint coverage
    # ========================================================

    train_keypoint_indices = []

    for idx in range(
        len(train)
    ):

        qid = str(
            train[idx]["query_id"]
        )

        if has_valid_keypoints(qid):
            train_keypoint_indices.append(
                idx
            )


    test_keypoint_indices = []

    for idx in range(
        len(test)
    ):

        qid = str(
            test[idx]["query_id"]
        )

        if has_valid_keypoints(qid):
            test_keypoint_indices.append(
                idx
            )


    print()
    print(
        "Train with valid keypoints:",
        len(train_keypoint_indices),
    )

    print(
        "Test with valid keypoints :",
        len(test_keypoint_indices),
    )


    if len(
        test_keypoint_indices
    ) < (
        DEV_SIZE
        + TEST_SIZE
    ):

        raise RuntimeError(
            "Not enough keypoint-covered "
            "test samples."
        )


    # ========================================================
    # RulePool100
    # ========================================================
    #
    # Rule pool remains in TRAIN.
    #
    # It is used for discovering AutoGEO rules,
    # so KPR/KPC annotations are not required.
    # ========================================================

    rule_rng = random.Random(
        SEED
    )

    rule_pool_indices = sorted(
        rule_rng.sample(
            range(len(train)),
            RULE_POOL_SIZE,
        )
    )


    # ========================================================
    # Dev20 + Test50
    # ========================================================
    #
    # Both are selected from the 1000 official
    # keypoint-annotated test queries.
    #
    # IMPORTANT:
    # Dev and Test are sampled simultaneously,
    # guaranteeing disjointness.
    # ========================================================

    eval_rng = random.Random(
        SEED + 1
    )

    selected_eval_indices = (
        eval_rng.sample(
            test_keypoint_indices,
            DEV_SIZE + TEST_SIZE,
        )
    )

    # Partition BEFORE sorting
    # so randomness is preserved.
    dev_indices = sorted(
        selected_eval_indices[
            :DEV_SIZE
        ]
    )

    test_indices = sorted(
        selected_eval_indices[
            DEV_SIZE:
        ]
    )


    # ========================================================
    # Build chunks
    # ========================================================

    dev_data = build_chunk(
        test,
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


    dev_ids = list(
        dev_data.keys()
    )

    rule_pool_ids = list(
        rule_pool_data.keys()
    )

    test_ids = list(
        test_data.keys()
    )


    # ========================================================
    # Quality10
    # ========================================================
    #
    # Quality10 MUST be a subset of Test50.
    # Select BEFORE seeing any model output.
    # ========================================================

    quality_rng = random.Random(
        SEED + 2
    )

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

    assert (
        len(rule_pool_ids)
        == RULE_POOL_SIZE
    )

    assert (
        len(test_ids)
        == TEST_SIZE
    )

    assert (
        len(quality_ids)
        == QUALITY_SIZE
    )


    # Dev/Test must never overlap.
    assert set(
        dev_ids
    ).isdisjoint(
        test_ids
    )


    # Quality10 must be Test50 subset.
    assert set(
        quality_ids
    ).issubset(
        test_ids
    )


    # Train/Test official splits
    # should be disjoint as well.
    assert set(
        rule_pool_ids
    ).isdisjoint(
        dev_ids
    )

    assert set(
        rule_pool_ids
    ).isdisjoint(
        test_ids
    )


    # ========================================================
    # Validate AutoGEO format
    # ========================================================

    validate_dataset(
        "Dev20",
        dev_data,
    )

    validate_dataset(
        "RulePool100",
        rule_pool_data,
    )

    validate_dataset(
        "Test50",
        test_data,
    )


    # ========================================================
    # Verify keypoints
    # ========================================================

    verify_keypoints(
        dev_ids
    )

    verify_keypoints(
        test_ids
    )

    verify_keypoints(
        quality_ids
    )


    # ========================================================
    # Save IDs
    # ========================================================

    save_json(
        SPLIT_DIR
        / "researchy_dev20_keypoint_v2_ids.json",
        dev_ids,
    )

    save_json(
        SPLIT_DIR
        / "researchy_rulepool100_keypoint_v2_ids.json",
        rule_pool_ids,
    )

    save_json(
        SPLIT_DIR
        / "researchy_test50_keypoint_v2_ids.json",
        test_ids,
    )

    save_json(
        SPLIT_DIR
        / "researchy_quality10_keypoint_v2_ids.json",
        quality_ids,
    )


    # ========================================================
    # Save frozen datasets
    # ========================================================

    save_json(
        DEV_OUTPUT_DIR
        / "datachunk_0.json",
        dev_data,
    )

    save_json(
        RULE_OUTPUT_DIR
        / "datachunk_0.json",
        rule_pool_data,
    )

    save_json(
        TEST_OUTPUT_DIR
        / "datachunk_0.json",
        test_data,
    )


    # ========================================================
    # Metadata + hashes
    # ========================================================

    metadata = {

        "split_version":
            SPLIT_VERSION,

        "dataset_repo":
            DATASET_REPO,

        "dataset_config":
            DATASET_CONFIG,

        "seed":
            SEED,

        "sizes": {

            "dev":
                DEV_SIZE,

            "rule_pool":
                RULE_POOL_SIZE,

            "test":
                TEST_SIZE,

            "quality":
                QUALITY_SIZE,
        },

        "source_splits": {

            "dev":
                (
                    "main/test "
                    "(development subset)"
                ),

            "rule_pool":
                "main/train",

            "test":
                (
                    "main/test "
                    "(held-out final subset)"
                ),

            "quality":
                "subset of Test50",
        },

        "keypoint_policy": {

            "dev_requires_keypoints":
                True,

            "test_requires_keypoints":
                True,

            "quality_requires_keypoints":
                True,

            "rule_pool_requires_keypoints":
                False,
        },

        "official_keypoint_coverage": {

            "train_total":
                len(train),

            "train_with_keypoints":
                len(
                    train_keypoint_indices
                ),

            "test_total":
                len(test),

            "test_with_keypoints":
                len(
                    test_keypoint_indices
                ),
        },

        "hashes": {

            "dev20_sha256":
                sha256_json(
                    dev_data
                ),

            "rulepool100_sha256":
                sha256_json(
                    rule_pool_data
                ),

            "test50_sha256":
                sha256_json(
                    test_data
                ),

            "dev20_keypoints_sha256":
                sha256_keypoints(
                    dev_ids
                ),

            "test50_keypoints_sha256":
                sha256_keypoints(
                    test_ids
                ),

            "quality10_keypoints_sha256":
                sha256_keypoints(
                    quality_ids
                ),
        },
    }


    save_json(
        CONFIG_DIR
        / "researchy_keypoint_v2_metadata.json",
        metadata,
    )


    # ========================================================
    # Final summary
    # ========================================================

    print()
    print("=" * 80)
    print(
        "Researchy-GEO keypoint_v2 "
        "split successfully frozen"
    )
    print("=" * 80)

    print()
    print(
        "RulePool100 :",
        len(rule_pool_ids),
        "(main/train)"
    )

    print(
        "Dev20       :",
        len(dev_ids),
        "(main/test)"
    )

    print(
        "Test50      :",
        len(test_ids),
        "(main/test)"
    )

    print(
        "Quality10   :",
        len(quality_ids),
        "(subset of Test50)"
    )


    print()
    print(
        "Dev/Test overlap:",
        len(
            set(dev_ids)
            & set(test_ids)
        )
    )

    print(
        "Rule/Dev overlap:",
        len(
            set(rule_pool_ids)
            & set(dev_ids)
        )
    )

    print(
        "Rule/Test overlap:",
        len(
            set(rule_pool_ids)
            & set(test_ids)
        )
    )

    print(
        "Quality subset of Test:",
        set(
            quality_ids
        ).issubset(
            test_ids
        )
    )


    print()
    print(
        "Dev keypoint coverage:",
        sum(
            has_valid_keypoints(qid)
            for qid in dev_ids
        ),
        "/",
        DEV_SIZE,
    )

    print(
        "Test keypoint coverage:",
        sum(
            has_valid_keypoints(qid)
            for qid in test_ids
        ),
        "/",
        TEST_SIZE,
    )

    print(
        "Quality keypoint coverage:",
        sum(
            has_valid_keypoints(qid)
            for qid in quality_ids
        ),
        "/",
        QUALITY_SIZE,
    )


    print()
    print(
        "Frozen Dev20 SHA256:"
    )

    print(
        metadata[
            "hashes"
        ][
            "dev20_sha256"
        ]
    )


    print()
    print(
        "Frozen Test50 SHA256:"
    )

    print(
        metadata[
            "hashes"
        ][
            "test50_sha256"
        ]
    )


    print()
    print(
        "Test50 keypoints SHA256:"
    )

    print(
        metadata[
            "hashes"
        ][
            "test50_keypoints_sha256"
        ]
    )


if __name__ == "__main__":
    main()