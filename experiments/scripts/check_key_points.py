from pathlib import Path
from datasets import load_dataset


DATASET_REPO = "cx-cmu/Researchy-GEO"
DATASET_CONFIG = "main"

KEYPOINT_DIR = Path(
    "data/Researchy-GEO/key_point"
)


def get_keypoint_ids():
    ids = set()

    for path in KEYPOINT_DIR.glob(
        "*_aggregated.json"
    ):
        qid = path.name.removesuffix(
            "_aggregated.json"
        )
        ids.add(qid)

    return ids


def get_split_ids(split):
    return {
        str(row["query_id"])
        for row in split
    }


def main():

    keypoint_ids = get_keypoint_ids()

    print(
        "Aggregated keypoint queries:",
        len(keypoint_ids)
    )

    train = load_dataset(
        DATASET_REPO,
        DATASET_CONFIG,
        split="train",
    )

    test = load_dataset(
        DATASET_REPO,
        DATASET_CONFIG,
        split="test",
    )

    train_ids = get_split_ids(train)
    test_ids = get_split_ids(test)

    train_with_kp = (
        train_ids & keypoint_ids
    )

    test_with_kp = (
        test_ids & keypoint_ids
    )

    print()
    print(
        "Train total:",
        len(train_ids)
    )
    print(
        "Train with keypoints:",
        len(train_with_kp)
    )

    print()
    print(
        "Test total:",
        len(test_ids)
    )
    print(
        "Test with keypoints:",
        len(test_with_kp)
    )

    print()
    print(
        "Enough for train Dev20:",
        len(train_with_kp) >= 20
    )

    print(
        "Enough for Test50:",
        len(test_with_kp) >= 50
    )

    print(
        "Enough for Dev20 + Test50 "
        "both from test:",
        len(test_with_kp) >= 70
    )


if __name__ == "__main__":
    main()