import hashlib
import json
import random
from pathlib import Path

from datasets import load_dataset


# ============================================================
# Configuration
# ============================================================

SEED = 20260922
N = 20

DATASETS = {
    "geo_bench": {
        "repo_id": "cx-cmu/GEO-Bench",
        "config": "main",
        "split": "test",
        "output_dir": Path(
            "experiments/frozen/e5/geo_bench_analysis20"
        ),
    },

    "ecommerce": {
        "repo_id": "cx-cmu/E-commerce",
        "config": "main",
        "split": "test",
        "output_dir": Path(
            "experiments/frozen/e5/ecommerce_analysis20"
        ),
    },
}


# ============================================================
# Helpers
# ============================================================

def sha256_file(path: Path) -> str:

    h = hashlib.sha256()

    with open(path, "rb") as f:
        while True:
            block = f.read(1024 * 1024)

            if not block:
                break

            h.update(block)

    return h.hexdigest()


def validate_record(qid, item):

    required = [
        "query",
        "text_list",
        "target_id",
    ]

    missing = [
        key
        for key in required
        if key not in item
    ]

    if missing:
        raise ValueError(
            f"{qid}: missing fields {missing}"
        )

    text_list = item["text_list"]
    target_id = item["target_id"]

    if not isinstance(text_list, list):
        raise TypeError(
            f"{qid}: text_list is not a list"
        )

    if not text_list:
        raise ValueError(
            f"{qid}: empty text_list"
        )

    if not isinstance(target_id, int):
        raise TypeError(
            f"{qid}: target_id is not int"
        )

    if not (
        0 <= target_id < len(text_list)
    ):
        raise ValueError(
            f"{qid}: invalid target_id={target_id}, "
            f"n_docs={len(text_list)}"
        )


# ============================================================
# Freeze one dataset
# ============================================================

def freeze_dataset(name, cfg):

    print()
    print("=" * 90)
    print(f"Preparing {name}")
    print("=" * 90)

    repo_id = cfg["repo_id"]
    config = cfg["config"]
    split = cfg["split"]
    output_dir = cfg["output_dir"]

    # --------------------------------------------------------
    # Download official main/test
    # --------------------------------------------------------

    ds = load_dataset(
        repo_id,
        name=config,
        split=split,
    )

    print(
        f"Official test size: {len(ds)}"
    )

    if len(ds) < N:
        raise ValueError(
            f"{name}: only {len(ds)} samples, "
            f"cannot sample {N}"
        )

    # --------------------------------------------------------
    # Deterministic random sample
    # --------------------------------------------------------

    rng = random.Random(SEED)

    sampled_indices = rng.sample(
        range(len(ds)),
        N,
    )

    # Sort after sampling only to make the stored file stable/readable.
    # This does NOT change which samples were randomly selected.
    sampled_indices = sorted(
        sampled_indices
    )

    frozen = {}
    query_ids = []

    for index in sampled_indices:

        row = dict(ds[index])

        if "query_id" not in row:
            raise KeyError(
                f"{name}: dataset row has no query_id"
            )

        qid = str(
            row.pop("query_id")
        )

        validate_record(
            qid,
            row,
        )

        frozen[qid] = row
        query_ids.append(qid)

    # --------------------------------------------------------
    # Write data
    # --------------------------------------------------------

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    data_path = (
        output_dir
        / "datachunk_0.json"
    )

    with open(
        data_path,
        "w",
        encoding="utf-8",
    ) as f:

        json.dump(
            frozen,
            f,
            ensure_ascii=False,
            indent=2,
        )

    # --------------------------------------------------------
    # Freeze manifest
    # --------------------------------------------------------

    manifest = {
        "dataset": name,
        "repo_id": repo_id,
        "config": config,
        "split": split,

        "sampling": "random_without_replacement",
        "seed": SEED,
        "sample_size": N,

        "source_split_size": len(ds),

        "sampled_indices": sampled_indices,
        "query_ids": query_ids,

        "data_file": str(data_path),
        "sha256": sha256_file(data_path),
    }

    manifest_path = (
        output_dir
        / "manifest.json"
    )

    with open(
        manifest_path,
        "w",
        encoding="utf-8",
    ) as f:

        json.dump(
            manifest,
            f,
            ensure_ascii=False,
            indent=2,
        )

    # --------------------------------------------------------
    # Summary
    # --------------------------------------------------------

    print(
        f"Frozen samples: {len(frozen)}"
    )

    print(
        f"Seed: {SEED}"
    )

    print(
        "Sampled indices:",
        sampled_indices,
    )

    print(
        "Query IDs:",
        query_ids,
    )

    print(
        "Saved:",
        data_path,
    )

    print(
        "Manifest:",
        manifest_path,
    )

    print(
        "SHA256:",
        manifest["sha256"],
    )


# ============================================================
# Main
# ============================================================

for name, cfg in DATASETS.items():
    freeze_dataset(
        name,
        cfg,
    )
