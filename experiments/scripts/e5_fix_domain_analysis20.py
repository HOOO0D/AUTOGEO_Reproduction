import hashlib
import json
import random
from pathlib import Path


SEED = 20260922
N = 20


DATASETS = {
    "geo_bench": {
        "name": "GEO-Bench",
        "src": Path("data/GEO-Bench/test"),
        "dst": Path(
            "experiments/artifacts/E5/data/"
            "geo_bench_analysis20"
        ),
    },

    "ecommerce": {
        "name": "E-commerce",
        "src": Path("data/E-commerce/test"),
        "dst": Path(
            "experiments/artifacts/E5/data/"
            "ecommerce_analysis20"
        ),
    },
}


def chunk_index(path: Path) -> int:
    """
    datachunk_0.json -> 0
    datachunk_10.json -> 10
    """
    return int(
        path.stem.replace(
            "datachunk_",
            "",
        )
    )


def sha256(path: Path) -> str:
    h = hashlib.sha256()

    with open(path, "rb") as f:
        for block in iter(
            lambda: f.read(1024 * 1024),
            b"",
        ):
            h.update(block)

    return h.hexdigest()


def clean_item(item: dict) -> dict:
    """
    Remove any previous experimental outputs.
    Keep only original dataset fields.
    """
    return {
        k: v
        for k, v in item.items()
        if not (
            k.endswith("_text")
            or k.endswith("_response")
            or k.endswith("_geo_score")
            or k.endswith("_geu_score")
        )
    }


def validate(qid: str, item: dict):

    required = [
        "query",
        "text_list",
        "target_id",
    ]

    missing = [
        k for k in required
        if k not in item
    ]

    if missing:
        raise ValueError(
            f"{qid}: missing {missing}"
        )

    if not isinstance(
        item["text_list"],
        list,
    ):
        raise TypeError(
            f"{qid}: text_list is not list"
        )

    if len(item["text_list"]) == 0:
        raise ValueError(
            f"{qid}: empty text_list"
        )

    target_id = item["target_id"]

    if not isinstance(target_id, int):
        raise TypeError(
            f"{qid}: target_id is not int"
        )

    if not (
        0 <= target_id
        < len(item["text_list"])
    ):
        raise ValueError(
            f"{qid}: invalid target_id={target_id}, "
            f"n_docs={len(item['text_list'])}"
        )


def freeze(short_name: str, cfg: dict):

    src = cfg["src"]
    dst = cfg["dst"]

    print()
    print("=" * 100)
    print(cfg["name"])
    print("=" * 100)

    if not src.exists():
        raise FileNotFoundError(
            f"Test directory does not exist: {src}"
        )

    chunk_files = sorted(
        src.glob("datachunk_*.json"),
        key=chunk_index,
    )

    if not chunk_files:
        raise RuntimeError(
            f"No datachunk_*.json found in {src}"
        )

    records = []

    for path in chunk_files:

        with open(
            path,
            encoding="utf-8",
        ) as f:
            chunk = json.load(f)

        if not isinstance(chunk, dict):
            raise TypeError(
                f"{path} is not dict-of-records"
            )

        for qid, item in chunk.items():
            records.append(
                (
                    str(qid),
                    item,
                )
            )

    print(
        "Source chunks:",
        len(chunk_files),
    )

    print(
        "Source samples:",
        len(records),
    )

    if len(records) < N:
        raise RuntimeError(
            f"Only {len(records)} samples; "
            f"cannot sample {N}"
        )

    # ---------------------------------------------------------
    # Deterministic random sample
    # ---------------------------------------------------------

    rng = random.Random(SEED)

    sampled_indices = rng.sample(
        range(len(records)),
        N,
    )

    # Sorting affects only storage order,
    # not which examples were sampled.
    sampled_indices = sorted(
        sampled_indices
    )

    frozen = {}
    sampled_qids = []

    for index in sampled_indices:

        qid, item = records[index]

        item = clean_item(item)

        validate(
            qid,
            item,
        )

        frozen[qid] = item
        sampled_qids.append(qid)

    # ---------------------------------------------------------
    # Save
    # ---------------------------------------------------------

    dst.mkdir(
        parents=True,
        exist_ok=True,
    )

    data_path = (
        dst / "datachunk_0.json"
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

    digest = sha256(data_path)

    manifest = {
        "dataset": cfg["name"],
        "source_dir": str(src),
        "split": "test",

        "sampling":
            "random_without_replacement",

        "seed": SEED,
        "sample_size": N,

        "source_sample_count":
            len(records),

        "sampled_indices":
            sampled_indices,

        "query_ids":
            sampled_qids,

        "sha256":
            digest,
    }

    with open(
        dst / "manifest.json",
        "w",
        encoding="utf-8",
    ) as f:

        json.dump(
            manifest,
            f,
            ensure_ascii=False,
            indent=2,
        )

    # ---------------------------------------------------------
    # Final validation
    # ---------------------------------------------------------

    with open(
        data_path,
        encoding="utf-8",
    ) as f:
        check = json.load(f)

    if len(check) != N:
        raise RuntimeError(
            f"Expected {N}, got {len(check)}"
        )

    print(
        "Frozen samples:",
        len(check),
    )

    print(
        "Seed:",
        SEED,
    )

    print(
        "Indices:",
        sampled_indices,
    )

    print(
        "QIDs:",
        sampled_qids,
    )

    print(
        "SHA256:",
        digest,
    )

    print(
        "Saved:",
        data_path,
    )


for short_name, cfg in DATASETS.items():
    freeze(
        short_name,
        cfg,
    )


print()
print("=" * 100)
print("ALL E5 DOMAIN ANALYSIS SETS FIXED")
print("=" * 100)
