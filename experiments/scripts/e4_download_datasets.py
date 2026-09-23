import json
from pathlib import Path

from datasets import load_dataset


DATASETS = {
    "researchy_geo": "cx-cmu/Researchy-GEO",
    "geo_bench": "cx-cmu/GEO-Bench",
    "ecommerce": "cx-cmu/E-commerce",
}

OUT = Path(
    "experiments/artifacts/E4/preference_pairs"
)

OUT.mkdir(
    parents=True,
    exist_ok=True
)


for short_name, repo_id in DATASETS.items():

    print("=" * 80)
    print(repo_id)
    print("=" * 80)

    ds = load_dataset(
        repo_id,
        name="rule_candidate",
        split="train",
    )

    print(
        "Total official candidates:",
        len(ds)
    )

    # Freeze exactly the first 100 samples.
    n = min(100, len(ds))
    subset = ds.select(range(n))

    data = {}

    for row in subset:

        # Published dataset contains query_id.
        qid = str(row["query_id"])

        item = {
            k: v
            for k, v in row.items()
            if k != "query_id"
        }

        data[qid] = item

    output = (
        OUT /
        f"{short_name}_rulepool100.json"
    )

    with open(
        output,
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(
            data,
            f,
            ensure_ascii=False,
            indent=2,
        )

    print(
        "Frozen:",
        len(data)
    )

    print(
        "Saved:",
        output
    )