import csv
import json
from pathlib import Path
from itertools import combinations


INPUT = Path(
    "experiments/artifacts/E4/annotations/rule_annotation.csv"
)

OUTPUT_DIR = Path(
    "experiments/artifacts/E4/results"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)

DATASETS = [
    "researchy_geo",
    "geo_bench",
    "ecommerce",
]


keyword_sets = {
    d: set()
    for d in DATASETS
}


with open(
    INPUT,
    encoding="utf-8-sig",
    newline=""
) as f:

    reader = csv.DictReader(f)

    for row in reader:

        dataset = row["dataset"].strip()

        k1 = row["keyword_1"].strip()
        k2 = row["keyword_2"].strip()

        if not k1:
            raise ValueError(
                f"{row['rule_id']} missing keyword_1"
            )

        keyword_sets[dataset].add(k1)

        if k2:
            keyword_sets[dataset].add(k2)


print("=" * 80)
print("KEYWORD SETS")
print("=" * 80)

for dataset in DATASETS:

    print()
    print(
        dataset.upper(),
        len(keyword_sets[dataset]),
    )

    for k in sorted(
        keyword_sets[dataset]
    ):
        print(" -", k)


print()
print("=" * 80)
print("PAIRWISE JACCARD")
print("=" * 80)


results = {}

for a, b in combinations(
    DATASETS,
    2
):

    A = keyword_sets[a]
    B = keyword_sets[b]

    inter = A & B
    union = A | B

    j = len(inter) / len(union)

    key = f"{a}_vs_{b}"

    results[key] = {
        "intersection_count":
            len(inter),

        "union_count":
            len(union),

        "jaccard":
            j,

        "overlap_percent":
            j * 100,

        "common_keywords":
            sorted(inter),

        f"{a}_only":
            sorted(A - B),

        f"{b}_only":
            sorted(B - A),
    }

    print()
    print(
        f"{a} vs {b}"
    )

    print(
        f"Intersection = "
        f"{len(inter)}"
    )

    print(
        f"Union = "
        f"{len(union)}"
    )

    print(
        f"Jaccard = "
        f"{j * 100:.2f}%"
    )


R = keyword_sets["researchy_geo"]
G = keyword_sets["geo_bench"]
E = keyword_sets["ecommerce"]

common_all = R & G & E

print()
print("=" * 80)
print("COMMON TO ALL THREE")
print("=" * 80)

for k in sorted(common_all):
    print(" -", k)


with open(
    OUTPUT_DIR / "jaccard.json",
    "w",
    encoding="utf-8"
) as f:

    json.dump(
        results,
        f,
        ensure_ascii=False,
        indent=4
    )


with open(
    OUTPUT_DIR / "keyword_sets.json",
    "w",
    encoding="utf-8"
) as f:

    json.dump(
        {
            k: sorted(v)
            for k, v
            in keyword_sets.items()
        },
        f,
        ensure_ascii=False,
        indent=4
    )


print()
print("Saved to:", OUTPUT_DIR)