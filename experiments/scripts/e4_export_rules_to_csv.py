import csv
import json
from pathlib import Path


RULE_FILES = {
    "researchy_geo": Path(
        "experiments/artifacts/E4/rules/researchy_geo/merged_rules.json"
    ),
    "geo_bench": Path(
        "experiments/artifacts/E4/rules/geo_bench/merged_rules.json"
    ),
    "ecommerce": Path(
        "experiments/artifacts/E4/rules/ecommerce/merged_rules.json"
    ),
}

OUTPUT = Path(
    "experiments/artifacts/E4/annotations/rule_annotation.csv"
)

OUTPUT.parent.mkdir(parents=True, exist_ok=True)

rows = []

for dataset, path in RULE_FILES.items():

    with open(path, encoding="utf-8") as f:
        data = json.load(f)

    rules = data["filtered_rules"]

    print(
        f"{dataset}: {len(rules)} filtered rules"
    )

    for i, rule in enumerate(rules, 1):

        rows.append({
            "dataset": dataset,
            "rule_id": f"{dataset}_{i:02d}",
            "rule": rule,
            "keyword_1": "",
            "keyword_2": "",
            "notes": "",
        })


with open(
    OUTPUT,
    "w",
    encoding="utf-8-sig",
    newline=""
) as f:

    writer = csv.DictWriter(
        f,
        fieldnames=[
            "dataset",
            "rule_id",
            "rule",
            "keyword_1",
            "keyword_2",
            "notes",
        ],
    )

    writer.writeheader()
    writer.writerows(rows)


print()
print("Total rules:", len(rows))
print("Saved:", OUTPUT)
