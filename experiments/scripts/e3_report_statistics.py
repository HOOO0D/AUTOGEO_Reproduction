import csv
import json
from pathlib import Path


# ============================================================
# Paths
# ============================================================

BASE = Path("experiments/artifacts/E3/results")

KEYWORD_SETS_PATH = BASE / "keyword_sets.json"
JACCARD_PATH = BASE / "jaccard.json"
OVERLAP_PATH = BASE / "common_shared_unique.json"
ANNOTATIONS_PATH = BASE / "rule_annotations.json"

SUMMARY_JSON = BASE / "e3_statistics.json"
ENGINE_CSV = BASE / "e3_engine_statistics.csv"
PAIRWISE_CSV = BASE / "e3_pairwise_statistics.csv"
REPORT_MD = BASE / "e3_statistics.md"


# ============================================================
# Paper reference values
# ============================================================

PAPER_JACCARD = {
    "gemini_vs_gpt": 78.95,
    "gemini_vs_claude": 84.21,
    "gpt_vs_claude": 84.21,
}

CANONICAL_KEYWORD_COUNT = 20


# ============================================================
# Load
# ============================================================

def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


keyword_sets = load_json(KEYWORD_SETS_PATH)
jaccard_results = load_json(JACCARD_PATH)
overlap = load_json(OVERLAP_PATH)
annotations = load_json(ANNOTATIONS_PATH)


# ============================================================
# Basic counts
# ============================================================

engines = ["gemini", "gpt", "claude"]

rule_counts = {
    engine: len(annotations[engine])
    for engine in engines
}

keyword_counts = {
    engine: len(keyword_sets[engine])
    for engine in engines
}

keyword_coverage = {
    engine:
        keyword_counts[engine]
        / CANONICAL_KEYWORD_COUNT
        * 100
    for engine in engines
}


# ============================================================
# Global union
# ============================================================

sets = {
    engine: set(keyword_sets[engine])
    for engine in engines
}

global_union = (
    sets["gemini"]
    | sets["gpt"]
    | sets["claude"]
)

global_intersection = (
    sets["gemini"]
    & sets["gpt"]
    & sets["claude"]
)

union_count = len(global_union)
common_count = len(global_intersection)

taxonomy_coverage = (
    union_count
    / CANONICAL_KEYWORD_COUNT
    * 100
)


# ============================================================
# Common / Shared / Unique counts
# ============================================================

shared_counts = {
    pair: len(keywords)
    for pair, keywords
    in overlap["shared"].items()
}

unique_counts = {
    engine: len(keywords)
    for engine, keywords
    in overlap["unique"].items()
}

shared_only_count = sum(
    shared_counts.values()
)

unique_only_count = sum(
    unique_counts.values()
)

# Keywords appearing in at least two engines
multi_engine_count = (
    common_count
    + shared_only_count
)

multi_engine_percent = (
    multi_engine_count
    / union_count
    * 100
    if union_count else 0
)

unique_percent = (
    unique_only_count
    / union_count
    * 100
    if union_count else 0
)


# ============================================================
# Pairwise statistics
# ============================================================

pairwise_stats = {}

for pair, result in jaccard_results.items():

    reproduced = result["overlap_percent"]
    paper = PAPER_JACCARD.get(pair)

    pairwise_stats[pair] = {
        "intersection":
            result["intersection_count"],

        "union":
            result["union_count"],

        "jaccard_percent":
            reproduced,

        "paper_percent":
            paper,

        "difference_pp":
            reproduced - paper
            if paper is not None
            else None,
    }


mean_jaccard = (
    sum(
        x["jaccard_percent"]
        for x in pairwise_stats.values()
    )
    / len(pairwise_stats)
)

paper_mean_jaccard = (
    sum(PAPER_JACCARD.values())
    / len(PAPER_JACCARD)
)

mean_gap = (
    mean_jaccard
    - paper_mean_jaccard
)


# ============================================================
# Keyword frequency
# Diagnostic only — not used for Jaccard
# ============================================================

keyword_frequency = {
    engine: {}
    for engine in engines
}

for engine in engines:

    for item in annotations[engine]:

        for keyword in item["keywords"]:

            keyword_frequency[engine][keyword] = (
                keyword_frequency[engine]
                .get(keyword, 0)
                + 1
            )


# ============================================================
# Console report
# ============================================================

print("=" * 80)
print("E3 STATISTICAL SUMMARY")
print("=" * 80)

print()
print("RULE COUNTS")

for engine in engines:
    print(
        f"{engine.upper():8s}: "
        f"{rule_counts[engine]}"
    )

print(
    f"TOTAL   : {sum(rule_counts.values())}"
)


print()
print("=" * 80)
print("KEYWORD COVERAGE")
print("=" * 80)

for engine in engines:

    print(
        f"{engine.upper():8s}: "
        f"{keyword_counts[engine]:2d}"
        f"/{CANONICAL_KEYWORD_COUNT} "
        f"({keyword_coverage[engine]:.2f}%)"
    )

print()
print(
    f"Observed union: "
    f"{union_count}/"
    f"{CANONICAL_KEYWORD_COUNT} "
    f"({taxonomy_coverage:.2f}%)"
)

print(
    f"Common to all three: "
    f"{common_count}"
)

print(
    f"Shared by exactly two: "
    f"{shared_only_count}"
)

print(
    f"Unique to one engine: "
    f"{unique_only_count}"
)

print(
    f"Appearing in >=2 engines: "
    f"{multi_engine_count}/{union_count} "
    f"({multi_engine_percent:.2f}%)"
)

print(
    f"Unique-only share: "
    f"{unique_only_count}/{union_count} "
    f"({unique_percent:.2f}%)"
)


print()
print("=" * 80)
print("PAIRWISE JACCARD")
print("=" * 80)

for pair, result in pairwise_stats.items():

    print()
    print(pair.upper())

    print(
        f"  Intersection : "
        f"{result['intersection']}"
    )

    print(
        f"  Union        : "
        f"{result['union']}"
    )

    print(
        f"  Reproduction : "
        f"{result['jaccard_percent']:.2f}%"
    )

    if result["paper_percent"] is not None:

        print(
            f"  Paper        : "
            f"{result['paper_percent']:.2f}%"
        )

        print(
            f"  Difference   : "
            f"{result['difference_pp']:+.2f} pp"
        )


print()
print(
    f"Mean reproduced Jaccard: "
    f"{mean_jaccard:.2f}%"
)

print(
    f"Mean paper Jaccard     : "
    f"{paper_mean_jaccard:.2f}%"
)

print(
    f"Mean difference        : "
    f"{mean_gap:+.2f} pp"
)


print()
print("=" * 80)
print("COMMON / SHARED / UNIQUE COUNTS")
print("=" * 80)

print(
    f"Common: "
    f"{common_count}"
)

for pair, count in shared_counts.items():

    print(
        f"Shared {pair}: "
        f"{count}"
    )

for engine, count in unique_counts.items():

    print(
        f"Unique {engine}: "
        f"{count}"
    )


# ============================================================
# Save engine statistics CSV
# ============================================================

with open(
    ENGINE_CSV,
    "w",
    encoding="utf-8-sig",
    newline=""
) as f:

    writer = csv.writer(f)

    writer.writerow([
        "engine",
        "rule_count",
        "keyword_count",
        "canonical_keyword_count",
        "keyword_coverage_percent",
    ])

    for engine in engines:

        writer.writerow([
            engine,
            rule_counts[engine],
            keyword_counts[engine],
            CANONICAL_KEYWORD_COUNT,
            f"{keyword_coverage[engine]:.2f}",
        ])


# ============================================================
# Save pairwise CSV
# ============================================================

with open(
    PAIRWISE_CSV,
    "w",
    encoding="utf-8-sig",
    newline=""
) as f:

    writer = csv.writer(f)

    writer.writerow([
        "pair",
        "intersection",
        "union",
        "reproduction_jaccard_percent",
        "paper_jaccard_percent",
        "difference_pp",
    ])

    for pair, result in pairwise_stats.items():

        writer.writerow([
            pair,
            result["intersection"],
            result["union"],
            f"{result['jaccard_percent']:.2f}",
            f"{result['paper_percent']:.2f}",
            f"{result['difference_pp']:+.2f}",
        ])


# ============================================================
# Save JSON
# ============================================================

summary = {
    "rule_counts": rule_counts,

    "keyword_counts": keyword_counts,

    "keyword_coverage_percent":
        keyword_coverage,

    "global": {
        "observed_union_count":
            union_count,

        "taxonomy_coverage_percent":
            taxonomy_coverage,

        "common_all_count":
            common_count,

        "shared_exactly_two_count":
            shared_only_count,

        "unique_one_engine_count":
            unique_only_count,

        "multi_engine_keyword_count":
            multi_engine_count,

        "multi_engine_percent":
            multi_engine_percent,

        "unique_percent":
            unique_percent,
    },

    "pairwise":
        pairwise_stats,

    "mean_jaccard_percent":
        mean_jaccard,

    "paper_mean_jaccard_percent":
        paper_mean_jaccard,

    "mean_difference_pp":
        mean_gap,

    "common_shared_unique":
        overlap,

    "keyword_frequency":
        keyword_frequency,
}

with open(
    SUMMARY_JSON,
    "w",
    encoding="utf-8"
) as f:

    json.dump(
        summary,
        f,
        ensure_ascii=False,
        indent=4
    )


# ============================================================
# Markdown report
# ============================================================

with open(
    REPORT_MD,
    "w",
    encoding="utf-8"
) as f:

    f.write(
        "# E3 Engine-Specific "
        "Preference Statistics\n\n"
    )

    f.write("## Rule and keyword counts\n\n")

    f.write(
        "| Engine | Rules | Keywords | "
        "Keyword coverage |\n"
    )

    f.write(
        "|---|---:|---:|---:|\n"
    )

    for engine in engines:

        f.write(
            f"| {engine.capitalize()} "
            f"| {rule_counts[engine]} "
            f"| {keyword_counts[engine]} "
            f"| {keyword_coverage[engine]:.2f}% |\n"
        )

    f.write("\n## Pairwise Jaccard\n\n")

    f.write(
        "| Pair | Intersection | Union | "
        "Reproduction | Paper | Difference |\n"
    )

    f.write(
        "|---|---:|---:|---:|---:|---:|\n"
    )

    for pair, result in pairwise_stats.items():

        f.write(
            f"| {pair} "
            f"| {result['intersection']} "
            f"| {result['union']} "
            f"| {result['jaccard_percent']:.2f}% "
            f"| {result['paper_percent']:.2f}% "
            f"| {result['difference_pp']:+.2f} pp |\n"
        )

    f.write("\n## Overall overlap\n\n")

    f.write(
        f"- Observed keyword union: "
        f"{union_count}/"
        f"{CANONICAL_KEYWORD_COUNT} "
        f"({taxonomy_coverage:.2f}%)\n"
    )

    f.write(
        f"- Common to all three: "
        f"{common_count}\n"
    )

    f.write(
        f"- Shared by exactly two: "
        f"{shared_only_count}\n"
    )

    f.write(
        f"- Unique to one engine: "
        f"{unique_only_count}\n"
    )

    f.write(
        f"- Keywords appearing in >=2 engines: "
        f"{multi_engine_count}/{union_count} "
        f"({multi_engine_percent:.2f}%)\n"
    )

    f.write(
        f"- Mean pairwise Jaccard: "
        f"{mean_jaccard:.2f}%\n"
    )

    f.write(
        f"- Paper mean pairwise Jaccard: "
        f"{paper_mean_jaccard:.2f}%\n"
    )

    f.write(
        f"- Mean difference: "
        f"{mean_gap:+.2f} pp\n"
    )


print()
print("=" * 80)
print("SAVED")
print("=" * 80)

print(SUMMARY_JSON)
print(ENGINE_CSV)
print(PAIRWISE_CSV)
print(REPORT_MD)