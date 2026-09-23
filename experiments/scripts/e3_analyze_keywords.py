import csv
import json
from pathlib import Path
from itertools import combinations


# ============================================================
# Configuration
# ============================================================

INPUT_CSV = Path(
    "experiments/artifacts/E3/annotations/rule_annotation.csv"
)

OUTPUT_DIR = Path(
    "experiments/artifacts/E3/results"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)


ENGINES = [
    "gemini",
    "gpt",
    "claude",
]


# Strictly use the 20 canonical keywords used for E3 annotation
CANONICAL_KEYWORDS = {
    "Source Citation",
    "Comprehensive",
    "Factual Accuracy",
    "Topic Focus",
    "Neutral Tone",
    "Balanced View",
    "Self-Contained",
    "Actionable",
    "In-depth",
    "Conclusion First",
    "Logical Structure",
    "Specific Evidence",
    "Clear Language",
    "Up-to-date",
    "Cohesive Flow",
    "Accessibility",
    "Conciseness",
    "Writing Quality",
    "Informational Purpose",
    "Single Idea",
}


# ============================================================
# Load annotations
# ============================================================

def load_annotations(path):
    rows = []

    with open(
        path,
        "r",
        encoding="utf-8-sig",
        newline=""
    ) as f:

        reader = csv.DictReader(f)

        required_columns = {
            "engine",
            "rule_id",
            "rule",
            "keyword_1",
            "keyword_2",
        }

        missing = (
            required_columns
            - set(reader.fieldnames or [])
        )

        if missing:
            raise ValueError(
                f"Missing columns: {sorted(missing)}"
            )

        for row in reader:

            # Ignore completely empty lines
            if not any(
                str(v).strip()
                for v in row.values()
                if v is not None
            ):
                continue

            # Normalize whitespace
            for key in row:
                if row[key] is not None:
                    row[key] = row[key].strip()

            rows.append(row)

    return rows


# ============================================================
# Validate annotations
# ============================================================

def validate_annotations(rows):

    errors = []
    warnings = []

    seen_rule_ids = set()

    for i, row in enumerate(
        rows,
        start=2,  # CSV header = line 1
    ):

        engine = row["engine"]
        rule_id = row["rule_id"]
        rule = row["rule"]

        k1 = row["keyword_1"]
        k2 = row["keyword_2"]

        # ----------------------------------------------------
        # Engine validation
        # ----------------------------------------------------

        if engine not in ENGINES:
            errors.append(
                f"Line {i}: invalid engine '{engine}'"
            )

        # ----------------------------------------------------
        # Rule ID validation
        # ----------------------------------------------------

        if not rule_id:
            errors.append(
                f"Line {i}: missing rule_id"
            )

        elif rule_id in seen_rule_ids:
            errors.append(
                f"Line {i}: duplicated rule_id "
                f"'{rule_id}'"
            )

        seen_rule_ids.add(rule_id)

        # ----------------------------------------------------
        # Rule text
        # ----------------------------------------------------

        if not rule:
            errors.append(
                f"Line {i}: empty rule text"
            )

        # ----------------------------------------------------
        # Each rule must have keyword_1
        # ----------------------------------------------------

        if not k1:
            errors.append(
                f"Line {i}: {rule_id} "
                f"has no keyword_1"
            )

        # ----------------------------------------------------
        # Keyword must belong to canonical taxonomy
        # ----------------------------------------------------

        for field, keyword in [
            ("keyword_1", k1),
            ("keyword_2", k2),
        ]:

            if (
                keyword
                and keyword not in CANONICAL_KEYWORDS
            ):
                errors.append(
                    f"Line {i}: {rule_id} "
                    f"has invalid {field}: "
                    f"'{keyword}'"
                )

        # ----------------------------------------------------
        # Same keyword shouldn't appear twice
        # ----------------------------------------------------

        if k1 and k2 and k1 == k2:
            warnings.append(
                f"Line {i}: {rule_id} "
                f"uses the same keyword twice: "
                f"'{k1}'"
            )

    return errors, warnings


# ============================================================
# Build engine keyword sets
# ============================================================

def build_keyword_sets(rows):

    keyword_sets = {
        engine: set()
        for engine in ENGINES
    }

    rule_annotations = {
        engine: []
        for engine in ENGINES
    }

    for row in rows:

        engine = row["engine"]

        keywords = []

        if row["keyword_1"]:
            keywords.append(
                row["keyword_1"]
            )

        if (
            row["keyword_2"]
            and
            row["keyword_2"]
            not in keywords
        ):
            keywords.append(
                row["keyword_2"]
            )

        keyword_sets[engine].update(
            keywords
        )

        rule_annotations[engine].append(
            {
                "rule_id":
                    row["rule_id"],

                "rule":
                    row["rule"],

                "keywords":
                    keywords,
            }
        )

    return (
        keyword_sets,
        rule_annotations
    )


# ============================================================
# Jaccard
# ============================================================

def jaccard(a, b):

    intersection = a & b
    union = a | b

    score = (
        len(intersection)
        / len(union)
        if union else 1.0
    )

    return {
        "intersection_count":
            len(intersection),

        "union_count":
            len(union),

        "jaccard":
            score,

        "overlap_percent":
            score * 100,

        "common_keywords":
            sorted(intersection),

        "a_only":
            sorted(a - b),

        "b_only":
            sorted(b - a),
    }


# ============================================================
# Common / Shared / Unique
# ============================================================

def analyze_overlap(keyword_sets):

    G = keyword_sets["gemini"]
    P = keyword_sets["gpt"]
    C = keyword_sets["claude"]

    common_all = (
        G & P & C
    )

    # Pairwise shared but NOT present in third model
    gemini_gpt_only = (
        (G & P) - C
    )

    gemini_claude_only = (
        (G & C) - P
    )

    gpt_claude_only = (
        (P & C) - G
    )

    # Unique to one engine
    gemini_unique = (
        G - (P | C)
    )

    gpt_unique = (
        P - (G | C)
    )

    claude_unique = (
        C - (G | P)
    )

    return {
        "common_all": sorted(
            common_all
        ),

        "shared": {
            "gemini_gpt":
                sorted(gemini_gpt_only),

            "gemini_claude":
                sorted(gemini_claude_only),

            "gpt_claude":
                sorted(gpt_claude_only),
        },

        "unique": {
            "gemini":
                sorted(gemini_unique),

            "gpt":
                sorted(gpt_unique),

            "claude":
                sorted(claude_unique),
        },
    }


# ============================================================
# Keyword presence matrix
# ============================================================

def save_presence_matrix(
    keyword_sets,
    output_path
):

    with open(
        output_path,
        "w",
        encoding="utf-8-sig",
        newline=""
    ) as f:

        writer = csv.writer(f)

        writer.writerow([
            "keyword",
            "gemini",
            "gpt",
            "claude",
            "count",
            "category",
        ])

        for keyword in sorted(
            CANONICAL_KEYWORDS
        ):

            presence = {
                engine:
                    keyword
                    in keyword_sets[engine]
                for engine in ENGINES
            }

            count = sum(
                presence.values()
            )

            if count == 3:
                category = "Common"

            elif count == 2:
                category = "Shared"

            elif count == 1:
                category = "Unique"

            else:
                category = "Absent"

            writer.writerow([
                keyword,
                "✓" if presence["gemini"] else "",
                "✓" if presence["gpt"] else "",
                "✓" if presence["claude"] else "",
                count,
                category,
            ])


# ============================================================
# Main
# ============================================================

def main():

    rows = load_annotations(
        INPUT_CSV
    )

    print("=" * 80)
    print("E3 ENGINE-SPECIFIC PREFERENCE ANALYSIS")
    print("=" * 80)

    print(
        f"Loaded rules: {len(rows)}"
    )

    # --------------------------------------------------------
    # Validation
    # --------------------------------------------------------

    errors, warnings = (
        validate_annotations(rows)
    )

    if warnings:

        print()
        print("[WARNINGS]")

        for warning in warnings:
            print(" -", warning)

    if errors:

        print()
        print("[ERRORS]")

        for error in errors:
            print(" -", error)

        raise SystemExit(
            "\nAnnotation validation failed."
        )

    print(
        "Annotation validation: PASS"
    )

    # --------------------------------------------------------
    # Build sets
    # --------------------------------------------------------

    (
        keyword_sets,
        rule_annotations
    ) = build_keyword_sets(rows)

    print()
    print("=" * 80)
    print("KEYWORD SETS")
    print("=" * 80)

    for engine in ENGINES:

        keywords = keyword_sets[
            engine
        ]

        print()
        print(
            f"{engine.upper()}: "
            f"{len(keywords)} keywords"
        )

        for keyword in sorted(keywords):
            print(
                "  -",
                keyword
            )

    # --------------------------------------------------------
    # Pairwise Jaccard
    # --------------------------------------------------------

    print()
    print("=" * 80)
    print("PAIRWISE JACCARD")
    print("=" * 80)

    jaccard_results = {}

    for a, b in combinations(
        ENGINES,
        2
    ):

        result = jaccard(
            keyword_sets[a],
            keyword_sets[b],
        )

        key = f"{a}_vs_{b}"

        jaccard_results[key] = result

        print()
        print(
            f"{a.upper()} vs "
            f"{b.upper()}"
        )

        print(
            f"Intersection = "
            f"{result['intersection_count']}"
        )

        print(
            f"Union = "
            f"{result['union_count']}"
        )

        print(
            f"Jaccard = "
            f"{result['overlap_percent']:.2f}%"
        )

    # --------------------------------------------------------
    # Common / Shared / Unique
    # --------------------------------------------------------

    overlap = analyze_overlap(
        keyword_sets
    )

    print()
    print("=" * 80)
    print("COMMON / SHARED / UNIQUE")
    print("=" * 80)

    print()
    print(
        "COMMON TO ALL THREE:"
    )

    for keyword in overlap[
        "common_all"
    ]:
        print(
            "  -",
            keyword
        )

    print()
    print("SHARED BY TWO:")

    for pair, keywords in (
        overlap["shared"].items()
    ):

        print()
        print(
            f"  {pair}:"
        )

        if not keywords:
            print("    (none)")

        else:
            for keyword in keywords:
                print(
                    "   -",
                    keyword
                )

    print()
    print("UNIQUE:")

    for engine, keywords in (
        overlap["unique"].items()
    ):

        print()
        print(
            f"  {engine.upper()}:"
        )

        if not keywords:
            print(
                "    (none)"
            )

        else:
            for keyword in keywords:
                print(
                    "   -",
                    keyword
                )

    # --------------------------------------------------------
    # Save JSON results
    # --------------------------------------------------------

    keyword_sets_json = {
        engine:
            sorted(keywords)
        for engine, keywords
        in keyword_sets.items()
    }

    with open(
        OUTPUT_DIR
        / "keyword_sets.json",
        "w",
        encoding="utf-8"
    ) as f:

        json.dump(
            keyword_sets_json,
            f,
            ensure_ascii=False,
            indent=4
        )

    with open(
        OUTPUT_DIR
        / "jaccard.json",
        "w",
        encoding="utf-8"
    ) as f:

        json.dump(
            jaccard_results,
            f,
            ensure_ascii=False,
            indent=4
        )

    with open(
        OUTPUT_DIR
        / "common_shared_unique.json",
        "w",
        encoding="utf-8"
    ) as f:

        json.dump(
            overlap,
            f,
            ensure_ascii=False,
            indent=4
        )

    with open(
        OUTPUT_DIR
        / "rule_annotations.json",
        "w",
        encoding="utf-8"
    ) as f:

        json.dump(
            rule_annotations,
            f,
            ensure_ascii=False,
            indent=4
        )

    save_presence_matrix(
        keyword_sets,
        OUTPUT_DIR
        / "keyword_presence.csv"
    )

    print()
    print("=" * 80)
    print("OUTPUT FILES")
    print("=" * 80)

    print(
        OUTPUT_DIR
        / "keyword_sets.json"
    )

    print(
        OUTPUT_DIR
        / "jaccard.json"
    )

    print(
        OUTPUT_DIR
        / "common_shared_unique.json"
    )

    print(
        OUTPUT_DIR
        / "rule_annotations.json"
    )

    print(
        OUTPUT_DIR
        / "keyword_presence.csv"
    )


if __name__ == "__main__":
    main()