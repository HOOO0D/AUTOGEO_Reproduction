import csv
import json
from pathlib import Path


# ============================================================
# Configuration
# ============================================================

RULE_FILES = {
    "gemini": Path(
        "experiments/artifacts/E3/rules/gemini/merged_rules.json"
    ),
    "gpt": Path(
        "experiments/artifacts/E3/rules/gpt/merged_rules.json"
    ),
    "claude": Path(
        "experiments/artifacts/E3/rules/claude/merged_rules.json"
    ),
}

OUTPUT_DIR = Path(
    "experiments/artifacts/E3/annotations"
)

COMBINED_OUTPUT = OUTPUT_DIR / "rule_annotation.csv"


# ============================================================
# Load filtered rules
# ============================================================

def load_filtered_rules(path: Path):
    """
    Load filtered_rules from a merged_rules.json file.
    """

    if not path.exists():
        raise FileNotFoundError(
            f"Rule file does not exist: {path}"
        )

    with open(
        path,
        "r",
        encoding="utf-8",
    ) as f:
        data = json.load(f)

    if "filtered_rules" not in data:
        raise KeyError(
            f"'filtered_rules' not found in {path}"
        )

    rules = data["filtered_rules"]

    if not isinstance(rules, list):
        raise TypeError(
            f"'filtered_rules' must be a list in {path}"
        )

    return rules


# ============================================================
# Export
# ============================================================

def main():

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    all_rows = []

    print("=" * 80)
    print("E3 Rule Annotation CSV Export")
    print("=" * 80)

    for engine, path in RULE_FILES.items():

        rules = load_filtered_rules(path)

        print(
            f"{engine.upper():8s}: "
            f"{len(rules)} filtered rules"
        )

        for index, rule in enumerate(
            rules,
            start=1,
        ):

            # Example:
            # gemini_01
            # gpt_01
            # claude_01
            rule_id = (
                f"{engine}_{index:02d}"
            )

            all_rows.append(
                {
                    "engine": engine,
                    "rule_id": rule_id,
                    "rule": rule,

                    # Manually annotate these two columns.
                    "keyword_1": "",
                    "keyword_2": "",

                    # Optional annotation notes.
                    "notes": "",
                }
            )

    # --------------------------------------------------------
    # Write combined CSV
    # utf-8-sig is used so Excel opens Chinese text correctly.
    # --------------------------------------------------------

    with open(
        COMBINED_OUTPUT,
        "w",
        encoding="utf-8-sig",
        newline="",
    ) as f:

        writer = csv.DictWriter(
            f,
            fieldnames=[
                "engine",
                "rule_id",
                "rule",
                "keyword_1",
                "keyword_2",
                "notes",
            ],
        )

        writer.writeheader()
        writer.writerows(all_rows)

    print()
    print("=" * 80)
    print("Export completed")
    print("=" * 80)

    print(
        f"Total rules: {len(all_rows)}"
    )

    print(
        f"Saved to: {COMBINED_OUTPUT}"
    )


if __name__ == "__main__":
    main()