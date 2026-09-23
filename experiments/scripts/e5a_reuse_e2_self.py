import json
import statistics
from pathlib import Path


E5_PATH = Path(
    "experiments/artifacts/E5/data/"
    "researchy_dev20/datachunk_0.json"
)

CASES = {
    "GPT": {
        "self_path": Path(
            "experiments/artifacts/E2/eval_runs/"
            "run1/gpt/strong/datachunk_0.json"
        ),
        "self_key":
            "e2_gpt_strong_geo_score",

        "transfer_key":
            "e5_ge_gpt_from_gemini_geo_score",
    },

    "Claude": {
        "self_path": Path(
            "experiments/artifacts/E2/eval_runs/"
            "run1/claude/strong/datachunk_0.json"
        ),
        "self_key":
            "e2_claude_strong_geo_score",

        "transfer_key":
            "e5_ge_claude_from_gemini_geo_score",
    },
}


with open(E5_PATH, encoding="utf-8") as f:
    e5 = json.load(f)


def get_overall(score):
    return score["wordpos"] * 100


results = {}


for engine, cfg in CASES.items():

    with open(
        cfg["self_path"],
        encoding="utf-8"
    ) as f:
        self_data = json.load(f)

    common = sorted(
        set(self_data)
        & set(e5)
    )

    # --------------------------------------------------------
    # Verify same evaluation samples
    # --------------------------------------------------------

    valid_ids = []

    for qid in common:

        if (
            self_data[qid].get(cfg["self_key"])
            and
            e5[qid].get(cfg["transfer_key"])
        ):
            valid_ids.append(qid)

    if len(valid_ids) != 20:
        raise RuntimeError(
            f"{engine}: expected 20 paired samples, "
            f"got {len(valid_ids)}"
        )

    self_scores = []
    transfer_scores = []
    deltas = []

    rows = []

    for qid in valid_ids:

        s = get_overall(
            self_data[qid][cfg["self_key"]]
        )

        t = get_overall(
            e5[qid][cfg["transfer_key"]]
        )

        d = t - s

        self_scores.append(s)
        transfer_scores.append(t)
        deltas.append(d)

        rows.append({
            "qid": qid,
            "self": s,
            "transfer": t,
            "delta_pp": d,
        })

    self_mean = statistics.mean(
        self_scores
    )

    transfer_mean = statistics.mean(
        transfer_scores
    )

    delta_pp = (
        transfer_mean
        - self_mean
    )

    relative_change = (
        delta_pp
        / self_mean
        * 100
    )

    transfer_wins = sum(
        d > 1e-12
        for d in deltas
    )

    self_wins = sum(
        d < -1e-12
        for d in deltas
    )

    ties = len(deltas) - transfer_wins - self_wins

    result = {
        "n": len(valid_ids),

        "self_mean":
            self_mean,

        "transfer_mean":
            transfer_mean,

        "delta_pp":
            delta_pp,

        "relative_change_percent":
            relative_change,

        "mean_paired_delta_pp":
            statistics.mean(deltas),

        "median_paired_delta_pp":
            statistics.median(deltas),

        "self_wins":
            self_wins,

        "transfer_wins":
            transfer_wins,

        "ties":
            ties,

        "per_query":
            sorted(
                rows,
                key=lambda x: x["delta_pp"]
            ),
    }

    results[engine] = result


print("=" * 90)
print("E5-A CROSS-GE RULE TRANSFER")
print("Self baseline reused from E2 Strong Run1")
print("=" * 90)


for engine, r in results.items():

    print()
    print(engine.upper())
    print("-" * 60)

    print(
        f"N                  = {r['n']}"
    )

    print(
        f"Self Overall       = "
        f"{r['self_mean']:.2f}"
    )

    print(
        f"Transfer Overall   = "
        f"{r['transfer_mean']:.2f}"
    )

    print(
        f"Transfer - Self    = "
        f"{r['delta_pp']:+.2f} pp"
    )

    print(
        f"Relative change    = "
        f"{r['relative_change_percent']:+.2f}%"
    )

    print(
        f"Median paired Δ    = "
        f"{r['median_paired_delta_pp']:+.2f} pp"
    )

    print(
        f"Self wins          = "
        f"{r['self_wins']}"
    )

    print(
        f"Transfer wins      = "
        f"{r['transfer_wins']}"
    )

    print(
        f"Ties               = "
        f"{r['ties']}"
    )


out = Path(
    "experiments/artifacts/E5/results/"
    "e5a_transfer_statistics.json"
)

out.parent.mkdir(
    parents=True,
    exist_ok=True
)

with open(
    out,
    "w",
    encoding="utf-8"
) as f:
    json.dump(
        results,
        f,
        ensure_ascii=False,
        indent=2,
    )

print()
print("Saved:", out)
