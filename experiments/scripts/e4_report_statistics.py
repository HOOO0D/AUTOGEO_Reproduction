import json
from pathlib import Path


BASE = Path("experiments/artifacts/E4/results")

KEYWORD_FILE = BASE / "keyword_sets.json"
JACCARD_FILE = BASE / "jaccard.json"

with open(KEYWORD_FILE, encoding="utf-8") as f:
    raw_sets = json.load(f)

with open(JACCARD_FILE, encoding="utf-8") as f:
    jaccard = json.load(f)


R = set(raw_sets["researchy_geo"])
G = set(raw_sets["geo_bench"])
E = set(raw_sets["ecommerce"])


# ============================================================
# Common / Shared / Unique
# ============================================================

common_all = R & G & E

rg_only = (R & G) - E
re_only = (R & E) - G
ge_only = (G & E) - R

r_unique = R - (G | E)
g_unique = G - (R | E)
e_unique = E - (R | G)

union_all = R | G | E


# ============================================================
# Jaccard values
# ============================================================

rg = (
    len(R & G)
    / len(R | G)
    * 100
)

re = (
    len(R & E)
    / len(R | E)
    * 100
)

ge = (
    len(G & E)
    / len(G | E)
    * 100
)

cross_ecommerce_mean = (
    re + ge
) / 2

open_domain_gap = (
    rg - cross_ecommerce_mean
)


# ============================================================
# Paper reference
# ============================================================

paper = {
    "researchy_geo_vs_geo_bench": 88.24,
    "researchy_geo_vs_ecommerce": 34.78,
    "geo_bench_vs_ecommerce": 40.00,
}


# ============================================================
# Print
# ============================================================

print("=" * 80)
print("E4 DOMAIN-SPECIFIC PREFERENCE SUMMARY")
print("=" * 80)

print()
print("KEYWORD COUNTS")

print(
    f"Researchy-GEO : {len(R)}"
)

print(
    f"GEO-Bench     : {len(G)}"
)

print(
    f"E-commerce    : {len(E)}"
)

print(
    f"Union         : {len(union_all)}"
)


print()
print("=" * 80)
print("PAIRWISE JACCARD")
print("=" * 80)

results = [
    (
        "Researchy-GEO vs GEO-Bench",
        rg,
        paper[
            "researchy_geo_vs_geo_bench"
        ],
    ),
    (
        "Researchy-GEO vs E-commerce",
        re,
        paper[
            "researchy_geo_vs_ecommerce"
        ],
    ),
    (
        "GEO-Bench vs E-commerce",
        ge,
        paper[
            "geo_bench_vs_ecommerce"
        ],
    ),
]

for name, reproduced, reference in results:

    print()
    print(name)

    print(
        f"  Reproduction : "
        f"{reproduced:.2f}%"
    )

    print(
        f"  Paper        : "
        f"{reference:.2f}%"
    )

    print(
        f"  Difference   : "
        f"{reproduced-reference:+.2f} pp"
    )


print()
print("=" * 80)
print("DOMAIN GAP")
print("=" * 80)

print(
    f"Open-domain overlap "
    f"(Researchy vs GEO-Bench): "
    f"{rg:.2f}%"
)

print(
    f"Mean E-commerce cross-domain overlap: "
    f"{cross_ecommerce_mean:.2f}%"
)

print(
    f"Open-domain advantage: "
    f"{open_domain_gap:.2f} pp"
)


print()
print("=" * 80)
print("COMMON TO ALL THREE")
print("=" * 80)

for k in sorted(common_all):
    print(" -", k)


print()
print("=" * 80)
print("SHARED BY EXACTLY TWO")
print("=" * 80)

print()
print("Researchy-GEO + GEO-Bench")

for k in sorted(rg_only):
    print(" -", k)

print()
print("Researchy-GEO + E-commerce")

for k in sorted(re_only):
    print(" -", k)

print()
print("GEO-Bench + E-commerce")

for k in sorted(ge_only):
    print(" -", k)


print()
print("=" * 80)
print("UNIQUE")
print("=" * 80)

print()
print("Researchy-GEO")

for k in sorted(r_unique):
    print(" -", k)

print()
print("GEO-Bench")

for k in sorted(g_unique):
    print(" -", k)

print()
print("E-commerce")

for k in sorted(e_unique):
    print(" -", k)


# ============================================================
# Save summary
# ============================================================

summary = {
    "keyword_counts": {
        "researchy_geo": len(R),
        "geo_bench": len(G),
        "ecommerce": len(E),
        "union": len(union_all),
    },

    "jaccard_percent": {
        "researchy_geo_vs_geo_bench":
            rg,

        "researchy_geo_vs_ecommerce":
            re,

        "geo_bench_vs_ecommerce":
            ge,
    },

    "domain_gap": {
        "open_domain_overlap":
            rg,

        "mean_ecommerce_cross_domain_overlap":
            cross_ecommerce_mean,

        "open_domain_advantage_pp":
            open_domain_gap,
    },

    "common_all":
        sorted(common_all),

    "shared_exactly_two": {
        "researchy_geo_geo_bench":
            sorted(rg_only),

        "researchy_geo_ecommerce":
            sorted(re_only),

        "geo_bench_ecommerce":
            sorted(ge_only),
    },

    "unique": {
        "researchy_geo":
            sorted(r_unique),

        "geo_bench":
            sorted(g_unique),

        "ecommerce":
            sorted(e_unique),
    },
}


output = BASE / "e4_statistics.json"

with open(
    output,
    "w",
    encoding="utf-8"
) as f:

    json.dump(
        summary,
        f,
        ensure_ascii=False,
        indent=4
    )


print()
print("Saved:", output)
