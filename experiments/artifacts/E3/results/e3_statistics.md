# E3 Engine-Specific Preference Statistics

## Rule and keyword counts

| Engine | Rules | Keywords | Keyword coverage |
|---|---:|---:|---:|
| Gemini | 13 | 15 | 75.00% |
| Gpt | 13 | 15 | 75.00% |
| Claude | 12 | 13 | 65.00% |

## Pairwise Jaccard

| Pair | Intersection | Union | Reproduction | Paper | Difference |
|---|---:|---:|---:|---:|---:|
| gemini_vs_gpt | 12 | 18 | 66.67% | 78.95% | -12.28 pp |
| gemini_vs_claude | 11 | 17 | 64.71% | 84.21% | -19.50 pp |
| gpt_vs_claude | 12 | 16 | 75.00% | 84.21% | -9.21 pp |

## Overall overlap

- Observed keyword union: 18/20 (90.00%)
- Common to all three: 10
- Shared by exactly two: 5
- Unique to one engine: 3
- Keywords appearing in >=2 engines: 15/18 (83.33%)
- Mean pairwise Jaccard: 68.79%
- Paper mean pairwise Jaccard: 82.46%
- Mean difference: -13.67 pp
