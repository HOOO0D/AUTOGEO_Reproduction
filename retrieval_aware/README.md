# E7 Retrieval-Aware AutoGEO — Frozen Protocol v1

This package is isolated from the original AutoGEO pipeline. It does not alter
`data/`, `experiments/frozen/`, the AutoGEO_API core, or E0/E1/Hijack/Poison.

## Track A — Original AutoGEO protocol

Track A preserves the paper/repository protocol and should preferentially read
existing E0/E1 results rather than rerun them.

| Identifier | Input context | Rewrite | Retrieval after rewrite |
| --- | --- | --- | --- |
| `vanilla_original_protocol` | original Researchy fixed `text_list[5]` | none | none |
| `autogeo_api_original_protocol` | original Researchy fixed `text_list[5]` | unchanged AutoGEO_API at original `target_id` | none |

The second arm retains the existing operation
`rewritten_text_list[target_id] = rewritten_text`, followed directly by GE and
GEO/GEU. E7 must not adapt or overwrite this baseline.

## Track B — E7 retrieval-aware protocol

The formal backend is `researchy_pooled_local`. It never depends on a
`CLUEWEB_API_KEY`, the DeepResearchGym API, or the mutable ClueWeb22 index.
Queries remain the original Researchy-GEO queries; NFCorpus/BEIR queries and
documents are not permitted.

The two split-isolated physical corpora are:

```text
DEV corpus  = RULE_POOL100 documents + DEV20 documents
TEST corpus = RULE_POOL100 documents + TEST50 documents
```

RULE_POOL100 is shared background. DEV never includes TEST50 documents, so dev
tuning cannot observe test documents. A query's five `text_list` entries are
called its **original candidate documents**. Existing `target_id` designates
the original target document for the experiment, but neither it nor the other
four candidates is claimed to be click/qrel ground truth.

Local retrieval ranks the query-specific corpus and keeps Top-100 as the fixed
local retrieval pool. Its non-target members are **competitive documents**.
The same original target, query, local pool, retriever, GE, and evaluator
configuration are shared by all three Track B arms:

```text
Original D*
├── original_e2e
├── autogeo_api_e2e
└── ra_autogeo_e2e
```

| Identifier | Target transformation | Retrieval feedback |
| --- | --- | --- |
| `original_e2e` | none | none |
| `autogeo_api_e2e` | unchanged AutoGEO_API, one shot | forbidden |
| `ra_autogeo_e2e` | iterative candidates using the same AutoGEO rules | allowed |

Every arm locally re-indexes after replacing only D*. `original_e2e` performs
the same operation with unchanged D* and acts as the retrieval calibration/base
case. AutoGEO_API and RA-AutoGEO are siblings: RA never starts from the
AutoGEO-rewritten text. Both rewrites start from the same original candidate
document selected by the frozen `target_id`.

For protocol v1:

```text
Rules_RA_v1 == Rules_AutoGEO == autogeo_filtered_rules
```

RA-v1 does not learn RetrievalRules. Its additional signal is query, current
rank, retrieval score, and retriever-in-the-loop feedback. It generates three
candidates per round for at most two rounds.

## Method-specific Top-5 and evaluation

Each arm produces its own ranking and actual Top-5. GE always runs on that
Top-5, including when D* is absent.

- If final target rank is greater than 5, target E2E visibility is exactly zero.
  D* has no within-context GEO score, but the actual Top-5 response and all GEU
  metrics are still required.
- If final target rank is at most 5, its source index is found dynamically in
  that arm's Top-5. Existing GEO functions score this dynamic index, and GEU is
  calculated on the generated answer.

Source indices are zero-based because the existing GE prompt emits
`### Source 0` through `### Source 4`; retrieval ranks are one-based.

## Statistical comparison boundary

Track A and Track B do not use the same target population or context protocol.
They must not be used in a same-target paired significance test. Cross-track
results may be reported descriptively with their protocol labels.

The primary paired experiment is exactly the three Track B arms. Pairing keys
are query ID, pool identity, and target document ID. A row with a different D*,
pool, or query is invalid for the paired analysis.

## Isolation and outputs

All new artifacts must pass `utils.assert_e7_output_path` and remain below:

```text
outputs/e7_retrieval_aware/
  corpus/
  mappings/
  pools/
  targets/
  audits/
  embeddings/
  rewrites/
  calibration/
  evaluation/
  logs/
```

`python -m retrieval_aware` validates the frozen registry/config and creates
only this directory layout. It does not retrieve, load an embedding model, or
call an LLM.

## Researchy-GEO pooled local corpus manifests

Build both split-isolated manifests with:

```text
python -m retrieval_aware.corpus_builder
```

The builder reads only `experiments/frozen/researchy_rulepool100`,
`researchy_dev20`, and `researchy_test50`. It writes only:

```text
outputs/e7_retrieval_aware/corpus/dev_corpus_manifest.json
outputs/e7_retrieval_aware/corpus/test_corpus_manifest.json
```

Frozen rows contain document text but no document ID or URL. Physical document
identity is therefore `e7doc_sha256_<normalized-text-sha256>`, where
normalization is NFKC plus whitespace collapse without case folding. One
physical document is retained for repeated content; the first occurrence is
the primary provenance and every later occurrence is preserved under
`duplicate_provenance`.

Each evaluation query also records exactly five `original_candidate_ids` and
the `original_target_id_document` selected by its frozen `target_id`. Rule-pool
questions contribute background documents but are not DEV/TEST evaluation
queries. Manifests are content-addressed and cannot be replaced by different
content at the same path.

## Frozen local retrievers: retained R0 and formal R1

R0 is retained as `r0_local_aligned` and pinned to:

```text
openbmb/MiniCPM-Embedding-Light
revision ce6cb0e22f4838f44731910be439651eebc76838
```

Its frozen contract is identical before and after rewriting:

- query: official `encode_query` with `Query:` instruction;
- document and rewritten target: official `encode_corpus`, no instruction;
- maximum length: 8192 tokens;
- dense dimension: 1024;
- output normalization: L2;
- similarity: exact float32 dot product;
- ranking: descending score, then ascending `document_id` tie-break.

R0 produced the retained DEV20 eligibility audit in `targets/dev_targets.json`.
All 100 original candidate slots occupied local ranks 1--5, so no query had a
natural rank 6--100 promotion target. That audit and R0's existing DEV/TEST
caches remain immutable and are not overwritten.

The formal independent Track B retriever is `r1_independent_dense`, pinned
before the R1 DEV eligibility audit to:

```text
BAAI/bge-base-en-v1.5
revision a5beb1e3e68b9ab74eb54cfd186867f64f240e1a
```

R1 follows the model author's query-to-passage recipe:

- query prefix: `Represent this sentence for searching relevant passages: `;
- documents: no prefix;
- pooling: first-token/CLS;
- maximum query and document length: 512 tokens with truncation;
- dense dimension: 768;
- output normalization: L2;
- similarity: exact float32 dot product;
- ranking: descending score, then ascending `document_id` tie-break.

These choices come from the official
[BGE model card](https://huggingface.co/BAAI/bge-base-en-v1.5) and
[FlagEmbedding inference documentation](https://github.com/FlagOpen/FlagEmbedding/tree/master/examples/inference/embedder).
R1 is independent of R0: it is a different public checkpoint and architecture
family with a different training lineage, query preprocessing, pooling,
embedding dimension, and maximum length. It is fully local, deterministic,
not trained on DEV20, and was fixed without comparing candidate eligibility
against any alternative retriever.

Both retrievers implement one model-independent contract containing
`retriever_id`, model name/revision, separate query/document encoding configs,
normalization, similarity, maximum length, dimension, and dtype. Downstream
loaders select this contract through `retriever_registry.py`; they do not name
model classes.

No ANN index exists. Retrieval scores the complete physical DEV or TEST corpus
and only then truncates to Top-100. The retained R0 DEV/TEST caches were built
from its pinned local snapshot with the E7 runtime:

```text
outputs/e7_retrieval_aware/runtime/minicpm-embedding/bin/python \
  -m retrieval_aware.frozen_local_retriever
```

The retained R0 `dev_corpus_cache/` and `test_corpus_cache/` each contain a
read-only `embeddings.npy` and `metadata.json`. Metadata binds the cache to the
model name/revision, encoder configuration, corpus manifest SHA-256, document
ID ordering, text hashes, dimension, normalization, dtype, and embedding-file
SHA-256. A later run may reuse an identical cache but cannot overwrite it.

`retrieve` and `rerank_with_rewritten_target` share the same query encoder,
document encoder, dot-product implementation, and deterministic ranking path.
Counterfactual re-indexing encodes only the supplied target text and asserts
that all N-1 competitor identities, texts, hashes, and cached vectors remain
unchanged. It never mutates the corpus or frozen cache.

R1 uses a disjoint namespace:

```text
outputs/e7_retrieval_aware/embeddings/r1_independent/
```

During the R1 bootstrap stage, only the full DEV corpus cache and one-query
smoke artifact were permitted. Run `python -m retrieval_aware.independent_retriever` to validate
model loading, corpus/query encoding, exact full ranking, deterministic repeat
ranking, and recording of all five original candidate slots. It does not run
DEV20 eligibility, select a target, process TEST50, rewrite text, or invoke
GE/GEO/GEU.

The subsequent DEV20 R1 eligibility-only audit is written separately to
`targets/dev_eligibility_r1.json`. It records each full-corpus Top-10 and all
five original candidate slots with their R1 ranks/scores, plus fixed rank-range
counts and a descriptive R0/R1 Top-5-retention comparison. It does not execute
the target-selection policy or create a formal target manifest:

```text
python -m retrieval_aware.r1_eligibility_audit
```

## Researchy pooled corpus V2

Pool V2 expands DEV once from the complete canonical Researchy-GEO train
source plus the fixed RULE_POOL100 and DEV20 provenance. Source decisions are
frozen before construction in `corpus/pool_v2/source_audit.json`; RL replicas,
generated rewrites, and all test-derived sources are excluded. Physical IDs
still use normalized-text SHA-256. To enforce content-level TEST50 isolation,
any canonical-train identity also present in TEST50 is excluded before corpus
construction. This filter is independent of retrieval and eligibility.

```text
python -m retrieval_aware.document_source_audit
python -m retrieval_aware.pooled_corpus_v2
python -m retrieval_aware.pool_v2_embeddings
```

The V2 manifest and frozen R1 cache live only under `pool_v2/` namespaces.
These commands do not rank queries, audit eligibility, select targets, rewrite
documents, or invoke GE/GEO/GEU.

Pool V2 contains 50,508 retained provenance slots and 44,264 deduplicated
physical documents. The R1 cache reuses the 597 byte-identical V1 document
vectors and encodes only 43,667 V2-only documents. The new vectors were built
with deterministic local CUDA execution after a 64-document CPU/CUDA check
showed repeat-exact CUDA output and a maximum CPU/CUDA absolute delta of
`1.4007091522216797e-06` (all values within `1e-5`). Execution runtime details
are construction metadata, not part of the frozen retriever specification;
model, revision, query/document preprocessing, CLS pooling, maximum length,
normalization, dtype, similarity, and ranking remain unchanged.

## Retained R0 DEV20 eligibility and target policy

`python -m retrieval_aware.local_target_selection` is restricted to DEV20. For
each query it runs an exact full-corpus ranking and audits all five original
candidate slots, retaining document ID, original text index, original
`target_id` flag, local rank, and local score. No pooled-corpus document outside
those five identities may become a target.

Eligibility is rank 6--100. An eligible original `target_id` document always
wins and is labelled `target_source: original_target_id`. Otherwise selection
checks alternate original candidates by `easy` 6--20, `medium` 21--50, then
`hard` 51--100, choosing the lowest local rank within the first non-empty
bucket. Such a target is labelled `alternate_original_candidate`. A query with
no eligible original candidate remains `ineligible`; no replacement query is
introduced and hard examples are never filtered.

The immutable audit is written to
`outputs/e7_retrieval_aware/targets/dev_targets.json`. TEST50 target selection
is intentionally not implemented or run at this stage.

## Optional future ClueWeb backend

The audited `DeepResearchGymClueWeb22B` adapter and its schemas remain in the
package for optional future work, but they are not the formal E7 backend and
are never called by the local corpus builder. The legacy preparation command
is explicit and non-default:

```text
python -m retrieval_aware.prepare_retrieval --limit 3 --run-retrieval
```

The fixed adapter follows the audited DeepResearchGym ClueWeb22-B contract:
`GET https://clueweb22.us/search`, query parameters `query` and `k`, and the
`X-API-Key` header. The response is a top-level `results` list whose entries are
Base64-encoded JSON objects with `URL`, `URL-hash`, `Language`, `ClueWeb22-ID`,
and `Clean-Text`. That response does not expose a retrieval score, so E7 stores
`score: null` instead of inventing one.

Audited upstream references are the
[DeepResearchGym site](https://www.deepresearchgym.ai/), its
[official repository](https://github.com/cxcscmu/deepresearch_benchmarking),
and the
[Researchy Questions dataset](https://huggingface.co/datasets/corbyrosset/researchy_questions).

Those optional artifacts are not inputs to the formal pooled-local experiment.

## Deferred implementation boundaries

- TEST50 target eligibility/selection;
- RA prompt and iterative candidate-selection implementation;
- GE/GEO/GEU execution.

The current stage intentionally contains no rewrite, GE, GEO, or GEU execution.
