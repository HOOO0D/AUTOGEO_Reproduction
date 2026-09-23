import argparse
import json
import os
from typing import Optional

from ..evaluation.metrics import (
    impression_wordpos_count_simple,
    impression_word_count_simple,
    impression_pos_count_simple,
    extract_citations_new,
)
from ..config import Dataset, get_full_path


def rule_candidate_data(
    num_examples: int = 100,
    data_dir: str = "data/Researchy-GEO/test",
    engine_llm: str = "gemini",
    dataset: str = "Researchy-GEO",
    output_path: Optional[str] = None,
) -> None:
    """
    Construct rule-candidate preference pairs from generative-engine responses.

    For every query:
      1. Load the target generative engine's vanilla response.
      2. Compute visibility for every candidate document:
             visibility = wordpos + word + pos
      3. Select the highest-visibility document as good_document.
      4. Select the lowest-visibility document as bad_document.
      5. Save the resulting preference pair.

    Args:
        num_examples:
            Maximum number of examples to process.

        data_dir:
            Directory containing datachunk_*.json files.

        engine_llm:
            Generative engine whose response is used to infer preferences.
            Examples:
                "gemini-2.5-flash-lite"
                "gpt-4o-mini"
                "claude-3-haiku-20240307"

        dataset:
            Dataset name.
            Examples:
                "Researchy-GEO"
                "E-commerce"
                "GEO-Bench"

        output_path:
            Optional explicit output JSON path.

            If None, the original AutoGEO default path is used:
                data/<dataset>/RL/rule_candidate.json

            For E2 it is strongly recommended to specify this explicitly
            so that GPT and Claude preference-pair files do not overwrite
            each other.
    """

    auto_rule_dict = {}

    processed_count = 0
    successful_count = 0
    skipped_count = 0
    tie_count = 0

    chunk_idx = 0

    response_key = f"{engine_llm}_response"

    print("=" * 80)
    print("Rule Candidate Construction")
    print("=" * 80)
    print(f"Dataset:       {dataset}")
    print(f"Data dir:      {data_dir}")
    print(f"Engine LLM:    {engine_llm}")
    print(f"Response key:  {response_key}")
    print(f"Num examples:  {num_examples}")
    print("=" * 80)

    while processed_count < num_examples:
        filename = os.path.join(
            data_dir,
            f"datachunk_{chunk_idx}.json",
        )

        # Stop once the next sequential chunk does not exist.
        #
        # In the original code this kept searching up to chunk 1000.
        # We retain that behavior so multi-chunk datasets still work
        # even if there happens to be a missing chunk.
        if not os.path.exists(filename):
            chunk_idx += 1

            if chunk_idx > 1000:
                break

            continue

        print(f"\nLoading: {filename}")

        with open(filename, "r", encoding="utf-8") as f:
            data = json.load(f)

        question_id_list = sorted(data.keys())

        for question_id in question_id_list:
            if processed_count >= num_examples:
                break

            processed_count += 1

            print(
                f"[{processed_count}/{num_examples}] "
                f"Evaluating question {question_id}"
            )

            try:
                item = data[question_id]

                # ---------------------------------------------------------
                # 1. Basic fields
                # ---------------------------------------------------------

                if "query" not in item:
                    print(
                        f"  [SKIP] {question_id}: "
                        f"missing 'query'"
                    )
                    skipped_count += 1
                    continue

                if "text_list" not in item:
                    print(
                        f"  [SKIP] {question_id}: "
                        f"missing 'text_list'"
                    )
                    skipped_count += 1
                    continue

                query = item["query"]
                text_list = item["text_list"]

                if not isinstance(text_list, list) or len(text_list) == 0:
                    print(
                        f"  [SKIP] {question_id}: "
                        f"empty or invalid text_list"
                    )
                    skipped_count += 1
                    continue

                # ---------------------------------------------------------
                # 2. Load vanilla response produced by target GE
                # ---------------------------------------------------------

                original_response = item.get(response_key)

                if not original_response:
                    print(
                        f"  [SKIP] {question_id}: "
                        f"missing or empty '{response_key}'"
                    )
                    skipped_count += 1
                    continue

                # ---------------------------------------------------------
                # 3. Parse citations
                # ---------------------------------------------------------

                citations = extract_citations_new(
                    original_response
                )

                n_docs = len(text_list)

                # ---------------------------------------------------------
                # 4. Compute visibility components for ALL documents
                #
                # IMPORTANT:
                # len(citations) is NOT the number of documents.
                #
                # extract_citations_new returns:
                #
                #     paragraphs
                #       -> sentences
                #          -> (tokens, sentence, citation_indices)
                #
                # Therefore we explicitly pass n=n_docs.
                # ---------------------------------------------------------

                wordpos_scores = (
                    impression_wordpos_count_simple(
                        citations,
                        n=n_docs,
                    )
                )

                word_scores = (
                    impression_word_count_simple(
                        citations,
                        n=n_docs,
                    )
                )

                pos_scores = (
                    impression_pos_count_simple(
                        citations,
                        n=n_docs,
                    )
                )

                # AutoGEO visibility:
                #
                #   Vis(d) = Overall(d) + Word(d) + Pos(d)
                #
                # where code-level "wordpos" corresponds to
                # paper-level "Overall".
                visibility_scores = [
                    wordpos_scores[i]
                    + word_scores[i]
                    + pos_scores[i]
                    for i in range(n_docs)
                ]

                if not visibility_scores:
                    print(
                        f"  [SKIP] {question_id}: "
                        f"no valid visibility scores"
                    )
                    skipped_count += 1
                    continue

                # ---------------------------------------------------------
                # 5. Highest / lowest visibility documents
                # ---------------------------------------------------------

                max_score_index = max(
                    range(n_docs),
                    key=lambda i: visibility_scores[i],
                )

                min_score_index = min(
                    range(n_docs),
                    key=lambda i: visibility_scores[i],
                )

                max_score = visibility_scores[
                    max_score_index
                ]

                min_score = visibility_scores[
                    min_score_index
                ]

                # If all scores are identical, Python max/min both choose
                # the first index. We keep the sample for compatibility
                # with the original AutoGEO behavior, but report it.
                if abs(max_score - min_score) < 1e-12:
                    tie_count += 1

                    print(
                        f"  [WARN] Visibility tie: "
                        f"{visibility_scores}"
                    )

                # ---------------------------------------------------------
                # 6. Store preference pair
                # ---------------------------------------------------------

                good_document = text_list[
                    max_score_index
                ]

                bad_document = text_list[
                    min_score_index
                ]

                auto_rule_dict[question_id] = {
                    "query": query,

                    # Original AutoGEO fields
                    "bad_document": bad_document,
                    "good_document": good_document,

                    "winner": "doc_a",

                    "document_a": good_document,
                    "document_b": bad_document,

                    "good_document_content": (
                        good_document
                    ),
                    "bad_document_content": (
                        bad_document
                    ),

                    # Extra metadata for E2 audit/debugging
                    "good_document_index": (
                        max_score_index
                    ),
                    "bad_document_index": (
                        min_score_index
                    ),

                    "good_visibility": max_score,
                    "bad_visibility": min_score,

                    "visibility_scores": (
                        visibility_scores
                    ),

                    "wordpos_scores": (
                        wordpos_scores
                    ),
                    "word_scores": (
                        word_scores
                    ),
                    "pos_scores": (
                        pos_scores
                    ),

                    "engine_llm": engine_llm,
                }

                successful_count += 1

                print(
                    f"  good = doc {max_score_index}, "
                    f"visibility = {max_score:.6f}"
                )

                print(
                    f"  bad  = doc {min_score_index}, "
                    f"visibility = {min_score:.6f}"
                )

            except Exception as e:
                skipped_count += 1

                print(
                    f"  [ERROR] {question_id}: {e}"
                )

        if processed_count >= num_examples:
            break

        chunk_idx += 1

    # ---------------------------------------------------------------------
    # Determine output path
    # ---------------------------------------------------------------------

    if output_path is None:
        try:
            dataset_enum = Dataset(dataset)

            output_path = get_full_path(
                dataset_enum,
                "rule_candidate_file",
            )

        except (ValueError, KeyError):
            dataset_short = (
                dataset
                .replace("-", "_")
                .replace(" ", "_")
                .lower()
            )

            output_path = os.path.join(
                data_dir.rsplit("/", 1)[0],
                "RL",
                f"{dataset_short}_rule_candidate.json",
            )

    output_dir = os.path.dirname(
        output_path
    )

    if output_dir:
        os.makedirs(
            output_dir,
            exist_ok=True,
        )

    # ---------------------------------------------------------------------
    # Save
    # ---------------------------------------------------------------------

    with open(
        output_path,
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(
            auto_rule_dict,
            f,
            ensure_ascii=False,
            indent=4,
        )

    # ---------------------------------------------------------------------
    # Summary
    # ---------------------------------------------------------------------

    print()
    print("=" * 80)
    print("Rule Candidate Construction Completed")
    print("=" * 80)

    print(
        f"Requested examples:       "
        f"{num_examples}"
    )

    print(
        f"Processed examples:       "
        f"{processed_count}"
    )

    print(
        f"Successful preference pairs: "
        f"{successful_count}"
    )

    print(
        f"Skipped / failed:         "
        f"{skipped_count}"
    )

    print(
        f"Visibility ties:          "
        f"{tie_count}"
    )

    print(
        f"Output file:              "
        f"{output_path}"
    )

    print("=" * 80)


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Construct AutoGEO rule-candidate preference "
            "pairs from vanilla GE responses."
        )
    )

    parser.add_argument(
        "--num_examples",
        type=int,
        default=100,
        help="Maximum number of examples to process",
    )

    parser.add_argument(
        "--data_dir",
        type=str,
        default="data/Researchy-GEO/test",
        help="Directory containing datachunk_*.json",
    )

    parser.add_argument(
        "--engine_llm",
        type=str,
        default="gemini",
        help=(
            "Target generative engine model name, "
            "e.g. gpt-4o-mini"
        ),
    )

    parser.add_argument(
        "--dataset",
        type=str,
        default="Researchy-GEO",
        choices=[
            d.value
            for d in Dataset
        ],
        help="Dataset name",
    )

    parser.add_argument(
        "--output_path",
        type=str,
        default=None,
        help=(
            "Explicit output JSON path. "
            "Recommended for E2."
        ),
    )

    args = parser.parse_args()

    rule_candidate_data(
        num_examples=args.num_examples,
        data_dir=args.data_dir,
        engine_llm=args.engine_llm,
        dataset=args.dataset,
        output_path=args.output_path,
    )


if __name__ == "__main__":
    main()