import argparse
import glob
import json
import os

from autogeo.evaluation.metrics.geo_score import (
    extract_citations_new,
    impression_wordpos_count_simple,
    impression_word_count_simple,
    impression_pos_count_simple,
)


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--data_dir", required=True)
    parser.add_argument("--engine_llm", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--num_examples", type=int, default=100)

    args = parser.parse_args()

    output = {}

    processed = 0
    tie_count = 0

    files = sorted(
        glob.glob(os.path.join(args.data_dir, "datachunk_*.json"))
    )

    for filename in files:
        with open(filename, "r", encoding="utf-8") as f:
            data = json.load(f)

        for qid in sorted(data.keys()):
            if processed >= args.num_examples:
                break

            item = data[qid]

            query = item["query"]
            text_list = item["text_list"]

            response_key = f"{args.engine_llm}_response"

            response = item.get(response_key)

            if not response:
                print(f"[SKIP] {qid}: missing {response_key}")
                continue

            citations = extract_citations_new(response)

            n = len(text_list)

            wordpos = impression_wordpos_count_simple(
                citations,
                n=n
            )

            word = impression_word_count_simple(
                citations,
                n=n
            )

            pos = impression_pos_count_simple(
                citations,
                n=n
            )

            visibility = [
                wordpos[i] + word[i] + pos[i]
                for i in range(n)
            ]

            max_idx = max(
                range(n),
                key=lambda i: visibility[i]
            )

            min_idx = min(
                range(n),
                key=lambda i: visibility[i]
            )

            if abs(
                visibility[max_idx] - visibility[min_idx]
            ) < 1e-12:
                tie_count += 1
                print(
                    f"[WARN] {qid}: all/equal visibility "
                    f"{visibility}"
                )

            output[qid] = {
                "query": query,

                "good_document": text_list[max_idx],
                "bad_document": text_list[min_idx],

                "good_index": max_idx,
                "bad_index": min_idx,

                "visibility": visibility,

                "wordpos": wordpos,
                "word": word,
                "pos": pos,

                "engine_llm": args.engine_llm,
            }

            processed += 1

        if processed >= args.num_examples:
            break

    os.makedirs(
        os.path.dirname(args.output),
        exist_ok=True
    )

    with open(
        args.output,
        "w",
        encoding="utf-8"
    ) as f:
        json.dump(
            output,
            f,
            ensure_ascii=False,
            indent=2
        )

    print()
    print("=" * 60)
    print("Preference pair construction finished")
    print("engine:", args.engine_llm)
    print("samples:", len(output))
    print("ties:", tie_count)
    print("output:", args.output)


if __name__ == "__main__":
    main()
