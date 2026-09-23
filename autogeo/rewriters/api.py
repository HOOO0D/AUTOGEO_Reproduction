"""
AutoGEO API module for document rewriting.
"""

import json
import os
import glob
import threading

from typing import Optional, Any
from concurrent.futures import ThreadPoolExecutor, as_completed

from tqdm import tqdm

from .core import rewrite_document
from ..config import Dataset, get_rewrite_method_name
from ..utils.logger import get_logger


def _process_single_question_rewrite(
    question_id: str,
    data: dict,
    rewrite_method_name: str,
    dataset_enum: Dataset,
    engine_llm: str,
    max_retries: int,
    logger: Any,
    file_lock: threading.Lock,
    filename: str,
    rule_path: Optional[str] = None,
) -> tuple[bool, bool]:
    """
    Process a single question for document rewriting.

    Args:
        question_id:
            Current question ID.

        data:
            Loaded data chunk.

        rewrite_method_name:
            Unique rewrite method name.
            For E2, examples:
                e2_gpt_strong
                e2_gpt_self
                e2_claude_strong
                e2_claude_self

        dataset_enum:
            Dataset enum.

        engine_llm:
            Target generative engine.

        max_retries:
            Maximum rewriting retries.

        logger:
            Logger instance.

        file_lock:
            Lock used when writing back to JSON.

        filename:
            Data chunk file path.

        rule_path:
            Optional explicit merged_rules.json path.

    Returns:
        (success, already_exists)

        success:
            True if rewriting succeeded.

        already_exists:
            True if this rewrite already existed and was reused.
    """

    rewrite_key = rewrite_method_name + "_text"

    # ---------------------------------------------------------
    # Resume support
    # ---------------------------------------------------------

    if (
        rewrite_key in data[question_id]
        and data[question_id][rewrite_key]
    ):
        return True, True

    try_count = 0
    success = False

    while try_count < max_retries:
        try:
            idx = data[question_id]["target_id"]

            original_text = (
                data[question_id]["text_list"][idx]
            )

            logger.debug(
                f"Rewriting document for question "
                f"{question_id}..."
            )

            # -------------------------------------------------
            # IMPORTANT FOR E2
            #
            # Explicitly pass rule_path down to core.py.
            #
            # This ensures:
            #
            # GPT Strong
            #   -> gpt/strong/merged_rules.json
            #
            # GPT Self
            #   -> gpt/self/merged_rules.json
            #
            # etc.
            # -------------------------------------------------

            rewritten_text = rewrite_document(
                document=original_text,
                dataset=dataset_enum.value,
                engine_llm=engine_llm,
                rule_path=rule_path,
            )

            with file_lock:
                data[question_id][
                    rewrite_key
                ] = rewritten_text

                with open(
                    filename,
                    "w",
                    encoding="utf-8",
                ) as f:
                    json.dump(
                        data,
                        f,
                        ensure_ascii=False,
                        indent=4,
                    )

            success = True
            break

        except Exception as e:
            try_count += 1

            logger.warning(
                f"Error rewriting question "
                f"{question_id} "
                f"(attempt {try_count}/{max_retries}): "
                f"{e}"
            )

            if try_count >= max_retries:
                logger.error(
                    f"Failed to rewrite question "
                    f"{question_id} after "
                    f"{max_retries} attempts."
                )

    return success, False


def api_rewrite_documents(
    num_examples: Optional[int],
    data_dir: str,
    dataset: str,
    engine_llm: str,
    max_retries: int = 5,
    log_dir: str = "logs",
    logger: Optional[Any] = None,
    max_workers: int = 64,
    progress_bar: Optional[Any] = None,
    rule_path: Optional[str] = None,
    rewrite_method_name: Optional[str] = None,
) -> None:
    """
    Rewrite documents using AutoGEO API.

    Args:
        num_examples:
            Number of examples to process.

        data_dir:
            Directory containing datachunk_*.json.

        dataset:
            Dataset name.

        engine_llm:
            Target generative engine.

        max_retries:
            Maximum number of rewriting retries.

        log_dir:
            Log directory.

        logger:
            Optional logger instance.

        max_workers:
            Maximum parallel workers.

        progress_bar:
            Optional external progress bar.

        rule_path:
            Explicit rule file path.

            For E2 this should point to one of:

                experiments/artifacts/E2/rules/
                    gpt/strong/merged_rules.json

                experiments/artifacts/E2/rules/
                    gpt/self/merged_rules.json

                experiments/artifacts/E2/rules/
                    claude/strong/merged_rules.json

                experiments/artifacts/E2/rules/
                    claude/self/merged_rules.json

        rewrite_method_name:
            Explicit experimental method name.

            Examples:
                e2_gpt_strong
                e2_gpt_self
                e2_claude_strong
                e2_claude_self

            If None, the original AutoGEO naming logic is used.
    """

    # ---------------------------------------------------------
    # Method name
    # ---------------------------------------------------------

    if rewrite_method_name is None:
        rewrite_method_name = get_rewrite_method_name(
            "autogeo_api",
            dataset,
            engine_llm,
        )

    # ---------------------------------------------------------
    # Logger
    # ---------------------------------------------------------

    if logger is None:
        task_name = (
            f"rewrite_{dataset}_{engine_llm}"
        )

        logger = get_logger(
            log_dir=log_dir,
            task_name=task_name,
        )

    logger.info(
        f"Starting document rewriting: "
        f"{dataset} with {engine_llm}"
    )

    logger.info(
        f"Rewrite method name: "
        f"{rewrite_method_name}"
    )

    if rule_path:
        logger.info(
            f"Explicit rule path requested: "
            f"{rule_path}"
        )

    if num_examples is None:
        logger.info(
            f"Processing all examples, "
            f"Method: {rewrite_method_name}"
        )
    else:
        logger.info(
            f"Number of examples: "
            f"{num_examples}, "
            f"Method: {rewrite_method_name}"
        )

    # ---------------------------------------------------------
    # Dataset enum
    # ---------------------------------------------------------

    try:
        dataset_enum = Dataset(dataset)

    except ValueError:
        logger.error(
            f"Unsupported dataset: {dataset}"
        )

        raise ValueError(
            f"Unsupported dataset: {dataset}. "
            f"Supported: "
            f"{[d.value for d in Dataset]}"
        )

    # ---------------------------------------------------------
    # Validate/load rules BEFORE spending API calls
    # ---------------------------------------------------------

    from .core import _load_rules_from_file

    rules, rule_file_path = _load_rules_from_file(
        dataset,
        engine_llm,
        rule_path,
    )

    # For an explicitly supplied path, fail fast instead of
    # silently falling back to a different rule set.
    if rule_path is not None:
        if not os.path.isfile(rule_path):
            raise FileNotFoundError(
                f"Explicit rule file does not exist: "
                f"{rule_path}"
            )

        if rule_file_path != rule_path or not rules:
            raise RuntimeError(
                "Failed to load valid filtered rules "
                f"from explicit rule path: {rule_path}"
            )

    if rule_file_path:
        logger.info(
            f"Using rules from: "
            f"{rule_file_path}"
        )

        logger.info(
            f"Loaded rule count: "
            f"{len(rules) if rules else 0}"
        )

    else:
        logger.info(
            f"Using default rules for "
            f"{dataset} with {engine_llm}"
        )

    # ---------------------------------------------------------
    # Rewrite
    # ---------------------------------------------------------

    processed_questions = 0

    logger.info(
        f"Target number of examples to process: "
        f"{num_examples if num_examples is not None else 'all'}"
    )

    logger.info(
        f"Using {max_workers} parallel workers"
    )

    total_questions = num_examples

    if num_examples is None:
        total_questions = 0

        chunk_files = sorted(
            glob.glob(
                f"{data_dir}/datachunk_*.json"
            )
        )

        for filename in chunk_files:
            try:
                with open(
                    filename,
                    "r",
                    encoding="utf-8",
                ) as f:
                    data = json.load(f)

                total_questions += len(data)

            except Exception:
                pass

    if progress_bar is not None:
        pbar = progress_bar

    elif total_questions is not None:
        pbar = tqdm(
            total=total_questions,
            desc="Step 2: Document Rewriting",
            unit="question",
            dynamic_ncols=True,
        )

    else:
        pbar = None

    chunk_idx = 0

    while (
        num_examples is None
        or processed_questions < num_examples
    ):
        filename = (
            f"{data_dir}/"
            f"datachunk_{chunk_idx}.json"
        )

        if not os.path.exists(filename):
            break

        try:
            with open(
                filename,
                "r",
                encoding="utf-8",
            ) as f:
                data = json.load(f)

        except Exception as e:
            logger.warning(
                f"Error reading {filename}: "
                f"{e}. Skipping."
            )

            chunk_idx += 1
            continue

        all_question_ids = sorted(
            list(data.keys())
        )

        if num_examples is None:
            question_id_list = (
                all_question_ids
            )

        else:
            remaining = (
                num_examples
                - processed_questions
            )

            if remaining <= 0:
                break

            question_id_list = (
                all_question_ids[:remaining]
            )

        file_lock = threading.Lock()

        with ThreadPoolExecutor(
            max_workers=max_workers,
            thread_name_prefix="RewriteWorker",
        ) as executor:

            future_to_qid = {
                executor.submit(
                    _process_single_question_rewrite,
                    question_id,
                    data,
                    rewrite_method_name,
                    dataset_enum,
                    engine_llm,
                    max_retries,
                    logger,
                    file_lock,
                    filename,
                    rule_path,
                ): question_id

                for question_id
                in question_id_list
            }

            for future in as_completed(
                future_to_qid
            ):
                question_id = (
                    future_to_qid[future]
                )

                try:
                    success, already_exists = (
                        future.result()
                    )

                    processed_questions += 1

                    if pbar is not None:
                        pbar.update(1)

                    if (
                        not success
                        and not already_exists
                    ):
                        logger.error(
                            f"Failed to process "
                            f"question {question_id}"
                        )

                    if (
                        num_examples is not None
                        and processed_questions
                        >= num_examples
                    ):
                        for f in future_to_qid:
                            f.cancel()

                        break

                except Exception as e:
                    logger.error(
                        f"Error processing question "
                        f"{question_id}: {e}"
                    )

        if (
            num_examples is not None
            and processed_questions
            >= num_examples
        ):
            break

        chunk_idx += 1

    if (
        pbar is not None
        and progress_bar is None
    ):
        pbar.close()

    if num_examples is not None:
        logger.log_progress(
            processed_questions,
            num_examples,
            "questions",
        )

        logger.info(
            f"Document rewriting completed. "
            f"Processed "
            f"{processed_questions}/"
            f"{num_examples} questions."
        )

    else:
        logger.info(
            f"Document rewriting completed. "
            f"Processed "
            f"{processed_questions} "
            f"questions in total."
        )

    # Do not close logger if supplied externally.