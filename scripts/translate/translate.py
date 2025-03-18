import argparse
import json
import logging
import os
import re
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import tqdm
from openai import OpenAI
from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_exponential

from lettucedetect.datasets.hallucination_dataset import HallucinationData, HallucinationSample

TRANSLATION_ANSWER = """
Translate the following text from {source_lang} to {target_lang}.
- If the original text contains <HAL> tags, translate the content inside <HAL> tags and ensure the number of the <HAL> tags remain exactly the same in the output.
- If the original text do not contain <HAL> tags, just translate the text.
- Do NOT add any <HAL> tags if they were not in the original text.
- Do NOT remove any <HAL> tags that were in the original text.
- Do not include any additional sentences summarizing or explaining the translation. 
- Your output should be just the translated text, nothing else.

{source_lang}: 
======START======
{text}
======END======

Output in {target_lang}:
"""

TRANSLATION_PROMPT = """
Translate the following prompt from {source_lang} to {target_lang}.
- Translate only the given prompt.
- Do not include any additional sentences summarizing or explaining the translation. 
- Your output should be just the translated prompt, nothing else.

{source_lang}:
======START-PROMPT======
{text}
======END-PROMPT======

Output in {target_lang}:
"""

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.StreamHandler(),
    ],
)
logger = logging.getLogger("translator")


class TranslationError(Exception):
    """Exception raised for errors during translation."""

    pass


def setup_logging(output_dir: Path) -> None:
    """Set up logging to file in the output directory.

    :param output_dir: Directory to save log file
    """
    log_file = output_dir / "translation.log"
    file_handler = logging.FileHandler(log_file)
    file_handler.setFormatter(
        logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    )
    logger.addHandler(file_handler)


def get_openai_client() -> OpenAI:
    """Get OpenAI client configured from environment variables.

    :return: Configured OpenAI client
    :raises ValueError: If API key is not set
    """
    api_key = os.getenv("OPENAI_API_KEY") or "EMPTY"
    base_url = os.getenv("OPENAI_BASE_URL") or "https://api.openai.com/v1"

    return OpenAI(
        api_key=api_key,
        base_url=base_url,
    )


@retry(
    retry=retry_if_exception_type((Exception)),
    stop=stop_after_attempt(5),
    wait=wait_exponential(multiplier=1, min=2, max=60),
    reraise=True,
    before_sleep=lambda retry_state: logger.warning(
        f"API call failed. Retrying in {retry_state.next_action.sleep} seconds... "
        f"(Attempt {retry_state.attempt_number}/5)"
    ),
)
def translate_text(
    text: str,
    client: OpenAI,
    model: str,
    source_lang: str = "EN",
    target_lang: str = "DE",
    prompt: bool = False,
) -> str:
    """Translate text using OpenAI API with automatic retries.

    :param text: Text to translate
    :param client: OpenAI client
    :param model: Model to use for translation
    :param source_lang: Source language code
    :param target_lang: Target language code
    :param prompt: Whether the text is a prompt
    :return: Translated text
    :raises TranslationError: If translation fails after retries
    """
    if not text.strip():
        return ""

    try:
        translation_prompt = TRANSLATION_ANSWER if prompt else TRANSLATION_PROMPT
        translation_prompt = translation_prompt.format(
            source_lang=source_lang, target_lang=target_lang, text=text
        )

        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "user", "content": translation_prompt},
            ],
            temperature=0.3,
        )

        # Strip lines starting with the character '='
        content = "\n".join(
            [
                line
                for line in response.choices[0].message.content.split("\n")
                if not line.strip().startswith("=")
            ]
        )

        return content.strip()
    except Exception as e:
        logger.error(f"Translation error: {e!s}")
        raise TranslationError(f"Failed to translate text: {e!s}") from e


def merge_overlapping_spans(labels: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Merge overlapping hallucination spans into a single span.

    :param labels: List of label spans to merge
    :return: List of merged spans
    """
    if not labels:
        return []

    labels_copy = sorted(labels, key=lambda x: x["start"])
    new_labels = []
    current_span = labels_copy[0].copy()

    for span in labels_copy[1:]:
        if span["start"] <= current_span["end"]:
            current_span["end"] = max(current_span["end"], span["end"])
        else:
            new_labels.append(current_span)
            current_span = span.copy()

    new_labels.append(current_span)
    return new_labels


def put_hallucination_tags(
    sample: HallucinationSample, answer: str
) -> tuple[str, list[dict[str, Any]]]:
    """Add hallucination tags to the text.

    :param sample: Sample containing labels
    :param answer: Text to add tags to
    :return: Tuple of (tagged text, merged labels)
    """
    # Skip the process if there are no labels
    if not sample.labels:
        return answer, []

    labels = merge_overlapping_spans(sample.labels)
    labels = sorted(labels, key=lambda x: (x["end"], x["start"]), reverse=True)
    tagged_answer = answer

    for label in labels:
        start, end = label["start"], label["end"]
        if start < 0 or end > len(tagged_answer) or start >= end:
            logger.warning(
                f"Invalid span: {start}-{end} for text of length {len(tagged_answer)}. Skipping."
            )
            continue

        tagged_answer = tagged_answer[:end] + "</HAL>" + tagged_answer[end:]
        tagged_answer = tagged_answer[:start] + "<HAL>" + tagged_answer[start:]

    return tagged_answer, labels


def find_hallucination_tags(
    text: str, labels: list[dict[str, Any]], sample_index: int
) -> tuple[list[tuple[int, int, str]], str]:
    """Find hallucination tags in the translated text and remove them.

    :param text: Text to search for tags
    :param labels: Original labels
    :param sample_index: Index of the sample
    :return: Tuple of (list of (start, end, label) tuples, cleaned text without HAL tags)
    """
    if not labels:
        return [], text

    # Find all <HAL> and </HAL> tags
    pattern = r"<(/?HAL)>"

    cleaned_text = ""
    open_tags = {}  # Maps an index to the starting position in cleaned text
    hal_spans = []  # List to store (start, end, label) tuples

    last_index = 0
    tag_count = 0

    for match in re.finditer(pattern, text):
        start, end = match.span()
        tag_type = match.group(1)

        cleaned_text += text[last_index:start]

        if tag_type == "HAL":  # Opening tag
            open_tags[tag_count] = len(cleaned_text)
        elif tag_type == "/HAL":  # Closing tag
            if tag_count in open_tags:
                label_text = "Unknown"
                if tag_count < len(labels):
                    label_text = labels[tag_count].get("label", "Unknown")
                else:
                    message = f"IndexError: No label for hallucinated text at sample ({sample_index}), span {tag_count}"
                    logger.warning(message)

                hal_spans.append((open_tags[tag_count], len(cleaned_text), label_text))
                tag_count += 1
            else:
                message = f"Warning: Found closing HAL tag without matching opening tag in sample {sample_index}"
                logger.warning(message)

        last_index = end

    # Add remaining text
    cleaned_text += text[last_index:]

    if tag_count < len(labels):
        message = f"Warning: Not all hallucination spans were found in sample {sample_index}. Found {tag_count}, expected {len(labels)}"
        logger.warning(message)

    return hal_spans, cleaned_text


def translate_sample(
    sample: HallucinationSample,
    client: OpenAI,
    model: str,
    sample_index: int,
    log_file: Path,
    source_lang: str,
    target_lang: str,
    dataset: str,
) -> HallucinationSample | None:
    """Translate a single sample.

    :param sample: Sample to translate
    :param client: OpenAI client
    :param model: Model to use
    :param sample_index: Sample index
    :param log_file: Path to log file
    :param source_lang: Source language code
    :param target_lang: Target language code
    :param dataset: Dataset name
    :return: Translated sample or None if translation failed
    """
    try:
        # Skip processing if we have an empty sample
        if not sample.prompt.strip() or not sample.answer.strip():
            logger.warning(f"Sample {sample_index} has empty prompt or answer. Skipping.")
            return None

        translated_prompt = translate_text(sample.prompt, client, model, source_lang, target_lang)

        tagged_answer, labels = put_hallucination_tags(sample, sample.answer)

        translated_answer = translate_text(
            tagged_answer, client, model, source_lang, target_lang, prompt=True
        )

        # Default to the translated answer (will be replaced if there are hallucination spans)
        cleaned_answer = translated_answer

        labels = []
        if sample.labels:
            # Get hallucination spans and cleaned text (without HAL tags)
            hal_spans, cleaned_answer = find_hallucination_tags(
                translated_answer, sample.labels, sample_index
            )

            for span in hal_spans:
                start, end, label = span
                labels.append(
                    {
                        "start": start,
                        "end": end,
                        "label": label,
                    }
                )

        return HallucinationSample(
            prompt=translated_prompt,
            answer=cleaned_answer,
            labels=labels,
            split=sample.split,
            task_type=sample.task_type,
            dataset=dataset,
            language=target_lang.lower(),
        )
    except Exception as e:
        logger.error(f"Error translating sample {sample_index}: {e!s}")
        with open(log_file, "a") as log:
            log.write(f"Error translating sample {sample_index}: {e!s}\n")
        return None


def load_check_existing_data(output_file: Path) -> HallucinationData:
    """Load existing data or create new data.

    :param output_file: Path to the output file
    :return: Existing HallucinationData or new empty HallucinationData
    """
    if output_file.exists():
        try:
            return HallucinationData.from_json(json.loads(output_file.read_text()))
        except (json.JSONDecodeError, KeyError) as e:
            logger.error(f"Error loading existing data: {e!s}. Starting with empty dataset.")
            return HallucinationData(samples=[])
    else:
        return HallucinationData(samples=[])


def translate_sample_wrapper(args):
    """Wrapper function for translate_sample to use with concurrent.futures.

    :param args: Tuple of arguments for translate_sample
    :return: Result of translate_sample
    """
    return translate_sample(*args)


def process_batch(
    samples: list[HallucinationSample],
    client: OpenAI,
    model: str,
    start_idx: int,
    log_file: Path,
    source_lang: str,
    target_lang: str,
    dataset: str,
    executor: ThreadPoolExecutor,
) -> list[HallucinationSample]:
    """Process a batch of samples in parallel using an existing executor.

    :param samples: List of samples to process
    :param client: OpenAI client
    :param model: Model to use
    :param start_idx: Starting index for the batch
    :param log_file: Path to log file
    :param source_lang: Source language code
    :param target_lang: Target language code
    :param dataset: Dataset name
    :param executor: ThreadPoolExecutor to use
    :return: List of translated samples (excluding failed translations)
    """
    futures = []
    for i, sample in enumerate(samples, start=start_idx):
        args = (sample, client, model, i, log_file, source_lang, target_lang, dataset)
        future = executor.submit(translate_sample_wrapper, args)
        futures.append(future)

    results = []
    for future in as_completed(futures):
        try:
            result = future.result()
            if result:
                results.append(result)
        except Exception as e:
            logger.error(f"Error in sample processing: {e!s}")

    return results


def save_progress(
    translated_data: HallucinationData,
    output_file: Path,
    dataset: str,
    target_lang: str,
    output_dir: Path,
):
    """Save progress to file with backup handling.

    :param translated_data: Data to save
    :param output_file: Primary output file
    :param dataset: Dataset name for backup file
    :param target_lang: Target language for backup file
    :param output_dir: Output directory for backup file
    """
    try:
        output_file.write_text(json.dumps(translated_data.to_json(), indent=2))
    except Exception as e:
        logger.error(f"Error saving progress: {e!s}")
        # Try to save to a backup file
        backup_file = (
            output_dir / f"{dataset}_data_{target_lang.lower()}_backup_{int(time.time())}.json"
        )
        try:
            backup_file.write_text(json.dumps(translated_data.to_json(), indent=2))
            logger.info(f"Saved backup to {backup_file}")
        except Exception as e2:
            logger.error(f"Error saving backup: {e2!s}")


def main(
    input_dir: Path,
    output_dir: Path,
    model: str,
    source_lang: str,
    target_lang: str,
    dataset: str = "ragtruth",
    batch_size: int = 5,
    max_workers: int = 5,
    resume: bool = True,
    test: bool = False,
):
    """Translates the preprocessed data using parallel processing.

    :param input_dir: Path to the input directory
    :param output_dir: Path to the output directory
    :param model: OpenAI model to use
    :param source_lang: Source language code
    :param target_lang: Target language code
    :param dataset: Dataset name (ragtruth, ragbench, etc.)
    :param batch_size: Number of samples to process in each batch
    :param max_workers: Maximum number of worker threads
    :param resume: Whether to resume from previous run
    :param test: Test mode, only translate 1 sample
    """
    # Set up directories
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Set up logging
    setup_logging(output_dir)

    # Set up files
    input_file = input_dir / f"{dataset}_data.json"
    output_file = output_dir / f"{dataset}_data_{target_lang.lower()}.json"
    log_file = output_dir / "error_log.txt"

    # Check input file
    if not input_file.exists():
        logger.error(f"Input file not found: {input_file}")
        raise FileNotFoundError(f"Input file not found: {input_file}")

    # Load data
    try:
        data = HallucinationData.from_json(json.loads(input_file.read_text()))
    except Exception as e:
        logger.error(f"Error loading input data: {e!s}")
        raise

    # Load existing translated data if resume is enabled
    if resume and output_file.exists():
        translated_data = load_check_existing_data(output_file=output_file)
        num_processed = len(translated_data.samples)
        logger.info(f"Resuming from {num_processed} previously translated samples")
    else:
        translated_data = HallucinationData(samples=[])
        num_processed = 0

    # Get samples to translate
    remaining_samples = data.samples[num_processed:]
    total_samples = len(remaining_samples)

    if total_samples == 0:
        logger.info("No samples to translate. Exiting.")
        return

    # Get OpenAI client
    client = get_openai_client()

    logger.info(f"Translating {dataset} from {source_lang} to {target_lang}")
    logger.info(f"Using model: {model}")
    logger.info(f"Total samples to process: {total_samples}")
    logger.info(f"Batch size: {batch_size}, Max workers: {max_workers}")

    # Create progress bar
    progress_bar = tqdm.tqdm(total=total_samples, desc="Translating")

    # Start time
    start_time = time.time()
    save_interval = 60  # Save at least every minute
    last_save_time = start_time

    try:
        # Process samples in batches
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            for i in range(0, total_samples, batch_size):
                if test and i > 0:
                    break

                batch = remaining_samples[i : i + batch_size]

                # Process the whole batch using the existing executor
                batch_results = process_batch(
                    batch,
                    client,
                    model,
                    num_processed + i,
                    log_file,
                    source_lang,
                    target_lang,
                    dataset,
                    executor,
                )

                # Add results to translated data
                translated_data.samples.extend(batch_results)
                progress_bar.update(len(batch_results))

                # Save progress periodically or at end of batch
                current_time = time.time()
                if current_time - last_save_time > save_interval or i + batch_size >= total_samples:
                    save_progress(translated_data, output_file, dataset, target_lang, output_dir)
                    last_save_time = current_time

                # Calculate and log progress
                current_count = len(translated_data.samples)
                elapsed_time = current_time - start_time
                samples_per_sec = current_count / elapsed_time if elapsed_time > 0 else 0

                logger.info(
                    f"Processed {current_count}/{num_processed + total_samples} samples "
                    f"({samples_per_sec:.2f} samples/sec)"
                )

    except KeyboardInterrupt:
        logger.info("Translation interrupted by user. Saving progress...")
        save_progress(translated_data, output_file, dataset, target_lang, output_dir)
        logger.info(f"Saved {len(translated_data.samples)} translated samples to {output_file}")
        return

    except Exception as e:
        logger.error(f"Unexpected error: {e!s}")
        save_progress(translated_data, output_file, dataset, target_lang, output_dir)
        raise

    finally:
        progress_bar.close()

    logger.info(f"Translation complete. Translated {len(translated_data.samples)} samples.")
    logger.info(f"Output saved to {output_file}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Translate hallucination dataset to another language"
    )
    parser.add_argument(
        "--input_dir", type=str, required=True, help="Directory containing input data files"
    )
    parser.add_argument(
        "--output_dir", type=str, required=True, help="Directory to save output files"
    )
    parser.add_argument(
        "--model", type=str, default="gpt-4o-mini", help="OpenAI model to use for translation"
    )
    parser.add_argument(
        "--source-lang",
        type=str,
        default="EN",
        help="Source language code (e.g., EN, DE, FR, etc.)",
    )
    parser.add_argument(
        "--target-lang",
        type=str,
        default="DE",
        help="Target language code (e.g., EN, DE, FR, etc.)",
    )
    parser.add_argument(
        "--dataset", type=str, default="ragtruth", help="Dataset name (ragtruth, ragbench, etc.)"
    )
    parser.add_argument(
        "--batch-size", type=int, default=5, help="Number of samples to process in each batch"
    )
    parser.add_argument(
        "--max-workers", type=int, default=5, help="Maximum number of worker threads"
    )
    parser.add_argument(
        "--no-resume", action="store_true", help="Don't resume from previous run, start fresh"
    )
    parser.add_argument("--test", action="store_true", help="Test mode, only translate 1 sample")
    args = parser.parse_args()
    main(
        Path(args.input_dir),
        Path(args.output_dir),
        args.model,
        args.source_lang,
        args.target_lang,
        args.dataset,
        args.batch_size,
        args.max_workers,
        not args.no_resume,
        args.test,
    )
