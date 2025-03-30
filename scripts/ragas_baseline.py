import argparse
import asyncio
import json
import os
import re
from pathlib import Path

from datasets import load_dataset
from langchain.chat_models import ChatOpenAI
from openai import OpenAI
from ragas.dataset_schema import SingleTurnSample
from ragas.llms import LangchainLLMWrapper
from ragas.metrics import Faithfulness, FaithfulnesswithHHEM
from torch.utils.data import DataLoader

from lettucedetect.datasets.hallucination_dataset import (HallucinationData,
                                                          HallucinationSample)


def get_api_key() -> str:
    """Get OpenAI client configured from environment variables.

    :return: Open AI API Key
    :raises ValueError: If API key is not set
    """
    api_key = os.getenv("OPENAI_API_KEY") or "EMPTY"
    if api_key == "EMPTY":
        raise ValueError("Provide an OpenAI API key.")
    return api_key


def split_prompt(sample):
    if sample["task_type"] == "Summary":
        user_input = sample["prompt"].split(":")[0] if ":" in sample["prompt"] else sample.task_type
        retrieved_contexts = (
            sample["prompt"].split(":")[1:] if ":" in sample["prompt"] else sample["prompt"]
        )
    else:
        user_input = (
            sample["prompt"].split(":")[:2] if ":" in sample["prompt"] else sample.task_type
        )
        retrieved_contexts = (
            sample["prompt"].split(":")[2:] if ":" in sample["prompt"] else sample["prompt"]
        )
    user_input = " ".join(user_input)
    return user_input, retrieved_contexts


def evaluate_metrics(sample, llm):
    user_input, retrieved_contexts = split_prompt(sample)
    sample = SingleTurnSample(
        user_input=user_input,
        response=sample["answer"],
        retrieved_contexts=retrieved_contexts,
    )
    metric = Faithfulness(llm=llm)
    results = {}
    try:
        results["faithfulness"] = metric.single_turn_score(sample)
    except Exception as e:
        results["faithfulness"] = f"Error: {e}"
    return results


def create_sample_baseline(sample, dataset_name, language, llm):
    """Creates a sample of data where the RAGAS faithfullness is stored in the labels list."""
    prompt = sample["prompt"]
    answer = sample["answer"]

    ragas_metrics = evaluate_metrics(sample, llm)
    for threshold in [0.4, 0.5, 0.6, 0.7]:
        ragas_metrics[f"hallucination_{threshold}"] = (
            1 if ragas_metrics["faithfulness"] < threshold else 0
        )
    task_type = sample["task_type"]
    return HallucinationSample(
        prompt, answer, [ragas_metrics], "test", task_type, dataset_name, language
    )


def load_check_existing_data(output_file: Path) -> HallucinationData:
    """Load existing data or create new data.
    :param output_file: Path to the output file
    :return: Existing HallucinationData or new empty HallucinationData
    """
    if output_file.exists():
        try:
            return HallucinationData.from_json(json.loads(output_file.read_text()))
        except (json.JSONDecodeError, KeyError) as e:
            return HallucinationData(samples=[])
    else:
        return HallucinationData(samples=[])


def main(
    input_dir: Path,
    output_dir: Path,
    dataset_name: str,
    split: str,
    lang: str,
):
    """Calculates RAGAS metrics for each sample.

    :param input_dir: HuggingFace directory.
    :param output_dir: Path to the output directory.
    :param split: Split of dataset the baseline should be created for.
    :param lang: Language of the dataset.
    """

    output_dir = Path(output_dir)
    output_file = output_dir / f"{dataset_name}_ragas_baseline.json"

    hallu_data = load_dataset(input_dir)
    samples = [sample for sample in hallu_data[split]]

    hallu_data_ragas = load_check_existing_data(output_file=output_file)
    num_processed = len(hallu_data_ragas.samples)
    total_samples = len(hallu_data_ragas.samples)

    llm = LangchainLLMWrapper(
        ChatOpenAI(model="gpt-3.5-turbo", openai_api_key=get_api_key(), temperature=0)
    )

    for i, sample in enumerate(samples, start=num_processed):
        print("--------", i, "--------")
        sample_ragas = create_sample_baseline(sample, dataset_name, lang, llm)
        hallu_data_ragas.samples.append(sample_ragas)
        if i % 1 == 0 or i == total_samples - 1:
            (output_file).write_text(json.dumps(hallu_data_ragas.to_json(), indent=4))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--dataset_name", type=str, required=True)
    parser.add_argument("--split", type=str, required=True)
    parser.add_argument("--lang", type=str, default="de")

    args = parser.parse_args()

    main(args.input_dir, args.output_dir, args.dataset_name, args.split, args.lang)
