import argparse
import asyncio
import json
import re
from pathlib import Path
import os

from datasets import load_dataset
from langchain.chat_models import ChatOpenAI
from openai import OpenAI
from torch.utils.data import DataLoader

from ragas.dataset_schema import SingleTurnSample
from ragas.llms import LangchainLLMWrapper
from ragas.metrics import Faithfulness, FaithfulnesswithHHEM

from lettucedetect.datasets.hallucination_dataset import (
    HallucinationData,
    HallucinationSample,
)


def evaluate_metrics(sample, llm):
    sample = SingleTurnSample(
        user_input=sample.task_type,
        response=sample.answer,
        retrieved_contexts=[sample.prompt],
    )
    metrics = {
        "Faithfulness": Faithfulness(llm=llm),
        "FaithfulnesswithHHEM": FaithfulnesswithHHEM(llm=llm),
    }
    results = {}
    for metric_name, metric in metrics.items():
        try:
            results[metric_name] = metric.single_turn_score(sample)
        except Exception as e:
            results[metric_name] = f"Error: {e}"
    return results


def create_sample_baseline(sample, dataset_name, lang, llm):
    """Creates a sample of data where the RAGAS metrics are stored in the labels list."""
    prompt = sample.prompt
    answer = sample.answer
    split = sample.split
    labels = [evaluate_metrics(sample, llm)]
    task_type = sample.task_type
    return HallucinationSample(prompt, answer, labels, split, task_type, dataset_name, lang)


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


def main(input_dir: Path, output_dir: Path, dataset_name: str, lang: str):
    """Calculates RAGAS metrics for each sample.

    :param input_dir: Path to the input directory.
    :param output_dir: Path to the output directory.
    """
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)
    input_file = input_dir / "ragtruth_data.json"
    output_file = output_dir / "hallu_data_ragas.json"

    hallu_data_de = HallucinationData.from_json(json.loads(input_file.read_text()))
    test_samples = [sample for sample in hallu_data_de.samples if sample.split == "test"]

    llm = LangchainLLMWrapper(
        ChatOpenAI(
            model="gpt-4o-mini",
            openai_api_key= os.environ["OPENAI_API_KEY"] 
        )
    )

    hallu_data_ragas = load_check_existing_data(output_file=output_file)
    num_processed = len(hallu_data_ragas.samples)
    total_samples = len(hallu_data_ragas.samples)

    for i, sample in enumerate(test_samples, start=num_processed):
        print("--------", i, "--------")
        sample_gpt = create_sample_baseline(sample, dataset_name, lang, llm)
        hallu_data_ragas.samples.append(sample_gpt)
        if i % 1 == 0 or i == total_samples - 1:
            (output_dir / "hallu_data_ragas.json").write_text(
                json.dumps(hallu_data_ragas.to_json(), indent=4)
            )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--dataset_name", type=str, required=True)
    parser.add_argument("--lang", type=str, default="de")

    args = parser.parse_args()

    main(args.input_dir, args.output_dir, args.dataset_name, args.lang)
