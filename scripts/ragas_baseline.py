import argparse
import json
import os
from pathlib import Path

from langchain_community.chat_models import ChatOpenAI
from ragas.dataset_schema import SingleTurnSample
from ragas.llms import LangchainLLMWrapper
from ragas.metrics import Faithfulness

from lettucedetect.datasets.hallucination_dataset import HallucinationData, HallucinationSample


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
    if sample.task_type == "Summary":
        user_input = sample.prompt.split(":")[0] if ":" in sample.prompt else sample.task_type
        retrieved_contexts = sample.prompt.split(":")[1:] if ":" in sample.prompt else sample.prompt
    else:
        user_input = sample.prompt.split(":")[:2] if ":" in sample.prompt else sample.task_type
        retrieved_contexts = sample.prompt.split(":")[2:] if ":" in sample.prompt else sample.prompt
    user_input = " ".join(user_input)
    return user_input, retrieved_contexts


def evaluate_metrics(sample, llm):
    user_input, retrieved_contexts = split_prompt(sample)
    sample = SingleTurnSample(
        user_input=user_input,
        response=sample.answer,
        retrieved_contexts=retrieved_contexts,
    )
    metric = Faithfulness(llm=llm)
    results = {}
    try:
        results["faithfulness"] = metric.single_turn_score(sample)
    except Exception as e:
        results["faithfulness"] = f"Error: {e}"
    return results


def create_sample_baseline(sample, llm):
    """Creates a sample of data where the RAGAS faithfullness is stored in the labels list."""
    prompt = sample.prompt
    answer = sample.answer

    ragas_metrics = evaluate_metrics(sample, llm)
    for threshold in [0.4, 0.5, 0.6, 0.7]:
        ragas_metrics[f"threshold_{threshold}"] = (
            1 if ragas_metrics["faithfulness"] < threshold else 0
        )
    task_type = sample.task_type
    dataset = sample.dataset
    split = sample.split
    language = sample.language
    return HallucinationSample(prompt, answer, [ragas_metrics], split, task_type, dataset, language)


def load_check_existing_data(output_file: Path) -> HallucinationData:
    """Load existing data or create new data.
    :param output_file: Path to the output file
    :return: Existing HallucinationData or new empty HallucinationData
    """
    if output_file.exists():
        try:
            return HallucinationData.from_json(json.loads(output_file.read_text()))
        except (json.JSONDecodeError, KeyError):
            return HallucinationData(samples=[])
    else:
        return HallucinationData(samples=[])


def main(
    input_file: Path,
    output_file: Path,
):
    """Creates RAGAS baseline for each sample.

    :param input_dir: Path to the input file.
    :param output_dir: Path to the output file.

    """
    input_file = Path(input_file)
    output_file = Path(output_file)

    hallucination_data = HallucinationData.from_json(json.loads(input_file.read_text()))
    samples = [sample for sample in hallucination_data.samples if sample.split == "test"]

    hallucination_data_ragas = load_check_existing_data(output_file=output_file)
    num_processed = len(hallucination_data_ragas.samples)
    total_samples = len(hallucination_data_ragas.samples)

    llm = LangchainLLMWrapper(
        ChatOpenAI(model="gpt-4o", openai_api_key=get_api_key(), temperature=0)
    )

    for i, sample in enumerate(samples, start=num_processed):
        print("--------", i, "--------")
        sample_ragas = create_sample_baseline(sample, llm)
        hallucination_data_ragas.samples.append(sample_ragas)
        if i % 50 == 0 or i == total_samples - 1:
            (output_file).write_text(json.dumps(hallucination_data_ragas.to_json(), indent=4))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_file", type=str, required=True)
    parser.add_argument("--output_file", type=str, required=True)

    args = parser.parse_args()

    main(args.input_file, args.output_file)
