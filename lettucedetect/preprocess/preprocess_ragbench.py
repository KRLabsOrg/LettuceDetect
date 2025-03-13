import argparse
import json
import re
from pathlib import Path

from datasets import load_dataset

from lettucedetect.datasets.hallucination_dataset import HallucinationData, HallucinationSample


def load_data(hugging_dir: str) -> dict:
    """Load the RAG Bench data.

    :param input_dir: Path to the input directory.
    """
    ragbench = {}
    for dataset in [
        "covidqa",
        "cuad",
        "delucionqa",
        "emanual",
        "expertqa",
        "finqa",
        "hagrid",
        "hotpotqa",
        "msmarco",
        "pubmedqa",
        "tatqa",
        "techqa",
    ]:
        ragbench[dataset] = load_dataset(hugging_dir, dataset)

    return ragbench


def create_labels(response, halucinations):
    labels = []
    resp = " ".join([sentence for label, sentence in response["response_sentences"]])
    for hal in halucinations:
        match = re.search(re.escape(hal), resp)
        labels.append({"start": match.start(), "end": match.end(), "label": "Not supported"})
    return labels


def create_sample(response: dict) -> HallucinationSample:
    """Create a sample from the RAGBench data.

    :param response: The response from the RAG bench data.
    """
    prompt = (
        "Instruction:"
        + "\n"
        + " Answer the question: "
        + response["question"]
        + "\n"
        + "Use only the following information:"
        + "\n".join(response["documents"])
    )
    answer = " ".join([sentence for label, sentence in response["response_sentences"]])
    split = response["dataset_name"].split("_")[1]
    task_type = response["dataset_name"].split("_")[0]
    labels = []
    hallucinations = []
    if len(response["unsupported_response_sentence_keys"]) > 0:
        hallucinations = [
            sentence
            for label, sentence in response["response_sentences"]
            if label in response["unsupported_response_sentence_keys"]
        ]
        labels = create_labels(response, hallucinations)

    return HallucinationSample(prompt, answer, labels, split, task_type, "ragbench", "en")


def get_data_split(data, name, split):
    dataset = data.get(name)
    data_split = dataset.get(split)
    return data_split


def main(input_dir: str, output_dir: Path):
    """Preprocess the RAGBench data.
    param input_dir: Path to HuggingFace directory
    param output_dir: Path to the output directory.
    """
    output_dir = Path(output_dir)
    data = load_data(input_dir)
    hallucination_data = HallucinationData(samples=[])

    for dataset_name in data:
        for split in ["train", "test", "validation"]:
            data_split = get_data_split(data, dataset_name, split)
            for response in data_split:
                if not response["dataset_name"]:
                    continue
                sample = create_sample(response)
                hallucination_data.samples.append(sample)

    (output_dir / "ragbench_data.json").write_text(
        json.dumps(hallucination_data.to_json(), indent=4)
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    args = parser.parse_args()
    main(args.input_dir, args.output_dir)
