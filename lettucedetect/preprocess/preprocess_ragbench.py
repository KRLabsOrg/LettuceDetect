import argparse
import json
import re
from pathlib import Path

from datasets import load_dataset

from lettucedetect.datasets.hallucination_dataset import HallucinationData, HallucinationSample

PROMPT_QA = """
Briefly answer the following question:
{question}
Bear in mind that your response should be strictly based on the following {num_passages} passages:
{context}
In case the passages do not contain the necessary information to answer the question, please reply with: "Unable to answer based on given passages."
output:
"""


def load_data(hugging_dir: str) -> dict:
    """Load the RAG Bench data.

    :param hugging_dir: Path to the HuggingFace directory.
    :return: A dictionary of the RAG Bench data.
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


def create_labels(response, hallucinations):
    """Create labels for the RAGBench data.

    :param response: The response from the RAG bench data.
    :param hallucinations: The hallucinations from the RAG bench data.
    :return: A list of labels.
    """
    labels = []
    resp = " ".join([sentence for _, sentence in response["response_sentences"]])
    for hal in hallucinations:
        match = re.search(re.escape(hal), resp)
        labels.append({"start": match.start(), "end": match.end(), "label": "Not supported"})
    return labels


def create_sample(response: dict) -> HallucinationSample:
    """Create a sample from the RAGBench data.

    :param response: The response from the RAG bench data.
    :return: A sample from the RAGBench data.
    """
    context_str = "\n".join(
        [f"passage {i + 1}: {passage}" for i, passage in enumerate(response["documents"])]
    )
    prompt = PROMPT_QA.format(
        question=response["question"],
        num_passages=len(response["documents"]),
        context=context_str,
    )
    answer = " ".join([sentence for _, sentence in response["response_sentences"]])
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


def main(input_dir: str, output_dir: Path):
    """Preprocess the RAGBench data.

    :param input_dir: Path to HuggingFace directory
    :param output_dir: Path to the output directory.
    """
    output_dir = Path(output_dir)
    data = load_data(input_dir)
    hallucination_data = HallucinationData(samples=[])

    for dataset_name in data:
        for split in ["train", "test", "validation"]:
            data_split = data[dataset_name][split]
            for response in data_split:
                if not response["dataset_name"]:
                    continue
                sample = create_sample(response)
                hallucination_data.samples.append(sample)
                break

    (output_dir / "ragbench_data.json").write_text(
        json.dumps(hallucination_data.to_json(), indent=4)
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    args = parser.parse_args()
    main(args.input_dir, args.output_dir)
