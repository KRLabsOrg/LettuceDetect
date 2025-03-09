import argparse
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from datasets import load_dataset


@dataclass
class RagBenchSample:
    prompt: str
    answer: str
    labels: list[dict]
    split: Literal["train", "dev", "test"]
    task_type: str

    def to_json(self) -> dict:
        return {
            "prompt": self.prompt,
            "answer": self.answer,
            "labels": self.labels,
            "split": self.split,
            "task_type": self.task_type,
        }

    @classmethod
    def from_json(cls, json_dict: dict) -> "RagBenchSample":
        return cls(
            prompt=json_dict["prompt"],
            answer=json_dict["answer"],
            labels=json_dict["labels"],
            split=json_dict["split"],
            task_type=json_dict["task_type"],
        )


@dataclass
class RagBenchData:
    samples: list[RagBenchSample]

    def to_json(self) -> list[dict]:
        return [sample.to_json() for sample in self.samples]

    @classmethod
    def from_json(cls, json_dict: list[dict]) -> "RagBenchSample":
        return cls(
            samples=[RagBenchSample.from_json(sample) for sample in json_dict],
        )


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


def create_sample(response: dict) -> RagBenchSample:
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

    return RagBenchSample(prompt, answer, labels, split, task_type)


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
    rag_bench_data = RagBenchData(samples=[])

    for dataset_name in data:
        for split in ["train", "test", "validation"]:
            data_split = get_data_split(data, dataset_name, split)
            for response in data_split:
                if not response["dataset_name"]:
                    continue
                sample = create_sample(response)
                rag_bench_data.samples.append(sample)

    (output_dir / "ragbench_data.json").write_text(json.dumps(rag_bench_data.to_json(), indent=4))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    args = parser.parse_args()
    main(args.input_dir, args.output_dir)
