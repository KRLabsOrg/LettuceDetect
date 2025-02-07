import argparse
from dataclasses import dataclass
import json
from pathlib import Path
from typing import Literal


@dataclass
class RagTruthSample:
    prompt: str
    answer: str
    labels: list[dict]
    split: Literal["train", "dev", "test"]

    def to_json(self) -> dict:
        return {
            "prompt": self.prompt,
            "answer": self.answer,
            "labels": self.labels,
            "split": self.split,
        }

    @classmethod
    def from_json(cls, json_dict: dict) -> "RagTruthSample":
        return cls(
            prompt=json_dict["prompt"],
            answer=json_dict["answer"],
            labels=json_dict["labels"],
            split=json_dict["split"],
        )


@dataclass
class RagTruthData:
    samples: list[RagTruthSample]

    def to_json(self) -> list[dict]:
        return [sample.to_json() for sample in self.samples]

    @classmethod
    def from_json(cls, json_dict: list[dict]) -> "RagTruthData":
        return cls(
            samples=[RagTruthSample.from_json(sample) for sample in json_dict],
        )


def load_data(input_dir: Path) -> tuple[list[dict], list[dict]]:
    """Load the RAG truth data.

    :param input_dir: Path to the input directory.
    """
    responses = [
        json.loads(line)
        for line in (input_dir / "response.jsonl").read_text().splitlines()
    ]
    sources = [
        json.loads(line)
        for line in (input_dir / "source_info.jsonl").read_text().splitlines()
    ]

    return responses, sources


def create_sample(response: dict, source: dict) -> RagTruthSample:
    """Create a sample from the RAG truth data.

    :param response: The response from the RAG truth data.
    :param source: The source from the RAG truth data.
    """
    prompt = source["prompt"]

    answer = response["response"]
    split = response["split"]

    labels = []

    for label in response["labels"]:
        start_char = label["start"]
        end_char = label["end"]
        labels.append(
            {
                "start": start_char,
                "end": end_char,
                "label": label["label_type"],
            }
        )

    return RagTruthSample(prompt, answer, labels, split)


def main(input_dir: Path, output_dir: Path):
    """Preprocess the RAG truth data.

    :param input_dir: Path to the input directory.
    :param output_dir: Path to the output directory.
    """

    input_dir = Path(input_dir)
    output_dir = Path(output_dir)

    responses, sources = load_data(input_dir)
    sources_by_id = {source["source_id"]: source for source in sources}

    rag_truth_data = RagTruthData(samples=[])

    for response in responses:
        sample = create_sample(response, sources_by_id[response["source_id"]])
        rag_truth_data.samples.append(sample)

    (output_dir / "ragtruth_data.json").write_text(
        json.dumps(rag_truth_data.to_json(), indent=4)
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    args = parser.parse_args()

    main(args.input_dir, args.output_dir)
