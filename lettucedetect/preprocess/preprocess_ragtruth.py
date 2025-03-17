import argparse
import json
from pathlib import Path

from lettucedetect.datasets.hallucination_dataset import HallucinationData, HallucinationSample


def load_data(input_dir: Path) -> tuple[list[dict], list[dict]]:
    """Load the RAG truth data.

    :param input_dir: Path to the input directory.
    """
    responses = [
        json.loads(line) for line in (input_dir / "response.jsonl").read_text().splitlines()
    ]
    sources = [
        json.loads(line) for line in (input_dir / "source_info.jsonl").read_text().splitlines()
    ]

    return responses, sources


def create_sample(response: dict, source: dict) -> HallucinationSample:
    """Create a sample from the RAG truth data.

    :param response: The response from the RAG truth data.
    :param source: The source from the RAG truth data.
    """
    prompt = source["prompt"]

    answer = response["response"]
    split = response["split"]
    task_type = source["task_type"]
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

    return HallucinationSample(prompt, answer, labels, split, task_type, "ragtruth", "en")


def main(input_dir: Path, output_dir: Path):
    """Preprocess the RAG truth data.

    :param input_dir: Path to the input directory.
    :param output_dir: Path to the output directory.
    """
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)

    responses, sources = load_data(input_dir)
    sources_by_id = {source["source_id"]: source for source in sources}

    hallucination_data = HallucinationData(samples=[])

    for response in responses:
        sample = create_sample(response, sources_by_id[response["source_id"]])
        hallucination_data.samples.append(sample)

    (output_dir / "ragtruth_data.json").write_text(
        json.dumps(hallucination_data.to_json(), indent=4)
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    args = parser.parse_args()

    main(args.input_dir, args.output_dir)
