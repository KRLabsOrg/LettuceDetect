import argparse
import json
from pathlib import Path

from datasets import load_dataset
from lettucedetect.datasets.hallucination_dataset import HallucinationData, HallucinationSample

from datasets import load_dataset


def create_sample(sample, split):
    prompt = sample["prompt"]
    answer = sample["answer"]
    labels = sample["labels"]
    dataset_name = sample["dataset"]
    task_type = sample["task_type"]
    lang = sample["language"]
    return HallucinationSample(prompt, answer, labels, split, task_type, dataset_name, lang)


def create_samples_list(ds):
    hallucination_data = HallucinationData(samples=[])
    for split in ["train", "test"]:
        hallucination_data.samples.extend([create_sample(sample, split) for sample in ds[split]])
    return hallucination_data


def main(input_dir: str, output_dir: Path):
    """Loads data from HuggingFace and saves it locally.

    :param input_dir: HuggingFace Directory
    :param output_dir: Path to the output directory.

    """
    input_dir = input_dir
    ds = load_dataset(input_dir)
    output_dir = Path(output_dir)

    hallucination_data = create_samples_list(ds)

    (output_dir / f"{input_dir.split('/')[-1]}.json").write_text(
        json.dumps(hallucination_data.to_json(), indent=4)
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input_dir", type=str, required=True, help="HuggingFace diretory and file name"
    )
    parser.add_argument("--output_dir", type=str, required=True)

    args = parser.parse_args()

    main(args.input_dir, args.output_dir)
