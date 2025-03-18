import json
from pathlib import Path
from torch.utils.data import DataLoader
import argparse
from lettucedetect.preprocess.preprocess_ragtruth import RagTruthData, RagTruthSample
from pathlib import Path
from datasets import load_dataset
from openai import OpenAI

import re


def ask_chat(sample):
    prompt = f"""
        Below is the original context for a given question:
        {sample.prompt}
        Below is a answer to the original context:
        {sample.answer}
        Your task is to determine whether the answer contains either or both of the following two types of hallucinations:
        1. conflict: instances where the answer presents direct contraction or opposition to the original news;
        2. baseless info: instances where the generated answer includes information which is not substantiated by or inferred from the
        original news.
        Then, compile the labeled hallucinated spans into a JSON dict, with a key "hallucination list" and its value is a list of
        hallucinated spans. Please also include the reason for hallucination which can be either conflict or baseless info.
        If there exist potential hallucinations, the output should be in the following JSON format: {{"hallucination
        list": [[hallucination span1, reason1 ], [hallucination span2, reason2], ...]}}. Otherwise, leave the value as a empty list as following: {{"hallucination
        list": []}}.
        Output:
    )"""
    response = client.chat.completions.create(
        model="gpt-4-turbo",
        messages=[
            {
                "role": "developer",
                "content": "You are a helpful assistant who can identify hallucination spans.",
            },
            {"role": "user", "content": prompt},
        ],
        temperature=0,
    )
    print(response)
    return response.choices[0].message.content


def create_labels(sample, chat_response):
    print(chat_response)
    labels = []
    answer = sample.answer
    print(chat_response["hallucination list"])
    for hal in chat_response["hallucination list"]:
        print(hal)
        match = re.search(re.escape(hal[0]), answer)
        labels.append({"start": match.start(), "end": match.end(), "label": hal[1]})
    return labels


def create_sample_baseline(sample):
    prompt = sample.prompt
    answer = sample.answer
    split = sample.split
    chat_response = ask_chat(sample)
    chat_response = json.loads(chat_response)
    labels = create_labels(sample, chat_response)
    task_type = sample.task_type
    return RagTruthSample(prompt, answer, labels, split, task_type)


def main(input_dir: Path, output_dir: Path):
    """Translates the already preprocessed RAG Truth Data

    :param input_dir: Path to the input directory.
    :param output_dir: Path to the output directory.
    """
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)
    input_file = input_dir / "ragtruth_data_de.json"
    rag_truth_data_de = RagTruthData.from_json(json.loads(input_file.read_text()))
    rag_truth_data_base = RagTruthData(samples=[])
    total_samples = len(rag_truth_data_de.samples)

    for i, sample in enumerate(rag_truth_data_de.samples[:20]):
        sample_de = create_sample_baseline(sample)
        rag_truth_data_base.samples.append(sample_de)
        if i % 3 == 0 or i == total_samples - 1:
            (output_dir / "rag_truth_data_base.json").write_text(
                json.dumps(rag_truth_data_base.to_json(), indent=4)
            )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)

    args = parser.parse_args()

    main(args.input_dir, args.output_dir)
