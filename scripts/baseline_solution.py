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
        Below is given the original question and the facts needed to answer the question.
        {sample.prompt}
        Below is the answer to the question:
        {sample.answer}
        Your task is to determine whether the answer contains either or both of the following two types of hallucinations:
        1. conflict: instances where the answer presents direct contraction or opposition to the original facts;
        2. baseless info: instances where the generated answer includes information which is not inferred from the original facts.
        Then, compile the labeled hallucinated spans into a JSON dict, with a key "hallucination list" and its value is a list of
        hallucinated spans. Do not assume hallucination spans are present in every case.
        If there exist potential hallucinations, the output should be in the following JSON format: {{"hallucination
        list": [hallucination span1, hallucination span2, ...]}}.If there are no hallucinations,return an empty list, as following: {{"hallucination
        list": []}}.
        Output:
    )"""
    response = client.chat.completions.create(
        model="gpt-4-turbo",
        messages=[
            {
                "role": "system",
                "content": "You are a helpful assistant who can identify hallucination spans.",
            },
            {"role": "user", "content": prompt},
        ],
        temperature=0,
    )
    return response.choices[0].message.content


def create_labels(sample, chat_response):
    labels = []
    answer = sample.answer
    print(chat_response)
    for hal in chat_response["hallucination list"]:
        # print("HALLLLLLL", hal)
        match = re.search(re.escape(hal), answer)
        if match:
            labels.append({"start": match.start(), "end": match.end()})
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
    test_samples = [sample for sample in rag_truth_data_de.samples if sample.split == "test"]
    rag_truth_data_base = RagTruthData(samples=[])
    total_samples = len(rag_truth_data_de.samples)

    for i, sample in enumerate(test_samples[:20]):
        print(i)
        sample_de = create_sample_baseline(sample)
        rag_truth_data_base.samples.append(sample_de)
        if i % 5 == 0 or i == total_samples - 1:
            (output_dir / "rag_truth_data_base_test.json").write_text(
                json.dumps(rag_truth_data_base.to_json(), indent=4)
            )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)

    args = parser.parse_args()

    main(args.input_dir, args.output_dir)
