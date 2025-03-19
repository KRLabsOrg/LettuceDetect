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
    '''
    prompt = f"""
        Unten findest du die ursprüngliche Frage und die dazugehörigen Fakten:
        {sample.prompt}
        Unten ist die Antwort auf die Frage:
        {sample.answer}
        Deine Aufgabe ist es zu bestimmen, ob die Antwort eine oder beider der folgenden Arten von Halluzinationen enthält:
        1. Widerspruch: Fälle, in denen die Antwort einen direkten Gegensatz zu den ursprünglichen Fakten darstellt
        2. Unbegründete Information: Fälle, in denen die Antwort Informationen enthält die nicht aus den ursprünglichen Fakten abgeleitet werden können
        Nicht immer enthält eine Antwort Halluzinationen.
        Du solltest du die Halluzinationen in einer JSON-Datenstruktur zusammenfassen.
        Verwende dazu das folgende Format:
        Falls Halluzinationen vorhanden sind, soll das JSON-Format wie folgt aussehen:
        {{"hallucination list": [hallucination span1, hallucination span2, ...]}}
        Falls keine Halluzinationen vorhanden sind, gib eine leere Liste zurück:
        {{"hallucination list": []}}
        Ausgabe:
    )"""
    '''

    prompt = f"""
        Below is given the original question and the facts needed to answer the question in german:
        {sample.prompt}
        Below is the answer to the question in german:
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
        match = re.search(re.escape(hal), answer)
        if match:
            labels.append({"start": match.start(), "end": match.end(), "text": hal})
    return labels


def create_sample_baseline(sample):
    """Creates a sample where the annotations / labels are based on the ChatGPT responses."""

    prompt = sample.prompt
    answer = sample.answer
    split = sample.split
    chat_response = ask_chat(sample)
    try:
        chat_response = json.loads(chat_response.strip())
    except json.JSONDecodeError:
        chat_response = {"hallucination list": []}
    labels = create_labels(sample, chat_response)
    task_type = sample.task_type
    return RagTruthSample(prompt, answer, labels, split, task_type)


def load_check_existing_data(output_file):
    if output_file.exists():
        return RagTruthData.from_json(json.loads(output_file.read_text()))
    else:
        return RagTruthData(samples=[])


def main(input_dir: Path, output_dir: Path):
    """Prompts ChatGPT to find hallucination spans in the german samples and saves the response in a new json file.

    :param input_dir: Path to the input directory.
    :param output_dir: Path to the output directory.
    """
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)
    input_file = input_dir / "ragtruth_data_de.json"
    output_file = output_dir / "ragtruth_data_chatgpt.json"

    rag_truth_data_de = RagTruthData.from_json(json.loads(input_file.read_text()))
    test_samples = [sample for sample in rag_truth_data_de.samples if sample.split == "test"]

    rag_truth_data_gpt = load_check_existing_data(output_file=output_file)
    num_processed = len(rag_truth_data_gpt.samples)
    total_samples = len(rag_truth_data_gpt.samples)

    for i, sample in enumerate(test_samples[num_processed:], start=num_processed):
        print(i)
        sample_gpt = create_sample_baseline(sample)
        rag_truth_data_gpt.samples.append(sample_gpt)
        if i % 10 == 0 or i == total_samples - 1:
            (output_dir / "ragtruth_data_chatgpt.json").write_text(
                json.dumps(rag_truth_data_gpt.to_json(), indent=4)
            )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)

    args = parser.parse_args()

    main(args.input_dir, args.output_dir)
