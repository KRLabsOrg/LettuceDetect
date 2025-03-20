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
        <task>
        You will act as an expert annotator to evaluate an answer against a provided source text.
        The source text will be given within <source>... </source> XML tags.
        The answer  will be given within <answer>... </answer> XML tags.

        For each answer, follow these steps:

        Step 1: Read and fully understand the answer. The answer is a text containing information related to the source text but it might also contain information not provided in the source text.
        Step 2: Thoroughly analyze how the answer relates to the information in the source text. Then write your reasoning in 1-3 sentences
        to determine whether the answer contains hallucinations. Hallucinations are sentences that contain one of the following information:
            a. conflict: instances where the answer presents direct contraction or opposition to the original facts;
            b. baseless info: instances where the generated answer includes information which is not inferred from the original facts.

        Step 3: Determine which sentence of the answer is an hallucination. Not every answer contains hallucinations.
        Step 4: Compile the labeled hallucinated spans found into a JSON dict, with a key "hallucination list" and its value is a list of
        hallucinated spans. If there exist potential hallucinations, the output should be in the following JSON format: {{"hallucination
        list": [hallucination span1, hallucination span2, ...]}}.In case of no hallucinations, please output an empty list : {{"hallucination
        list": []}}.
        Return *ONLY* the JSON dict.
     
        </task>

        <example>
        Given below is an example for you to comprehend the task. It guides you in identifying hallucinations.

        Source: What is the capital of France? What is the population of France? France is a country in Europe. The capital of France is Paris. The population of France is 67 million.
        Answer: The capital of France is Paris. The population of France is 69 million.

        1.The answer states that Paris is capital of France.- This matches the fact and is correct.
        2.The answer states that the population of France is 69 million. This condradicts the fact that the population is actually 67 million. 
        Hallucination -> "The population of France is 69 million."
        Therefore, output only the JSON dict {{"hallucination list": ["The population of France is 69 million." ]}}
        </example>
        \n 

        <source>
        {sample.prompt}
        </source>
        \n 
        <answer>
        {sample.answer}
        </answer>
       
    )"""

    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {
                "role": "developer",
                "content": "You are a helpful assistant.",
            },
            {"role": "user", "content": prompt},
        ],
        temperature=0,
    )
    print(response.choices[0].message)
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
    output_file = output_dir / "ragtruth_data_chatgpt4o.json"

    rag_truth_data_de = RagTruthData.from_json(json.loads(input_file.read_text()))
    test_samples = [sample for sample in rag_truth_data_de.samples if sample.split == "test"]

    rag_truth_data_gpt = load_check_existing_data(output_file=output_file)
    num_processed = len(rag_truth_data_gpt.samples)
    total_samples = len(rag_truth_data_gpt.samples)

    for i, sample in enumerate(test_samples, start=num_processed):
        print("--------",i,"--------")
        sample_gpt = create_sample_baseline(sample)
        rag_truth_data_gpt.samples.append(sample_gpt)
        if i % 1 == 0 or i == total_samples - 1:
            (output_dir / "ragtruth_data_chatgpt4o.json").write_text(
                json.dumps(rag_truth_data_gpt.to_json(), indent=4)
            )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)

    args = parser.parse_args()

    main(args.input_dir, args.output_dir)
