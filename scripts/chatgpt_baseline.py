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
        Step 1: Read and fully understand the answer in german. The answer is a text containing information related to the source text.
        Step 2: Thoroughly analyze how the answer relates to the information in the source text. Determine whether the answer contains hallucinations. Hallucinations are sentences that contain one of the following information:
            a. conflict: instances where the answer presents direct contraction or opposition to the original source.
            b. baseless info: instances where the generated answer includes information which is not inferred from the original source. General knowledge or logical deductions should not be considered hallucinations unless they contradict the source.
        Step 3: Determine whether the answer contains any hallucinations. If no hallucinations are found, return an empty list.
        Step 4: Compile the labeled hallucinated spans found into a JSON dict, with a key "hallucination list" and its value is a list of
        hallucinated spans. If there exist potential hallucinations, the output should be in the following JSON format: {{"hallucination
        list": [hallucination span1, hallucination span2, ...]}}. In case of no hallucinations, please output an empty list : {{"hallucination
        list": []}}.
        Output only the JSON dict.
     
        </task>

        Given below are three examples for you to comprehend the task.
        <example1>
       

        Source: Was ist die Hauptstadt von Frankreich? Wie hoch ist die Bevölkerung Frankreichs? Frankreich ist ein Land in Europa. Die Hauptstadt von Frankreich ist Paris. Die Bevölkerung Frankreichs beträgt 67 Millionen.
        Answer: Die Hauptstadt von Frankreich ist Paris. Die Bevölkerung Frankreichs beträgt 69 Millionen.

        1.The answer states that Paris is capital of France. This matches the source and is correct.
        2.The answer states that the population of France is 69 million. This condradicts the source that the population is actually 67 million. 
        Hallucination -> "The population of France is 69 million."
        Therefore, output only {{"hallucination list": ["Die Bevölkerung Frankreichs beträgt 69 Millionen." ]}}
        </example1>

        <example2>
        Source: Was ist die Hauptstadt von Frankreich? Wie hoch ist die Bevölkerung Frankreichs?  Die Hauptstadt von Frankreich ist Paris. Die Bevölkerung von Frankreich beträgt 67 Millionen.
        Answer: Die Hauptstadt von Frankreich ist Paris. Die Bevölkerung von Frankreich beträgt 67 Millionen, und die Amtssprache ist Spanisch.

        1.The answer states that Paris is capital of France. This matches the source and is correct.
        2.The answer states that the population of France is 69 million. This matches the source and is correct.
        3. The answer states that the language spoken in France is Spanish. This is incorrect and not supported by the source.
        Hallucination -> "die Amtssprache ist Spanisch"
        Therefore, output only {{"hallucination list": ["die Amtssprache ist Spanisch" ]}}

        </example2>

        <example3>
        Source: Was ist die Hauptstadt von Österreich? Wie hoch ist die Bevölkerung Österreich? Österreich ist ein Land in Europa. Die Hauptstadt von Österreich ist Wien. Die Bevölkerung Österreichs beträgt 9.1 Millionen.
        Answer: Die Hauptstadt von Österreich ist Wien. Die Bevölkerung Österreichs beträgt 9.1 Millionen.
        1.The answer states that Vienna is capital of Austria. This matches the source and is correct.
        2.The answer states that the population of Austria is 9.1 million. This matches the source and is correct.
        Hallucination -> No hallucinations found
        Therefore, output only {{"hallucination list": []}}
        </example3>

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
                "role": "system",
                "content": "You are a helpful assistant.",
            },
            {"role": "user", "content": prompt},
        ],
        temperature=0,
    )
    return response.choices[0].message.content


def create_labels(sample, chat_response):
    labels = []
    answer = sample.answer
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
    match = re.search(r"\{.*?\}", chat_response, re.DOTALL)
    try:
        extracted_json = match.group(0)
        extracted_json = json.loads(extracted_json)
    except json.JSONDecodeError:
        chat_response = {"hallucination list": []}
    labels = create_labels(sample, extracted_json)
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
        print("--------", i, "--------")
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
