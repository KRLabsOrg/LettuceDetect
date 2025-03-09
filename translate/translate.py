import argparse
import json
import re
from pathlib import Path

from vllm import LLM
from vllm.sampling_params import SamplingParams

from lettucedetect.preprocess.preprocess_ragtruth import RagTruthData, RagTruthSample


def translate_text(text, model, sampling_params, source_lang="EN", target_lang="DE", hal=False):
    if hal:
        translation_prompt = f"""Translate the following text from {source_lang} to {target_lang}.  
    - If the original text contains <HAL> tags, translate the content inside <HAL> tags and ensure the number of the <HAL> tags remain exactly the same in the output.
    - Do NOT add any <HAL>  tags if they were not in the original text.
    - Do NOT remove any <HAL>  tags that were in the original text.
    - Do not include any additional sentences summarizing or explaining the translation.  

    {source_lang}: {text}  
    {target_lang}:  
    """
    else:
        translation_prompt = f"""Translate the following text from {source_lang} to {target_lang}.  
    - Translate only the given text.
    - Do not include any additional sentences summarizing or explaining the translation.  

    {source_lang}: {text}  
    {target_lang}:  
    """

    system_prompt = f"You are an expert linguist, specializing in translation from {source_lang} to {target_lang}. Translate only the given text."

    messages = [
        {"role": "system", "content": system_prompt},
        {
            "role": "user",
            "content": translation_prompt,
        },
    ]

    res = model.chat(messages=messages, sampling_params=sampling_params)
    return res[0].outputs[0].text


def merge_overlapping_spans(labels):
    """Merge overlapping hallucination spans into a single span."""
    if not labels:
        return []
    labels.sort(key=lambda x: x["start"])
    new_labels = []
    current_span = labels[0]
    for span in labels[1:]:
        if span["start"] <= current_span["end"]:
            current_span["end"] = max(current_span["end"], span["end"])
        else:
            new_labels.append(current_span)
            current_span = span

    new_labels.append(current_span)
    return new_labels


def put_hallucination_tags(sample, answer):
    labels = merge_overlapping_spans(sample.labels)
    labels = sorted(labels, key=lambda x: (x["end"], x["start"]), reverse=True)
    for label in labels:
        start, end = label["start"], label["end"]
        answer = answer[:end] + "<HAL>" + answer[end:]
        answer = answer[:start] + "<HAL>" + answer[start:]

    return answer, labels


def find_hallucination_tags(text, labels, i, log_file):
    pattern = r"<HAL>(.*?)<HAL>"
    hal_spans = []
    j = 0
    with open(log_file, "a") as log:
        for span in re.finditer(pattern, text):
            start = span.start(1)
            end = span.end(1)
            if j < len(labels):
                label = labels[j]["label"]
            else:
                label = "Unknown"
                log.write(f"IndexError: No label for hallucinated text at sample ({i})\n")
            hal_spans.append((start, end, label))
            j += 1
    return hal_spans


def create_sample_de(dict):
    """Create a sample from the RAG truth data.

    :param response: The response from the RAG truth data.
    :param source: The source from the RAG truth data.
    """
    prompt = dict["prompt"]

    answer = dict["answer"]
    split = dict["split"]
    labels = []

    for label in dict["labels"]:
        start_char = label["start"]
        end_char = label["end"]
        labels.append(
            {
                "start": start_char,
                "end": end_char,
                "label": label["label"],
            }
        )
    task_type = dict["task_type"]
    return RagTruthSample(prompt, answer, labels, split, task_type)


def translate_sample(sample, model, sampling_params, i, log_file):
    """Translate each sample of the RAG truth data."""
    hal = len(sample.labels) > 0
    dict_de = {}
    dict_de["prompt"] = translate_text(sample.prompt, model, sampling_params)
    answer, labels = put_hallucination_tags(sample, sample.answer)
    dict_de["answer"] = translate_text(answer, model, sampling_params, hal=hal)
    dict_de["split"] = sample.split
    dict_de["task_type"] = translate_text(sample.task_type, model, sampling_params)
    dict_de["labels"] = []
    if hal:
        hal_spans = find_hallucination_tags(dict_de["answer"], labels, i, log_file)
        for span in hal_spans:
            start, end, label = span
            dict_de["labels"].append(
                {
                    "start": start,
                    "end": end,
                    "label": translate_text(label, model, sampling_params),
                }
            )

    sample_de = create_sample_de(dict_de)
    return sample_de


def load_check_existing_data(output_file):
    if output_file.exists():
        return RagTruthData.from_json(json.loads(output_file.read_text()))
    else:
        return RagTruthData(samples=[])


def main(input_dir: Path, output_dir: Path):
    """Translates the already preprocessed RAG Truth Data

    :param input_dir: Path to the input directory.
    :param output_dir: Path to the output directory.
    """
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)
    input_file = input_dir / "ragtruth_data.json"
    output_file = output_dir / "ragtruth_data_de.json"
    log_file = output_dir / "error_log.txt"
    rag_truth_data = RagTruthData.from_json(json.loads(input_file.read_text()))

    rag_truth_data_de = load_check_existing_data(output_file=output_file)
    num_processed = len(rag_truth_data_de.samples)
    total_samples = len(rag_truth_data.samples)

    model_name = "mistralai/Mistral-7B-Instruct-v0.3"
    sampling_params = SamplingParams(
        max_tokens=3000,
        seed=1111,
    )
    model = LLM(model=model_name)

    for i, sample in enumerate(rag_truth_data.samples[num_processed:], start=num_processed):
        sample_de = translate_sample(sample, model, sampling_params, i, log_file)
        rag_truth_data_de.samples.append(sample_de)
        if i % 50 == 0 or i == total_samples - 1:
            (output_dir / "ragtruth_data_de.json").write_text(
                json.dumps(rag_truth_data_de.to_json(), indent=4)
            )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)

    args = parser.parse_args()

    main(args.input_dir, args.output_dir)
