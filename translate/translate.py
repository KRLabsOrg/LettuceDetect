import json
import os
import re, string
from pathlib import Path
from lettucedetect.models.trainer import Trainer
from lettucedetect.datasets.ragtruth import RagTruthDataset
from lettucedetect.preprocess.preprocess_ragtruth import RagTruthData
from lettucedetect.preprocess.preprocess_ragtruth import RagTruthData, RagTruthSample
from torch.utils.data import DataLoader
import argparse
from transformers import AutoTokenizer
from sentence_transformers import SentenceTransformer
import torch
import numpy as np
import re
from huggingface_hub import snapshot_download
from pathlib import Path
import mistral_common
from mistral_inference.transformer import Transformer
from mistral_inference.generate import generate
from mistral_common.tokens.tokenizers.mistral import MistralTokenizer
from mistral_common.protocol.instruct.messages import UserMessage
from mistral_common.protocol.instruct.request import ChatCompletionRequest



def translate_text(text,model, tokenizer,source_lang = 'EN', target_lang = 'DE', hal =  False):
    #model = Transformer.from_folder(mistral_models_path).to("cuda")

    if hal:
      translation_prompt = f"""Translate the following text from {source_lang} to {target_lang}.  

    - If the original text contains `<HAL>` tags, translate the content inside `<HAL>` tags and put the content again between `<HAL>` tags in the output.  
    - If there are no `<HAL>` tags in the original text, do NOT add any `<HAL>` tags.  
    - Translate the text exactly as it is, without adding any extra commentary, explanations, or conclusions.  
    - Do not include any additional sentences summarizing or explaining the translation.  

    {source_lang}: {text}  
    {target_lang}:  
    """
    else:
       translation_prompt = f"""Translate the following text from {source_lang} to {target_lang}.  
    - Translate the text exactly as it is, without adding any extra commentary, explanations, or conclusions.  
    - Do not include any additional sentences summarizing or explaining the translation.  

    {source_lang}: {text}  
    {target_lang}:  
    """

    completion_request = ChatCompletionRequest(messages=[UserMessage(content=translation_prompt)])

    tokens = tokenizer.encode_chat_completion(completion_request).tokens

    out_tokens, _ = generate([tokens], model, max_tokens=2048, temperature=0.0, eos_id=tokenizer.instruct_tokenizer.tokenizer.eos_id)
    result = tokenizer.instruct_tokenizer.tokenizer.decode(out_tokens[0])

    return (result)

def merge_overlapping_spans(labels):
    """Merge overlapping hallucination spans into a single span."""
    if not labels:
        return []
    labels.sort(key=lambda x: x["start"])  
    new_labels = []
    current_span = labels[0]
    for span in labels[1:]:
        if span["start"] <= current_span["end"]:  
            current_span["end"] = max(current_span["end"], span["end"])  # Extend span
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

   return answer,labels

def create_sample_de(dict) :
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


def find_hallucination_tags(text,labels):
    pattern = r"<HAL>(.*?)<HAL>" 
    hal_spans = []
    i = 0
    for  span in re.finditer(pattern, text):
        start = span.start(1)  # Start of the hallucinated text
        end = span.end(1)      # End of the hallucinated text
        hal_spans.append((start, end, labels[i]['label']))
        i+=1
    return hal_spans

def create_sample_de(dict) :
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
def colab_print(text, max_width = 120):
  words = text.split()
  line = ""
  for word in words:
    if len(line) + len(word) + 1 > max_width:
      print(line)
      line = ""
    line += word + " "
  print (line)

def translate_sample(sample,  model,tokenizer):
    """Translate each sample of the RAG truth data."""
    hal =len(sample.labels) > 0 
    dict_de = {}
    dict_de["prompt"] = translate_text(sample.prompt,model, tokenizer)
    answer,labels = put_hallucination_tags(sample, sample.answer)
    dict_de["answer"] = translate_text(answer,model, tokenizer, hal = hal)
    dict_de["split"] = sample.split
    dict_de["task_type"] = translate_text(sample.task_type,model, tokenizer)
    dict_de["labels"] = []
    colab_print(dict_de["answer"])
    print(labels)
    if hal:
        hal_spans = find_hallucination_tags(dict_de["answer"],labels)
        for span in hal_spans:
            print(span)
            start, end, label = span
            dict_de["labels"].append({
                    "start": start,
                    "end": end,
                    "label": translate_text(label,model, tokenizer),
                })
            print(dict_de["answer"][start:end])
    
    
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
    rag_truth_data = RagTruthData.from_json(json.loads(input_file.read_text()))

    rag_truth_data_de = load_check_existing_data(output_file=output_file )
    mistral_models_path = Path.home().joinpath('mistral_models', '7B-Instruct-v0.3')
    mistral_models_path.mkdir(parents=True, exist_ok=True)
    snapshot_download(repo_id="mistralai/Mistral-7B-Instruct-v0.3", allow_patterns=["params.json", "consolidated.safetensors", "tokenizer.model.v3"], local_dir=mistral_models_path)
    tokenizer = MistralTokenizer.from_file(f"{mistral_models_path}/tokenizer.model.v3")
    model = Transformer.from_folder(mistral_models_path).half().to("cuda")

    num_processed = len(rag_truth_data_de.samples) 
    total_samples = len(rag_truth_data.samples)

    print(f"Continuing on sample {num_processed}/{total_samples}...")

    for i, sample in enumerate(rag_truth_data.samples[num_processed:],start=num_processed):

        print(i)
        sample_de = translate_sample(sample, model, tokenizer)
        rag_truth_data_de.samples.append(sample_de)
        if i % 5 == 0 or i == total_samples - 1:
            (output_dir / "ragtruth_data_de.json").write_text(
                json.dumps(rag_truth_data_de.to_json(), indent=4)
            )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)

    args = parser.parse_args()

    main(args.input_dir, args.output_dir)
