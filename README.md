# LettuceDetect 🥬🔍


<p align="center">
  <img src="https://github.com/KRLabsOrg/LettuceDetect/blob/main/assets/lettuce_detective.png?raw=true" alt="LettuceDetect Logo" width="400"/>
  <br><em>Because even AI needs a reality check! 🥬</em>
</p>

LettuceDetect is a lightweight and efficient tool for detecting hallucinations in Retrieval-Augmented Generation (RAG) systems. It identifies unsupported parts of an answer by comparing it to the provided context. The tool is trained and evaluated on the [RAGTruth](https://aclanthology.org/2024.acl-long.585/) dataset and leverages [ModernBERT](https://github.com/AnswerDotAI/ModernBERT) for long-context processing, making it ideal for tasks requiring extensive context windows.

Our models are inspired from the [Luna](https://aclanthology.org/2025.coling-industry.34/) paper which is an encoder-based model and uses a similar token-level approach.

[![PyPI](https://img.shields.io/pypi/v/lettucedetect)](https://pypi.org/project/lettucedetect/)
[![License](https://img.shields.io/badge/License-MIT-blue.svg)](https://opensource.org/licenses/MIT)
[![Hugging Face](https://img.shields.io/badge/🤗-Models-yellow.svg)](https://huggingface.co/KRLabsOrg)
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1Ubca5aMaBGdHtJ1rpqj3Ke9SLEr-PaDn?usp=sharing)
[![arXiv](https://img.shields.io/badge/arXiv-2502.17125-b31b1b.svg)](https://arxiv.org/abs/2502.17125)

## Highlights

- LettuceDetect addresses two critical limitations of existing hallucination detection models:
  - Context window constraints of traditional encoder-based methods
  - Computational inefficiency of LLM-based approaches
- Our models currently **outperforms** all other encoder-based and prompt-based models on the RAGTruth dataset and are significantly faster and smaller 
- Achieves higher score than some fine-tuned LLMs e.g. LLAMA-2-13B presented in [RAGTruth](https://aclanthology.org/2024.acl-long.585/), coming up just short of the LLM fine-tuned in the [RAG-HAT paper](https://aclanthology.org/2024.emnlp-industry.113.pdf)
- We release the code, the model and the tool under the **MIT license**

## Get going  

### Features

- ✨ **Token-level precision**: detect exact hallucinated spans
- 🚀 **Optimized for inference**: smaller model size and faster inference
- 🧠 **4K context window** via ModernBERT
- ⚖️ **MIT-licensed** models & code
- 🤖 **HF Integration**: one-line model loading
- 📦 **Easy to use python API**: can be downloaded from pip and few lines of code to integrate into your RAG system

### Installation

Install from the repository:
```bash
pip install -e .
```

From pip:
```bash
pip install lettucedetect
```

### Quick Start

Check out our models published to Huggingface: 
- lettucedetect-base: https://huggingface.co/KRLabsOrg/lettucedect-base-modernbert-en-v1
- lettucedetect-large: https://huggingface.co/KRLabsOrg/lettucedect-large-modernbert-en-v1

You can get started right away with just a few lines of code.

```python
from lettucedetect.models.inference import HallucinationDetector

# For a transformer-based approach:
detector = HallucinationDetector(
    method="transformer", model_path="KRLabsOrg/lettucedect-base-modernbert-en-v1"
)

contexts = ["France is a country in Europe. The capital of France is Paris. The population of France is 67 million.",]
question = "What is the capital of France? What is the population of France?"
answer = "The capital of France is Paris. The population of France is 69 million."

# Get span-level predictions indicating which parts of the answer are considered hallucinated.
predictions = detector.predict(context=contexts, question=question, answer=answer, output_format="spans")
print("Predictions:", predictions)

# Predictions: [{'start': 31, 'end': 71, 'confidence': 0.9944414496421814, 'text': ' The population of France is 69 million.'}]
```

## Performance

**Example level results**

We evaluate our model on the test set of the [RAGTruth](https://aclanthology.org/2024.acl-long.585/) dataset. Our large model, **lettucedetect-large-v1**, achieves an overall F1 score of 79.22%, outperforming prompt-based methods like GPT-4 (63.4%) and encoder-based models like [Luna](https://aclanthology.org/2025.coling-industry.34.pdf) (65.4%). It also surpasses fine-tuned LLAMA-2-13B (78.7%) (presented in [RAGTruth](https://aclanthology.org/2024.acl-long.585/)) and is competitive with the SOTA fine-tuned LLAMA-3-8B (83.9%) (presented in the [RAG-HAT paper](https://aclanthology.org/2024.emnlp-industry.113.pdf)). Overall, **lettucedetect-large-v1** and **lettucedect-base-v1** are very performant models, while being very effective in inference settings.

The results on the example-level can be seen in the table below.

<p align="center">
  <img src="https://github.com/KRLabsOrg/LettuceDetect/blob/main/assets/example_level_lettucedetect.png?raw=true" alt="Example-level Results" width="800"/>
</p>

**Span-level results**

At the span level, our model achieves the best scores across all data types, significantly outperforming previous models. The results can be seen in the table below. Note that here we don't compare to models, like [RAG-HAT](https://aclanthology.org/2024.emnlp-industry.113.pdf), since they have no span-level evaluation presented.

<p align="center">
  <img src="https://github.com/KRLabsOrg/LettuceDetect/blob/main/assets/span_level_lettucedetect.png?raw=true" alt="Span-level Results" width="800"/>
</p>


## How does it work?

The model is a token-level model that predicts whether a token is hallucinated or not. The model is trained to predict the tokens that are hallucinated in the answer given the context and the question.

```mermaid
flowchart LR
    subgraph Inputs
        Context["**Context**: France is a country in Europe. Population is 67 million."]
        Question["**Question**: What is the capital? What is the population?"]
        Answer["**Answer**: The capital of France is Paris. The population is 69 million."]
    end

    Model["**LettuceDetect**: Token Classification"]
    Tokens["**Token Probabilities**: <br> ... <br> The [0.01] <br> population [0.02] <br> is [0.01] <br> 69 [0.95] <br> million [0.95]"]

    Context --> Model
    Question --> Model
    Answer --> Model
    Model --> Tokens

```

### Training a Model

You need to download the RAGTruth dataset first from [here](https://github.com/ParticleMedia/RAGTruth/tree/main/dataset), then put it under the `data/ragtruth` directory. Then run

```bash
python lettucedetect/preprocess/preprocess_ragtruth.py --input_dir data/ragtruth --output_dir data/ragtruth
```

This will create a `data/ragtruth/ragtruth_data.json` file which contains the processed data.

Then you can train the model with the following command.

```bash
python scripts/train.py \
    --ragtruth-path data/ragtruth/ragtruth_data.json \
    --model-name answerdotai/ModernBERT-base \
    --output-dir output/hallucination_detector \
    --batch-size 4 \
    --epochs 6 \
    --learning-rate 1e-5 
```

We trained our models for 6 epochs with a batch size of 8 on a single A100 GPU.

### Evaluation

You can evaluate the models on each level (example, token and span) and each data-type.

```bash
python scripts/evaluate.py \
    --model_path outputs/hallucination_detector \
    --data_path data/ragtruth/ragtruth_data.json \
    --evaluation_type example_level
```

### Model Output Format

The model can output predictions in two formats:

#### Span Format
```python
[{
    'text': str,        # The hallucinated text
    'start': int,       # Start position in answer
    'end': int,         # End position in answer
    'confidence': float # Model's confidence (0-1)
}]
```

### Token Format
```python
[{
    'token': str,       # The token
    'pred': int,        # 0: supported, 1: hallucinated
    'prob': float       # Model's confidence (0-1)
}]
```

## Streamlit Demo

Check out the Streamlit demo to see the model in action.

Install streamlit:

```bash
pip install streamlit
```

Run the demo:

```bash
streamlit run demo/streamlit_demo.py
```

## Use the Web API

LettuceDetect comes with it's own web API and python client library. To use it, make sure to install the package with the optional API dependencies:

```bash
pip install -e .[api]
```

or

```bash
pip install lettucedetect[api]
```

Start the API server with the `scripts/start_api.py` script:

```bash
python scripts/start_api.py dev  # use "prod" for production environments
```

Usage:

```bash
usage: start_api.py [-h] [--model MODEL] [--method {transformer}] {prod,dev}

Start lettucedetect Web API.

positional arguments:
  {prod,dev}            Choose "dev" for development or "prod" for production
                        environments. The serve script uses "fastapi dev" for "dev" or
                        "fastapi run" for "prod" to start the web server. Additionally
                        when choosing the "dev" mode, python modules can be directly
                        imported from the repositroy without installing the package.

options:
  -h, --help            show this help message and exit
  --model MODEL         Path or huggingface URL to the model. The default value is
                        "KRLabsOrg/lettucedect-base-modernbert-en-v1".
  --method {transformer}
                        Hallucination detection method. The default value is
                        "transformer".
````

Example using the python client library:

```python
from lettucedetect_api.client import LettuceClient

contexts = [
    "France is a country in Europe. "
    "The capital of France is Paris. "
    "The population of France is 67 million.",
]
question = "What is the capital of France? What is the population of France?"
answer = "The capital of France is Paris. The population of France is 69 million."

client = LettuceClient("http://127.0.0.1:8000")
response = client.detect_spans(contexts, question, answer)
print(response.predictions)

# [SpanDetectionItem(start=31, end=71, text=' The population of France is 69 million.', hallucination_score=0.989198625087738)]
```

See `demo/detection_api.ipynb` for more examples.
For async support use the `LettuceClientAsync` class instead.

## License

MIT License - see LICENSE file for details.

## Citation

Please cite the following paper if you use LettuceDetect in your work:

```bibtex
@misc{Kovacs:2025,
      title={LettuceDetect: A Hallucination Detection Framework for RAG Applications}, 
      author={Ádám Kovács and Gábor Recski},
      year={2025},
      eprint={2502.17125},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
      url={https://arxiv.org/abs/2502.17125}, 
}
```
