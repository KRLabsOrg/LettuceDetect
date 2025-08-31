# 🥬 TinyLettuce: Efficient Hallucination Detection with 17–68M Encoders

<p align="center">
  <img src="https://github.com/KRLabsOrg/LettuceDetect/blob/dev/assets/tinytinylettuce.png?raw=true" alt="TinyLettuce Detective" width="400"/>
  <br>
  <em>Small, task‑specialized encoders trained on synthetic data</em>
</p>

---

We present **TinyLettuce**, our approach to efficient hallucination detection. By training tiny Ettin encoders (17-68M parameters), we achieve better accuracy than billion-parameter LLM judges while running in real-time on CPU.

## TL;DR

- We're releasing a pipeline for generating synthetic training data for hallucination detection and training tiny Ettin encoders on it.
- **TinyLettuce‑17M** (17M parameters) reaches **90.87% F1** 🎯 on synthetic test data, outperforming GPT‑5‑mini (83.69%), GPT‑OSS‑120B (83.38%), and Qwen3‑235B (79.84%)
- Runs in **real-time on CPU** with low latency and large throughput
- **Synthetic data generation** creates training data **significantly cheaper** than manual annotation
- Complete **end‑to‑end pipeline** for domain-specific model training - generate data and train in minutes
- All models and code are **MIT licensed** and ready for production deployment

---

## Quick Links

- **GitHub**: [github.com/KRLabsOrg/LettuceDetect](https://github.com/KRLabsOrg/LettuceDetect)  
- **PyPI**: [pypi.org/project/lettucedetect](https://pypi.org/project/lettucedetect/)
- **Hugging Face Models**:  
  - [TinyLettuce Collection](https://huggingface.co/collections/KRLabsOrg/tinylettuce-68b42a66b8b6aaa4bf287bf4)
- **Notebook/Demo**: [TinyLettuce end‑to‑end](https://github.com/KRLabsOrg/LettuceDetect/blob/main/demo/tinylettuce.ipynb)
- **Ettin Paper (LightOn)**: https://huggingface.co/papers/2507.11412

---

## Quickstart

Install:

```bash
pip install lettucedetect
```

### Detect Hallucinations (Real-time CPU)

Take one of our pre-trained models and use it for detecting hallucinations in your data:

```python
from lettucedetect.models.inference import HallucinationDetector

# Load tiny but powerful model
detector = HallucinationDetector(
    method="transformer", 
    model_path="KRLabsOrg/tinylettuce-ettin-17m-en-v1"
)

# Detect hallucinations in medical context
spans = detector.predict(
    context=[
        "Ibuprofen is an NSAID that reduces inflammation and pain. The typical adult dose is 400-600mg every 6-8 hours, not exceeding 2400mg daily."
    ],
    question="What is the maximum daily dose of ibuprofen?",
    answer="The maximum daily dose of ibuprofen for adults is 3200mg.",
    output_format="spans",
)
print(spans)
# Output: [{"start": 51, "end": 57, "text": "3200mg"}]
```

### Generate Synthetic Training Data

With **lettucedetect**, you can create training data automatically with controllable error types using the HallucinationGenerator class. Generate domain-specific training data with just a few lines of code while controlling error types and intensity.

```python
from lettucedetect import HallucinationGenerator

# Initialize generator (GPT‑5 requires temperature=1.0)
generator = HallucinationGenerator(model="gpt-5-mini", temperature=1.0)

# Generate numerical error
result_medical = generator.generate(
    context=[
        "Ibuprofen is an NSAID that reduces inflammation and pain. The typical adult dose is 400-600mg every 6-8 hours, not exceeding 2400mg daily."
    ],
    question="What is the maximum daily dose of ibuprofen?",
    answer="The maximum daily dose of ibuprofen for adults is 2400mg.",
    error_types=["numerical"],
    intensity=0.4,
)
print(f"Original: {result_medical['original_answer']}")
print(f"Hallucinated: {result_medical['hallucinated_answer']}")

# Generate temporal error
result_historical = generator.generate(
    context=[
        "Apollo 11 was the first crewed mission to land on the Moon, touching down on July 20, 1969."
    ],
    question="On what date did Apollo 11 land on the Moon?",
    answer="Apollo 11 landed on the Moon on July 20, 1969.",
    error_types=["temporal"],
    intensity=0.5,
)
print(f"Original: {result_historical['original_answer']}")
print(f"Hallucinated: {result_historical['hallucinated_answer']}")
```

**See the notebook for complete end‑to‑end examples**: [TinyLettuce notebook](https://github.com/KRLabsOrg/LettuceDetect/blob/main/demo/tinylettuce.ipynb)

---

## Motivation

RAG systems require hallucination detection, but current solutions have painful trade-offs between accuracy, cost, and speed.

**Current hallucination detection approaches:**

1. **Prompt-based detectors** - Use LLM APIs for zero/few-shot detection
   - Can be expensive for large-scale production deployments
   - Latency issues (2-10s per request) unsuitable for real-time use
   - Multiple API calls per detection increase costs

2. **Fine-tuned LLM detectors** - Large models (Llama-2-13B, Llama-3-8B) fine-tuned for detection
   - High accuracy but resource-intensive to train and deploy
   - Need GPU clusters, slow inference, high operational costs

3. **Encoder-based detectors** - BERT-style models for token classification
   - Fast and efficient but historically limited by short context (512 tokens)
   - Can't handle typical RAG contexts which often exceed this limit

**LettuceDetect's novel approach**: We solved the context problem by leveraging ModernBERT's 8K token capacity, achieving better accuracy than fine-tuned LLMs at a fraction of the computational cost. This shows that encoder-based detection could work at scale.

**Can we go even smaller and faster?**

**Enter TinyLettuce with Ettin encoders**: Ettin encoders released by LightOn (see the [HF collection](https://huggingface.co/collections/jhu-clsp/encoders-vs-decoders-the-ettin-suite-686303e16142257eed8e6aeb) and [paper](https://huggingface.co/papers/2507.11412)) are small, long‑context encoders with modern architectures. These lightweight transformers (17–68M parameters) support long contexts and are optimized for classification and retrieval, focusing on efficient representation learning for fast, accurate detection.

**The key insight**: With the right synthetic training data, a 17M parameter Ettin encoder can outperform 235B parameter LLMs at hallucination detection while running real-time on CPU. TinyLettuce makes it easy to use small models for hallucination detection by making it accessible, fast, and cost-effective for any deployment.

## Approach

**Specialized training data can matter more than parameter count**:

1. **Generate synthetic data** using LettuceDetect's HallucinationGenerator class - no manual annotation needed
2. **Train tiny Ettin encoders** (17M-68M parameters) on this specialized data  
3. **Deploy on CPU** for real-time inference with low latency and high throughput
4. **Scale effortlessly** - no GPU clusters or API limits (it's just a trained model)

## Synthetic Hallucination Data

You can use LettuceDetect's HallucinationGenerator class to generate training pairs automatically at scale.

### Production-Scale Generation

For large datasets, use our generation script:

```bash
# Generate 2,000 training samples (1,000 hallucinated + 1,000 non-hallucinated)
python scripts/generate_synthetic_data.py \
  --dataset rag-mini-bioasq \
  --num-samples 2000 \
  --model gpt-oss-120b \
  --output-format ragtruth \
  --output data/synthetic_2k.json
```

### Data Schema (RAGTruth format)

Minimal entry used for training:

```json
{
  "prompt": "...",
  "answer": "...",
  "labels": [{"start": 31, "end": 71, "label": "hallucinated"}],
  "split": "train",
  "task_type": "qa",
  "dataset": "synthetic",
  "language": "en"
}
```

## TinyLettuce Models (Ettin Encoders)

Built on the **Ettin encoder** (LightOn) — a lightweight, efficient transformer optimized for classification — these models achieve strong accuracy with low latency.

### Model Family

| Model | Parameters | Context Length | Key Advantage |
|-------|------------|----------------|---------------|
| **Ettin-17M** | 17 million | 8K tokens | Edge deployment |
| **Ettin-32M** | 32 million | 8K tokens | Very fast, good accuracy |
| **Ettin-68M** | 68 million | 8K tokens | Higher accuracy, still very fast |

Why Ettin encoders work well:
- 8K token context windows (longer than most inputs)
- Modern transformer design (RoPE, GLU activations)
- Optimized for token classification, not generation
- Efficient CPU inference without GPU overhead (smaller than ModernBERT models)

### Data & Training Setup (Published Models)

We show two training approaches for TinyLettuce models:

**1. General-Purpose Models (RAGTruth + Synthetic):**
- Base: Original RAGTruth dataset for broad hallucination detection capabilities
- Synthetic augmentation: 3,000 total samples (1,500 hallucinated + 1,500 non-hallucinated) from `enelpol/rag-mini-bioasq` generated using GPT-OSS-120b
- Training recipe: Ettin encoders fine-tuned on combined RAGTruth + synthetic data for robust performance across domains

**2. Domain-Specific Models (Synthetic-Only):**
- Pure synthetic data generation for targeted domain applications  
- Controllable error types and intensity for specific use cases
- Faster training and deployment for specialized scenarios
- Trained on 3,000 synthetic samples (1,500 hallucinated + 1,500 non-hallucinated)

### Training Hyperparameters (Released Models)

- Optimizer: AdamW; learning rate `1e-5`; weight decay `0.01`.
- Epochs: 5
- Batch size: 16; max sequence length: 4096 tokens.
- Tokenization: `AutoTokenizer`; label pad `-100`; `DataCollatorForTokenClassification`.

## Results

We trained several variants of Ettin encoders on synthetic data and tested them against larger scale LLM judges and fine-tuned encoders.

### Synthetic Data Evaluation (example-level)

Metrics are computed at example level (answer contains any hallucination vs none). Precision/recall/F1 reflect this binary decision; thresholds and post‑processing can affect absolute values.

**Test Set**: 600 synthetic examples (300 hallucinated + 300 non-hallucinated) generated with GPT-5-mini for fair evaluation.

*When trained and evaluated on domain-specific synthetic data, tiny models dominate (LettuceDetect-base shown without synthetic training):*

| Model | Parameters | Precision (%) | Recall (%) | F1 (%) | Hardware |
|-------|------------|---------------|------------|---------|----------|
| **TinyLettuce-17M** | **17M** | 84.56 | 98.21 | **90.87** | **CPU** |
| **TinyLettuce-32M** | **32M** | 80.36 | 99.10 | 88.76 | **CPU** |
| **TinyLettuce-68M** | **68M** | **89.54** | 95.96 | **92.64** | **CPU** |
| LettuceDetect-base (ModernBERT) | 150M | 79.06 | 98.21 | 87.60 | GPU |
| GPT-5-mini | ~200B | 71.95 | **100.00** | 83.69 | API/GPU |
| GPT-OSS-120B | 120B | 72.21 | 98.64 | 83.38 | GPU |
| Qwen3-235B | 235B | 66.74 | 99.32 | 79.84 | GPU |

### RAGTruth Benchmark Evaluation (example-level)

*Strong performance on standard benchmarks (Ettin models trained on RAGTruth + synthetic data):*

| Model | Parameters | F1 (%) |
|-------|------------|---------|
| **TinyLettuce-17M** | **17M** | 68.52 |
| **TinyLettuce-32M** | **32M** | 72.15 |
| **TinyLettuce-68M** | **68M** | **74.97** |
| LettuceDetect-base (ModernBERT) | 150M | 76.07 |
| LettuceDetect-large (ModernBERT) | 395M | **79.22** |
| Llama-2-13B (RAGTruth FT) | 13B | 78.70 |

TinyLettuce Ettin models demonstrate impressive performance given their compact size. These models are trained on both RAGTruth and synthetic data, achieving strong results across both evaluation sets. While ModernBERT models achieve slightly higher accuracy, TinyLettuce offers 6-23x parameter reduction with competitive results, making them ideal for resource-constrained deployments.

Baselines and judges: we compare against commonly used LLM judges (e.g., GPT‑5‑mini, GPT‑OSS‑120B, Qwen3‑235B) and fine‑tuned encoders/decoders reported in RAGTruth and follow‑up work (e.g., Llama‑2‑13B FT). Beyond benchmarks, deployment characteristics often determine real‑world value.

### Evaluation Method

- Span construction from tokens: threshold 0.5 on token hallucination prob; contiguous tokens merged into spans.
- Reported F1 is example‑level unless explicitly noted.
- Example command:

```bash
python scripts/evaluate.py \
  --model_path output/tinylettuce_68m \
  --data_path data/ragtruth/ragtruth_data.json \
  --evaluation_type example_level
```

## Real‑Time CPU Inference

TinyLettuce's biggest advantage isn't just accuracy — it's accessibility ⚡. These models run in real time on standard CPUs, making hallucination detection practical to deploy widely.

### End-to-End Workflow

```bash
# Step 1: Generate synthetic training data
python scripts/generate_synthetic_data.py \
  --dataset rag-mini-bioasq \
  --num-samples 50000 \
  --model gpt-oss-120b \
  --batch-size 50 \
  --output-format ragtruth \
  --output data/synthetic_large.json

# Step 2: Train TinyLettuce model
python scripts/train.py \
  --ragtruth-path data/train_combined_large.json \
  --model-name jhu-clsp/ettin-encoder-17m \
  --output-dir output/tinylettuce_17m \
  --batch-size 8 \
  --epochs 3

# Step 3: Deploy on CPU for real-time inference
python scripts/start_api.py prod --model output/tinylettuce_17m
```


---

## Bonus: Triplet‑Based RAGFactChecker

We have implemented a triplet-based hallucination detection model that you can use the same way as the standard lettucedetect models.

Generate triplets from any text:
```python
from lettucedetect.models.inference import HallucinationDetector
from lettucedetect.ragfactchecker import RAGFactChecker

detector = HallucinationDetector(
    method="rag_fact_checker",
)

rag = RAGFactChecker(model="gpt-5-mini")  # requires OPENAI_API_KEY
triplets = rag.generate_triplets("Paris is the capital of France.")
print(triplets)  # e.g., [["Paris", "is_capital_of", "France"]]
```

Compare triplets against each other:
```python
compare = rag.analyze_text_pair(
    "France is a country in Europe.", "France is a country in Asia."
)
print(compare)
#{
#    'answer_triplets': [['France', 'is', 'a country in Europe']],
#    'reference_triplets': [['France', 'is', 'a country in Asia']],
#    'comparison': {
#        'fact_check_results': {0: False},
#        'raw_output': FactCheckerOutput(fact_check_prediction_binary={0: False})
#    }
#}
```

Use it for detecting hallucinations in your data:
```python
# You can use it for detecting hallucinations in your data
result = detector.predict(
    context="The capital of France is Paris.",
    question="What is the capital of France?",
    answer="The capital of France is Berlin.",
    output_format="detailed",
)
print(result)
#{
#    'spans': [
#        {
#            'start': 0,
#            'end': 31,
#            'text': 'The capital of France is Berlin',
#            'confidence': 0.9,
#            'triplet': ['the capital of France', 'is', 'Berlin']
#        }
#    ],
#    'triplets': {
#        'answer': [['the capital of France', 'is', 'Berlin']],
#        'context': [['The capital of France', 'is', 'Paris']],
#        'hallucinated': [['the capital of France', 'is', 'Berlin']]
#    },
#    'fact_check_results': {0: False}
#}
```

This complements token/span detectors with interpretable, fact-level explanations.

---

## Limitations & Notes

- Results labeled “synthetic” reflect evaluation on generated data; real‑world performance depends on domain match. Consider adding a small, manually curated eval set.
- Baselines: we report GPT‑5‑mini and open‑source LLM baselines where available; prompt configuration impacts absolute scores.
- Metrics: synthetic and RAGTruth F1 are example-level unless otherwise noted; thresholds and post‑processing influence outcomes.

---

## Citation

If you find this work useful, please cite it as follows:

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

---

## References

[1] [RAGTruth: A Dataset for Hallucination Detection in Retrieval-Augmented Generation](https://aclanthology.org/2024.acl-long.585/)

[2] [LettuceDetect: A Hallucination Detection Framework for RAG Applications](https://arxiv.org/abs/2502.17125)

[3] [Ettin: Encoder Models by LightOn (paper)](https://huggingface.co/papers/2507.11412)

[4] [Ettin Encoder Models (HF models)](https://huggingface.co/jhu-clsp/ettin-encoder-68m)

[5] [RAGFactChecker](https://github.com/KRLabsOrg/RAGFactChecker)
