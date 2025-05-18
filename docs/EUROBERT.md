# 🥬 LettuceDetect Goes Multilingual: Fine-tuning EuroBERT on Synthetic RAGTruth Translations

<p align="center">
  <img src="https://github.com/KRLabsOrg/LettuceDetect/blob/feature/cn_llm_eval/assets/lettuce_detective_multi.png?raw=true" alt="LettuceDetect Multilingual Task Force" width="520"/>
  <br>
  <em>Expanding hallucination detection across languages for RAG pipelines.</em>
</p>

---

## 🏷️ TL;DR

- We present the first multilingual hallucination detection framework for Retrieval-Augmented Generation (RAG).
- We translated the [RAGTruth dataset](https://arxiv.org/abs/2401.00396) into German, French, Italian, Spanish, Polish, and Chinese while preserving hallucination annotations.
- We fine-tuned [**EuroBERT**](https://huggingface.co/blog/EuroBERT/release) for token-level hallucination detection across all these languages.
- Our experiments show that **EuroBERT** significantly outperforms prompt-based LLM judges like GPT-4.1-mini by up to **17 F1 points**.
- All translated datasets, fine-tuned models, and translation scripts are available under MIT license.

---

## Quick Links

- **GitHub**: [github.com/KRLabsOrg/LettuceDetect](https://github.com/KRLabsOrg/LettuceDetect)  
- **PyPI**: [pypi.org/project/lettucedetect](https://pypi.org/project/lettucedetect/)  
- **arXiv Paper**: [2502.17125](https://arxiv.org/abs/2502.17125)
- **Hugging Face Models**:  
  - [Our HF collection](https://huggingface.co/collections/KRLabsOrg/multilingual-hallucination-detection-682a2549c18ecd32689231ce)
- **Demo**: [Hugging Face Space](https://huggingface.co/spaces/KRLabsOrg/LettuceDetect-Multilingual)


## Get Started

Install the package:

```bash
pip install lettucedetect
```

### Transformer-based Hallucination Detection (German)

```python
from lettucedetect.models.inference import HallucinationDetector

detector = HallucinationDetector(
    method="transformer",
    model_path="KRLabsOrg/lettucedect-610m-eurobert-de-v1",
    lang="de",
    trust_remote_code=True
)

contexts = [
    "Frankreich ist ein Land in Europa. Die Hauptstadt von Frankreich ist Paris. Die Bevölkerung Frankreichs beträgt 67 Millionen."
]
question = "Was ist die Hauptstadt von Frankreich? Wie groß ist die Bevölkerung Frankreichs?"
answer = "Die Hauptstadt von Frankreich ist Paris. Die Bevölkerung Frankreichs beträgt 69 Millionen."

predictions = detector.predict(context=contexts, question=question, answer=answer, output_format="spans")
print("Predictions:", predictions)
```

### LLM-based Hallucination Detection (Our Baseline)

```python
from lettucedetect.models.inference import HallucinationDetector

detector = HallucinationDetector(method="llm", lang="de")

contexts = [
    "Frankreich ist ein Land in Europa. Die Hauptstadt von Frankreich ist Paris. Die Bevölkerung Frankreichs beträgt 67 Millionen."
]
question = "Was ist die Hauptstadt von Frankreich? Wie hoch ist die Bevölkerung Frankreichs?"
answer = "Die Hauptstadt von Frankreich ist Paris. Die Bevölkerung Frankreichs beträgt 82222 Millionen."

predictions = detector.predict(context=contexts, question=question, answer=answer, output_format="spans")
print("Predictions:", predictions)
```


## Background

**LettuceDetect** ([blog](https://huggingface.co/blog/adaamko/lettucedetect)) is a lightweight, open-source hallucination detector for RAG pipelines that leverages ModernBERT for efficient token-level detection. It was originally trained on [RAGTruth](https://aclanthology.org/2024.acl-long.585/), a manually annotated English dataset for hallucination detection. The initial research demonstrated that encoder-based models can outperform large LLM judges while being significantly faster and more cost-effective.

Despite these advances, many real-world RAG applications are multilingual, and detecting hallucinations across languages remains challenging due to the lack of specialized models and datasets for non-English content.


## Our Approach

To address this gap, we created multilingual versions of the RAGTruth dataset and fine-tuned EuroBERT for hallucination detection across languages. For translation, we used [google/gemma-3-27b-it](https://huggingface.co/google/gemma-3-27b-it) with [vllm](https://github.com/vllm-project/vllm) running on a single A100 GPU. This setup enabled parallel translation of approximately 30 examples at a time, with each language translation pass taking about 12 hours.

Our pipeline works as follows:

1. **Annotation Tagging**: In the English RAGTruth data, hallucinated answer spans are tagged using `<hal>` XML tags.  
   Example:
   ```
   <answer>
   The French Revolution started in <hal>1788</hal>.
   </answer>
   ```

2. **LLM-based Translation**: We translate context, question, and answer while preserving all `<hal>` tags. For easier translation, we merge overlapping tags.

3. **Extraction & Validation**: We extract the translated content and annotations, maintaining the same format as the original RAGTruth data.

4. **Fine-tuning**: We train EuroBERT models for token classification to identify hallucinated content in each language.


### Supported Languages

Our models support hallucination detection in Chinese, French, German, Italian, Spanish, and Polish.

### Translation Example

To illustrate our approach, here's an example from the original RAGTruth data and its German translation:

**English**
```xml
The first quartile (Q1) splits the lowest 25% of the data, while the second quartile (Q2) splits the data into two equal halves, with the median being the middle value of the lower half. Finally, the third quartile (Q3) splits the <hal>highest 75%</hal> of the data.
```

- *The phrase "highest 75%" is hallucinated, as the reference correctly states "lowest 75% (or highest 25%)".*

**German**

```xml
Das erste Quartil (Q1) teilt die unteren 25% der Daten, während das zweite Quartil (Q2) die Daten in zwei gleiche Hälften teilt, wobei der Median den Mittelpunkt der unteren Hälfte bildet. Schließlich teilt das dritte Quartil (Q3) die <hal>höchsten 75%</hal> der Daten.
```

Here, the phrase "höchsten 75%" is hallucinated, as the reference correctly states "unteren 75% (oder höchsten 25%)".


## Model Architecture

We leverage [**EuroBERT**](https://huggingface.co/blog/EuroBERT/release), a recently released transformer model that represents a significant advancement in encoder architectures with its long-context capabilities and multilingual support.

Trained on a massive 5 trillion-token corpus spanning 15 languages, EuroBERT processes sequences up to 8,192 tokens. The architecture incorporates modern innovations including grouped query attention, rotary positional embeddings, and advanced normalization techniques. These features enable both high computational efficiency and strong generalization abilities.

For multilingual hallucination detection, we trained both the 210M and 610M parameter variants across all supported languages.

## Training Process

Our EuroBERT-based hallucination detection models follow the original LettuceDetect training methodology:

**Input Processing**
- Concatenate Context, Question, and Answer with special tokens
- Cap sequences at 4,096 tokens for computational efficiency
- Use AutoTokenizer for appropriate tokenization and segment markers

**Label Assignment**
- Mask context and question tokens (label = -100)
- Assign binary labels to answer tokens: 0 (supported) or 1 (hallucinated)

**Model Configuration**
- Use EuroBERT within AutoModelForTokenClassification framework
- Add only a linear classification head without additional pretraining

**Training Details**
- AdamW optimizer (learning rate = 1 × 10⁻⁵, weight decay = 0.01)
- Six epochs with batch size of 8
- Dynamic padding via DataCollatorForTokenClassification
- Single NVIDIA A100 GPU (80GB) per language

During inference, tokens with hallucination probabilities above 0.5 are merged into contiguous spans, providing precise identification of problematic content.

## Results

We evaluated our models on the translated RAGTruth dataset and compared them to a prompt-based baseline using GPT-4.1-mini. This baseline was implemented using few-shot prompting to identify hallucinated spans directly.

### Synthetic Multilingual Results

| Language | Model           | Precision (%) | Recall (%) | F1 (%) | GPT-4.1-mini Precision (%) | GPT-4.1-mini Recall (%) | GPT-4.1-mini F1 (%) | Δ F1 (%) |
|----------|-----------------|---------------|------------|--------|----------------------------|-------------------------|---------------------|----------|
| Chinese | EuroBERT-210M   | 75.46         | 73.38      | 74.41  | 43.97                      | 95.55                   | 60.23               | +14.18   |
| Chinese | EuroBERT-610M   | **78.90**     | **75.72**  | **77.27**  | 43.97                      | 95.55                   | 60.23               | **+17.04**   |
| French  | EuroBERT-210M   | 58.86         | 74.34      | 65.70  | 46.45                      | 94.91                   | 62.37               | +3.33    |
| French  | EuroBERT-610M   | **67.08**     | **80.38**  | **73.13**  | 46.45                      | 94.91                   | 62.37               | **+10.76**   |
| German  | EuroBERT-210M   | 66.70         | 66.70      | 66.70  | 44.82                      | 95.02                   | 60.91               | +5.79    |
| German  | EuroBERT-610M   | **77.04**     | **72.96**  | **74.95**  | 44.82                      | 95.02                   | 60.91               | **+14.04**   |
| Italian | EuroBERT-210M   | 60.57         | 72.32      | 65.93  | 44.87                      | 95.55                   | 61.06               | +4.87    |
| Italian | EuroBERT-610M   | **76.67**     | **72.85**  | **74.71**  | 44.87                      | 95.55                   | 61.06               | **+13.65**   |
| Spanish | EuroBERT-210M   | 69.48         | 73.38      | 71.38  | 46.56                      | 94.59                   | 62.40               | +8.98    |
| Spanish | EuroBERT-610M   | **76.32**     | 70.41      | **73.25**  | 46.56                      | 94.59                   | 62.40               | **+10.85**   |
| Polish  | EuroBERT-210M   | 63.62         | 69.57      | 66.46  | 42.92                      | 95.76                   | 59.27               | +7.19    |
| Polish  | EuroBERT-610M   | **77.16**     | 69.36      | **73.05**  | 42.92                      | 95.76                   | 59.27               | **+13.78**   |

Across all languages, the EuroBERT-610M model consistently outperforms both the 210M variant and the GPT-4.1-mini baseline.

### Manual Validation (German)

For a more rigorous evaluation, we manually reviewed 300 examples covering all task types from RAGTruth (QA, summarization, data-to-text). After correcting any annotation errors, we found that performance remained strong, validating our translation approach:

| Model            | Precision (%) | Recall (%) | F1 (%) |
|------------------|---------------|------------|--------|
| EuroBERT-210M    | 68.32         | 68.32      | 68.32  |
| EuroBERT-610M    | **74.47**     | 69.31      | **71.79** |
| GPT-4.1-mini     | 44.50         | **92.08**  | 60.00  |

An interesting pattern: GPT-4.1-mini shows high recall but poor precision - it identifies most hallucinations but produces many false positives, making it less reliable in production settings.

## Trade-offs: Model Size vs Performance

When choosing between model variants, consider these trade-offs:

- **EuroBERT-210M** – Approximately 3× faster inference, smaller memory footprint, 5-10% lower F1 scores
- **EuroBERT-610M** – Highest detection accuracy across all languages, requires more compute resources

## Key Takeaways

- **Translating annotation can be effective**: Preserving hallucination tags through translation enables rapid creation of multilingual detection datasets when sufficient data is not available.
- **EuroBERT is a good choice for multilingual hallucination detection**: Its long-context capabilities and efficient attention mechanisms make it ideal for RAG verification.
- **Open framework for multilingual RAG**: All components are available under MIT license: translation, training, and inference.


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

## References

[1] [Niu et al., 2024, RAGTruth: A Dataset for Hallucination Detection in Retrieval-Augmented Generation](https://aclanthology.org/2024.acl-long.585/)

[2] [Luna: A Simple and Effective Encoder-Based Model for Hallucination Detection in Retrieval-Augmented Generation](https://aclanthology.org/2025.coling-industry.34/)

[3] [ModernBERT: A Modern BERT Model for Long-Context Processing](https://huggingface.co/blog/modernbert)

[4] [Gemma 3](https://blog.google/technology/developers/gemma-3/)

[5] [EuroBERT](https://huggingface.co/blog/EuroBERT/release)