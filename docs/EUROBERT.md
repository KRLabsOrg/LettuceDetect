# 🥬 LettuceDetect Goes Multilingual: Fine-tuning EuroBERT on Synthetic RAGTruth Translations

<p align="center">
  <img src="https://github.com/KRLabsOrg/LettuceDetect/blob/feature/cn_llm_eval/assets/lettuce_detective_multi.png?raw=true" alt="LettuceDetect Multilingual Task Force" width="520"/>
  <br>
  <em>Expanding hallucination detection across languages for robust RAG pipelines.</em>
</p>

---

## 🏷️ TL;DR

- We present the first *multilingual hallucination detection* encoder-based models for Retrieval-Augmented Generation (RAG).
- We translated the [RAGTruth dataset](https://arxiv.org/abs/2401.00396) (with hallucination tags preserved) into German, French, Italian, Spanish, Polish, and Chinese to create multilingual training data.
- We fine-tuned the highly efficient and long-context [**EuroBERT**](https://huggingface.co/blog/EuroBERT/release) encoder for token-level hallucination detection in all these languages.
- Our experiments show that **EuroBERT** significantly outperforms prompt-based LLM judges (e.g., GPT-4.1-mini), showing the effectiveness of encoder-based detection.
- We release all the translated dataset, the fine-tuned models, and the scrips to translate and fine-tune the models under MIT license for the community to use.

---

## Quick Links

- **GitHub**: [github.com/KRLabsOrg/LettuceDetect](https://github.com/KRLabsOrg/LettuceDetect)  
- **PyPI**: [pypi.org/project/lettucedetect](https://pypi.org/project/lettucedetect/)  
- **arXiv Paper**: [2502.17125](https://arxiv.org/abs/2502.17125)
- **Hugging Face Models**:  
  - TODO
- **Streamlit Demo**: TODO


## Background

**LettuceDetect** ([blog](https://huggingface.co/blog/adaamko/lettucedetect)) is a lightweight, open-source hallucination detector for RAG pipelines, originally leveraging ModernBERT for efficient token-level detection. It was trained on [RAGTruth](https://aclanthology.org/2024.acl-long.585/), a manually annotated, span-level English dataset for hallucination detection. LettuceDetect showed that **encoder-based models can outperform even large LLM judges,** all while running much faster and cheaper.

However, many real-world RAG applications are *multilingual* and current state-of-the-art still struggle to detect hallucinations in multilingual settings because of the lack of multilingual hallucination detection models and datasets.


## The Approach

To fill this gap, we created multilingual versions of the RAGTruth dataset and fine-tuned the highly efficient and long-context [**EuroBERT**](https://huggingface.co/blog/EuroBERT/release) encoder for token-level hallucination detection in all these languages. For translation, we used the [google/gemma-3-27b-it](https://huggingface.co/google/gemma-3-27b-it) model and [vllm](https://github.com/vllm-project/vllm) for running the translations in our private GPU. For the translations we've used a single A100 GPU, with vllm we were able to translate ~30 examples in parallel, and a single lang-to-lang translation took ~12 hours. We provide the translation script under MIT license so the community can experiment with other models and languages.

Our pipeline works as follows:

1. **Annotation Tagging**: In the English RAGTruth data, hallucinated answer spans are tagged using `<hal>` XML tags.  
   Example:
   ```
   <answer>
   The French Revolution started in <hal>1788</hal>.
   </answer>
   ```

2. **LLM-based Translation**: We use the [google/gemma-3-27b-it](https://huggingface.co/google/gemma-3-27b-it) model to translate context, question, and answer *while preserving all `<hal>` tags*. For easier translation, we merge overlapping `<hal>` tags into a single tag. The prompts are released in our repository. After translation, we extract the translated context, question, and answer and the annotated tags and save them the same format as the original RAGTruth data.

3. **Extraction & Validation**: We extract the translated context, question, and answer and the annotated tags and save them the same format as the original RAGTruth data. We also provide a script to validate the translation quality by comparing the original and translated data.

4. **Fine-tuning**: We fine-tune the [**EuroBERT**](https://huggingface.co/blog/EuroBERT/release) encoder for token-level hallucination detection in all these languages. We use the [**EuroBERT**](https://huggingface.co/blog/EuroBERT/release) encoder for token-level hallucination detection in all these languages. 


**Supported Languages**:

We support the following languages: Chinese, French, German, Italian, Spanish, and Polish.

### Examples

To show an example of the translation, here's an example from the original RAGTruth data:


**English**
```xml
The first quartile (Q1) splits the lowest 25% of the data, while the second quartile (Q2) splits the data into two equal halves, with the median being the middle value of the lower half. Finally, the third quartile (Q3) splits the <hal>highest 75%</hal> of the data.
```

- *The phrase “highest 75%” is hallucinated, as the reference correctly states “lowest 75% (or highest 25%)”.*

---

**German**

```xml
Das erste Quartil (Q1) teilt die unteren 25% der Daten, während das zweite Quartil (Q2) die Daten in zwei gleiche Hälften teilt, wobei der Median den Mittelpunkt der unteren Hälfte bildet. Schließlich teilt das dritte Quartil (Q3) die <hal>höchsten 75%</hal> der Daten.
```

Here, the phrase "höchsten 75%" is hallucinated, as the reference correctly states "unteren 75% (oder höchsten 25%)".


# Get Going

Install the package:

```bash
pip install lettucedetect
```

### Transformer-based Hallucination Detection (German)

```python
from lettucedetect.models.inference import HallucinationDetector

detector = HallucinationDetector(
    method="transformer",
    model_path="KRLabsOrg/lettucedect-base-modernbert-de-v1",
    lang="de"
)

contexts = [
    "Frankreich ist ein Land in Europa. Die Hauptstadt von Frankreich ist Paris. Die Bevölkerung Frankreichs beträgt 67 Millionen."
]
question = "Was ist die Hauptstadt von Frankreich? Wie groß ist die Bevölkerung Frankreichs?"
answer = "Die Hauptstadt von Frankreich ist Paris. Die Bevölkerung Frankreichs beträgt 69 Millionen."

predictions = detector.predict(context=contexts, question=question, answer=answer, output_format="spans")
print("Vorhersagen:", predictions)
```

### LLM-based Hallucination Detection (German)

```python
from lettucedetect.models.inference import HallucinationDetector

detector = HallucinationDetector(method="llm", lang="de")

contexts = [
    "Frankreich ist ein Land in Europa. Die Hauptstadt von Frankreich ist Paris. Die Bevölkerung Frankreichs beträgt 67 Millionen."
]
question = "Was ist die Hauptstadt von Frankreich? Wie hoch ist die Bevölkerung Frankreichs?"
answer = "Die Hauptstadt von Frankreich ist Paris. Die Bevölkerung Frankreichs beträgt 82222 Millionen."

predictions = detector.predict(context=contexts, question=question, answer=answer, output_format="spans")
print("Vorhersagen:", predictions)
```


## Model

We've made use of the [**EuroBERT**](https://huggingface.co/blog/EuroBERT/release) model recently released and marked a big milestone in modern BERT architecture that supports long context and multilingual support.
Trained on a massive 5 trillion-token corpus spanning 15 languages, EuroBERT natively supports long-context processing, handling input sequences up to 8,192 tokens. The model architecture incorporates modern transformer innovations—including grouped query attention, rotary positional embeddings, and advanced normalization techniques—to achieve both high computational efficiency and strong generalization. EuroBERT is released in several parameter sizes (210M, 610M, and 2.1B). Across a wide spectrum of tasks, EuroBERT consistently demonstrates performance competitive with or surpassing prior open models

For the multilingual hallucination detection, we've trained the 210M and 610M models on all of the translated languages (Chinese, French, German, Italian, Spanish, and Polish). We've used the [**EuroBERT**](https://huggingface.co/blog/EuroBERT/release) encoder for token-level hallucination detection in all these languages. 

## Training

Our multilingual EuroBERT-based hallucination detection models closely follow the original LettuceDetect training approach. The input sequence is constructed by concatenating Context, Question, and Answer segments separated by special tokens ([CLS] for context and [SEP] for question/answer boundaries), capped at 4,096 tokens for computational feasibility. Tokenization is performed using Hugging Face's AutoTokenizer, which automatically inserts appropriate segment markers ([CLS], [SEP]) and handles subword segmentation.

During labeling, context and question tokens are masked by assigning a label of -100 (to exclude them from loss computation), while answer tokens receive binary labels—0 indicating supported tokens, and 1 for hallucinated tokens. We utilize the EuroBERT encoder within Hugging Face’s AutoModelForTokenClassification framework, augmented only with a linear classification head; no further pretraining steps (such as NLI) are included.

Training employs the AdamW optimizer (learning rate = 1 × 10⁻⁵, weight decay = 0.01), over six epochs with a batch size of 8. Dynamic padding via DataCollatorForTokenClassification efficiently manages variable sequence lengths. All experiments run on a single NVIDIA A100 GPU (40 GB) per language. The best-performing model checkpoint is selected based on token-level F1 scores on a validation set and saved in the safetensors format.

During inference, the model outputs hallucination probabilities at the token level. Tokens with probabilities above 0.5 are merged into contiguous spans, providing precise, span-level hallucination predictions. This consistent training and inference methodology ensures efficient, accurate detection across multiple languages.

## Results

We've evaluated the performance of the fine-tuned models on the translated RAGTruth dataset. We've used the [**EuroBERT**](https://huggingface.co/blog/EuroBERT/release) encoder for token-level hallucination detection in all these languages. 

To assess the effectiveness of our multilingual EuroBERT models, we compared them to a prompt-based baseline using GPT-4.1-mini, which we implemented ourselves. This baseline employs a few-shot prompting approach to request span-level hallucination identification directly from the LLM. The prompt-based method and evaluation script are also publicly released for reproducibility.

### Synthetic Multilingual Results

| Language | Model           | Precision (%) | Recall (%) | F1 (%) | GPT-4.1-mini Precision (%) | GPT-4.1-mini Recall (%) | GPT-4.1-mini F1 (%) | Δ F1 (%) |
|----------|-----------------|---------------|------------|--------|----------------------------|-------------------------|---------------------|----------|
| Chinese | EuroBERT-210M   | 75.46         | 73.38      | 74.41  | 43.97                      | 95.55                   | 60.23               | +14.18   |
| Chinese | EuroBERT-610M   | 78.90         | 75.72      | **77.27**  | 43.97                      | 95.55                   | 60.23               | +17.04   |
| French  | EuroBERT-210M   | 58.86         | 74.34      | 65.70  | 46.45                      | 94.91                   | 62.37               | +3.33    |
| French  | EuroBERT-610M   | 67.08         | 80.38      | **73.13**  | 46.45                      | 94.91                   | 62.37               | +10.76   |
| German  | EuroBERT-210M   | 66.70         | 66.70      | 66.70  | 44.82                      | 95.02                   | 60.91               | +5.79    |
| German  | EuroBERT-610M   | 77.04         | 72.96      | **74.95**  | 44.82                      | 95.02                   | 60.91               | +14.04   |
| Italian | EuroBERT-210M   | 60.57         | 72.32      | 65.93  | 44.87                      | 95.55                   | 61.06               | +4.87    |
| Italian | EuroBERT-610M   | 76.67         | 72.85      | **74.71**  | 44.87                      | 95.55                   | 61.06               | +13.65   |
| Spanish | EuroBERT-210M   | 69.48         | 73.38      | 71.38  | 46.56                      | 94.59                   | 62.40               | +8.98    |
| Spanish | EuroBERT-610M   | 76.32         | 70.41      | **73.25**  | 46.56                      | 94.59                   | 62.40               | +10.85   |
| Polish  | EuroBERT-210M   | 63.62         | 69.57      | 66.46  | 42.92                      | 95.76                   | 59.27               | +7.19    |
| Polish  | EuroBERT-610M   | 77.16         | 69.36      | **73.05**  | 42.92                      | 95.76                   | 59.27               | +13.78   |

We can see accross all the languages, the EuroBERT-610M model performs the best, outperforming the 210M variant (while also being much bigger in terms of size).




### Manual Validation (German)

For manual validation, we selected and carefully reviewed 100 examples covering all task types from RAGTruth (QA, summarization, data-to-text), correcting annotation errors as necessary. Performance consistency on this manual dataset strongly suggests that our synthetic translation approach produces high-quality annotations.

| Model            | Precision (%) | Recall (%) | F1 (%) |
|------------------|---------------|------------|--------|
| EuroBERT-210M | 68.32         | 68.32      | 68.32  |
| EuroBERT-610M | 74.47         | 69.31      | **71.79**  |
| GPT-4.1-mini   | 44.50         | 92.08      | 60.00  |

These results highlight that EuroBERT-based multilingual detectors significantly outperform GPT-4.1-mini's prompt-based approach across all evaluated languages.


## Key Takeaways

- **Annotation projection provides strong alternative when sufficient data is not available:** Preserving HAL tags through translation enables rapid, parallel data creation for any language.
- **EuroBERT is efficient and robust:** Long-context support, generalization, and optimized attention enable fast, high-accuracy hallucination detection in RAG.
- **Reproducible and open:** Scripts, models, and methodology are open for the research community.


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