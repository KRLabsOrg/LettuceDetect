# Multilingual Support

LettuceDetect supports hallucination detection in multiple languages with one checkpoint per language.

## Supported Languages

| Language | Code | Model |
|----------|------|-------|
| English | en | [lettucedect-base-modernbert-en-v1](https://huggingface.co/KRLabsOrg/lettucedect-base-modernbert-en-v1) (ModernBERT) |
| German | de | [lettucedect-210m-eurobert-de-v1](https://huggingface.co/KRLabsOrg/lettucedect-210m-eurobert-de-v1) (EuroBERT) |
| French | fr | [lettucedect-210m-eurobert-fr-v1](https://huggingface.co/KRLabsOrg/lettucedect-210m-eurobert-fr-v1) (EuroBERT) |
| Spanish | es | [lettucedect-210m-eurobert-es-v1](https://huggingface.co/KRLabsOrg/lettucedect-210m-eurobert-es-v1) (EuroBERT) |
| Italian | it | [lettucedect-210m-eurobert-it-v1](https://huggingface.co/KRLabsOrg/lettucedect-210m-eurobert-it-v1) (EuroBERT) |
| Polish | pl | [lettucedect-210m-eurobert-pl-v1](https://huggingface.co/KRLabsOrg/lettucedect-210m-eurobert-pl-v1) (EuroBERT) |
| Chinese | cn | [lettucedect-210m-eurobert-cn-v1](https://huggingface.co/KRLabsOrg/lettucedect-210m-eurobert-cn-v1) (EuroBERT) |
| Hungarian | hu | [lettucedect-mmbert-base-hu-v1](https://huggingface.co/KRLabsOrg/lettucedect-mmbert-base-hu-v1) (mmBERT) |

The EuroBERT models also come in a larger, more accurate 610M variant (`lettucedect-610m-eurobert-<lang>-v1`), and Hungarian in a smaller one (`lettucedect-mmbert-small-hu-v1`). See the [multilingual collection on Hugging Face](https://huggingface.co/collections/KRLabsOrg/multilingual-hallucination-detection-682a2549c18ecd32689231ce).

## Usage

Pick the checkpoint for your language and pass the language code:

```python
from lettucedetect.models.inference import HallucinationDetector

detector = HallucinationDetector(
    method="transformer",
    model_path="KRLabsOrg/lettucedect-210m-eurobert-de-v1",
    lang="de",
    trust_remote_code=True,
)
```

> **Note:** the EuroBERT models load remote code that is not yet compatible with transformers 5.x; install `pip install "transformers>=4.48.3,<5"` when using them. ModernBERT and mmBERT models are unaffected.
