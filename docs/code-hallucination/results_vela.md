# Vela 2.0 0.3B through `VelaDetector`

[Vela 2.0](https://huggingface.co/collections/vllm-sr/vela-20) reports hallucination example-F1 on
the same 10,698-example test set as our v2 models
([EVALUATION.md](https://huggingface.co/vllm-sr/Vela-2.0-0.3B/blob/main/EVALUATION.md)).
This note reproduces the 0.3B number with LettuceDetect's own evaluator, so the
results can be compared directly with the
[mmBERT-base](model_card_v2_mmbert_base.md) and [Qwen-2b](model_card_v2_qwen.md) model cards.

## Setup

- **Model:** `vllm-sr/Vela-2.0-0.3B`, loaded with `VelaDetector.from_pretrained`(),
 which runs the inference code from the model repository as is.
- **Question:** Vela's documented hallucination span question (`VelaConfig()` defaults) at Vela's
  calibrated threshold.
- **Data:** the test splits of `KRLabsOrg/lettucedetect-code-hallucination` and
  `KRLabsOrg/lettucedetect-prose-hallucination`, 10,698 rows.
- **Input:** each row's `prompt` (user request plus context) goes in as `PromptInput`, which
  `VelaDetector` sends to Vela as the context, with the `answer` as the answer.
- **Metrics:** `scripts/span_eval_metrics.py`, the same char-overlap span-F1, example-F1 and IoU
  as the v2 model cards. Vela's headline number is **example-F1**.

```bash
pip install -e ".[vela]" datasets
python scripts/evaluate_span_model.py \
    --method vela \
    --model-path vllm-sr/Vela-2.0-0.3B \
    --revision <commit> \
    --dataset KRLabsOrg/lettucedetect-code-hallucination \
    --dataset KRLabsOrg/lettucedetect-prose-hallucination \
    --split test
```

## Results (test split, 10,698 samples)

| | example-F1 |
|---|--:|
| Vela 2.0 0.3B, reported by Vela | 0.848 |
| Vela 2.0 0.3B, this evaluation | 0.847 |

| dataset | n | span_F1 | span_P | span_R | ex_F1 | IoU |
|---|--:|--:|--:|--:|--:|--:|
| **ALL** | 10698 | **0.629** | 0.587 | 0.677 | 0.847 | 0.592 |
| acl | 440 | 0.467 | 0.550 | 0.405 | 0.744 | 0.458 |
| code-agent | 2015 | 0.473 | 0.495 | 0.452 | 0.750 | 0.463 |
| readme | 641 | 0.649 | 0.651 | 0.646 | 0.823 | 0.636 |
| tool-output | 617 | 0.513 | 0.594 | 0.452 | 0.744 | 0.540 |
| wikipedia | 1388 | 0.650 | 0.628 | 0.674 | 0.880 | 0.684 |
| psiloqa | 2897 | 0.725 | 0.621 | 0.869 | 0.959 | 0.627 |
| ragtruth | 2700 | 0.491 | 0.532 | 0.455 | 0.707 | 0.628 |

The target is the reported example-F1 within ±0.01; the result is 0.0006 below it. The RAGTruth
example-F1 (0.707) also matches Vela's reported RAGTruth F1 for the 0.3B (0.706).

## Notes

- The inference code comes from the model repository and is used directly. A new commit there can
  change both the code and the scores, so a rerun may not match these numbers exactly.
- Inputs over the model's 8,192-token budget have their context cut by Vela. `VelaDetector` reports
  those rows with the `input_truncated` warning.
  