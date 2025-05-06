# Evaluation

## Use LLM baselines

```bash
python scripts/evaluate_llm.py --model "gpt-4o-mini" --data_path "data/translated/ragtruth-de-translated-300sample.json" --evaluation_type "example_level"
```

## Use HallucinationDetector

```bash
python scripts/evaluate.py --model_path "output/hallucination_detection_de_210m" --data_path "data/translated/ragtruth-de-translated-300sample.json" --evaluation_type "example_level"
```
