# Web API

LettuceDetect includes a FastAPI server for HTTP-based hallucination detection.

## Start the Server

```bash
# Development
python scripts/start_api.py dev

# Production
python scripts/start_api.py prod

# Custom model
python scripts/start_api.py dev --model KRLabsOrg/lettucedect-large-modernbert-en-v1
```

## Python Client

```python
from lettucedetect_api.client import LettuceClient

client = LettuceClient("http://127.0.0.1:8000")

contexts = ["The capital of France is Paris."]
question = "What is the capital of France?"
answer = "The capital of France is Berlin."

response = client.detect_spans(contexts, question, answer)
print(response)
```

## Installation

```bash
pip install lettucedetect[api]
```
