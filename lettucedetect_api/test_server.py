from fastapi.testclient import TestClient

from .server import app

TOKEN = "token"  # noqa: S105
HALLUCINATION_SCORE = "hallucination_score"
SPAN_START = "start"
SPAN_END = "end"
SPAN_TEXT = "text"


def test_valid_token_level_request() -> None:
    """Test valid request for token level detection."""
    request = {
        "contexts": ["France is a country in Europe. The capital of France is Paris."],
        "question": "What is the capital of France? What is the population of France?",
        "answer": "The capital of France is Paris. The population of France is 69 million.",
    }
    with TestClient(app) as client:
        response = client.post("/v1/lettucedetect/token", json=request)
    assert response.status_code == 200
    response_data = response.json()
    assert "predictions" in response_data
    assert len(response_data["predictions"]) >= 1
    for item in response_data["predictions"]:
        assert item.keys() == {TOKEN, HALLUCINATION_SCORE}
        assert isinstance(item[TOKEN], str)
        assert isinstance(item[HALLUCINATION_SCORE], float)


def test_valid_span_level_request() -> None:
    """Test valid request for span level detection."""
    request = {
        "contexts": ["France is a country in Europe. The capital of France is Paris."],
        "question": "What is the capital of France? What is the population of France?",
        "answer": "The capital of France is Paris. The population of France is 69 million.",
    }
    with TestClient(app) as client:
        response = client.post("/v1/lettucedetect/spans", json=request)
    assert response.status_code == 200
    response_data = response.json()
    assert "predictions" in response_data
    assert len(response_data["predictions"]) >= 1
    for item in response_data["predictions"]:
        assert item.keys() == {SPAN_START, SPAN_END, SPAN_TEXT, HALLUCINATION_SCORE}
        assert isinstance(item[SPAN_START], int)
        assert isinstance(item[SPAN_END], int)
        assert isinstance(item[SPAN_TEXT], str)
        assert isinstance(item[HALLUCINATION_SCORE], float)


def test_empty_request_token_level_request() -> None:
    """Test request with empty values for token level detection."""
    request = {
        "contexts": [],
        "question": "",
        "answer": "",
    }
    with TestClient(app) as client:
        response = client.post("/v1/lettucedetect/token", json=request)
    assert response.status_code == 200
    response_data = response.json()
    assert "predictions" in response_data
    assert len(response_data["predictions"]) == 0


def test_empty_request_span_level_request() -> None:
    """Test request with empty values for span level detection."""
    request = {
        "contexts": [],
        "question": "",
        "answer": "",
    }
    with TestClient(app) as client:
        response = client.post("/v1/lettucedetect/spans", json=request)
    assert response.status_code == 200
    response_data = response.json()
    assert "predictions" in response_data
    assert len(response_data["predictions"]) == 0


def test_invalid_request_token_level_request() -> None:
    """Test request with invalid type for token level detection."""
    request = {
        "contexts": ["France is a country in Europe. The capital of France is Paris."],
        "question": "What is the capital of France? What is the population of France?",
        "answer": None,
    }
    with TestClient(app) as client:
        response = client.post("/v1/lettucedetect/token", json=request)
    assert response.status_code == 422
    response_data = response.json()
    assert "detail" in response_data


def test_invalid_request_span_level_request() -> None:
    """Test request with invalid type for span level detection."""
    request = {
        "contexts": ["France is a country in Europe. The capital of France is Paris."],
        "question": "What is the capital of France? What is the population of France?",
        "answer": None,
    }
    with TestClient(app) as client:
        response = client.post("/v1/lettucedetect/spans", json=request)
    assert response.status_code == 422
    response_data = response.json()
    assert "detail" in response_data
