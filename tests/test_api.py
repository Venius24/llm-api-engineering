import httpx
import pytest
from fastapi.testclient import TestClient
from openai import APIConnectionError

from app import main
from app.schemas import RouteDecision


@pytest.fixture
def client():
    return TestClient(main.app)


def provider_down():
    return APIConnectionError(request=httpx.Request("POST", "https://api.deepseek.com/responses"))


def test_health(client):
    assert client.get("/health").json() == {"status": "ok"}


def test_chat_streams_generator_output(client, monkeypatch):
    seen = []

    def fake_generate(message):
        seen.append(message)
        yield "Произ"
        yield "водная"

    monkeypatch.setattr(main, "generate_answer", fake_generate)

    response = client.post("/chat", json={"message": "Что такое производная?"})

    assert response.status_code == 200
    assert response.headers["content-type"] == "text/plain; charset=utf-8"
    assert response.text == "Производная"
    assert seen == ["Что такое производная?"]


def test_chat_returns_503_when_provider_is_down(client, monkeypatch):
    def failing(message):
        raise provider_down()
        yield  # makes this a generator, like generate_answer

    monkeypatch.setattr(main, "generate_answer", failing)

    assert client.post("/chat", json={"message": "q"}).status_code == 503


@pytest.mark.parametrize("body", [{}, {"message": ""}, {"message": "x" * 4001}])
def test_chat_validates_message(client, body):
    assert client.post("/chat", json=body).status_code == 422


def test_route_returns_structured_decision(client, monkeypatch):
    decision = RouteDecision(use_rag=False, search_query="", reason="приветствие")
    monkeypatch.setattr(main, "route_question", lambda q: decision)

    assert client.post("/route", json={"message": "Привет"}).json() == decision.model_dump()


def test_route_maps_invalid_model_output_to_502(client, monkeypatch):
    def bad(q):
        return RouteDecision.model_validate_json("{}")

    monkeypatch.setattr(main, "route_question", bad)

    assert client.post("/route", json={"message": "q"}).status_code == 502


def test_route_returns_503_when_provider_is_down(client, monkeypatch):
    def down(q):
        raise provider_down()

    monkeypatch.setattr(main, "route_question", down)

    assert client.post("/route", json={"message": "q"}).status_code == 503
