import pytest
from fastapi.testclient import TestClient

from app import main


@pytest.fixture
def client():
    return TestClient(main.app)


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


def test_chat_requires_message(client):
    assert client.post("/chat", json={}).status_code == 422
