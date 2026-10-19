import httpx
import pytest

from app import tools
from app.schemas import SearchArgs


def test_tool_definition_matches_function_and_schema():
    (tool,) = tools.TOOLS
    assert tool["type"] == "function"
    assert tool["name"] == tools.search_study_materials.__name__
    assert tool["parameters"] == SearchArgs.model_json_schema()
    assert tool["parameters"]["required"] == ["query"]


def test_search_posts_question_to_manual_rag(monkeypatch):
    seen = {}

    def fake_post(url, json, timeout):
        seen.update(url=url, json=json, timeout=timeout)
        return httpx.Response(200, json={"answer": "42"}, request=httpx.Request("POST", url))

    monkeypatch.setattr(tools.httpx, "post", fake_post)

    assert tools.search_study_materials("производная") == {"answer": "42"}
    assert seen["url"] == "http://127.0.0.1:8000/ask"
    assert seen["json"] == {"question": "производная"}
    assert seen["timeout"] == 60


def test_search_raises_on_rag_error(monkeypatch):
    def fake_post(url, json, timeout):
        return httpx.Response(500, request=httpx.Request("POST", url))

    monkeypatch.setattr(tools.httpx, "post", fake_post)

    with pytest.raises(httpx.HTTPStatusError):
        tools.search_study_materials("q")
