import json
from types import SimpleNamespace

import pytest
from pydantic import ValidationError

from app import llm
from tests.fakes import FakeResponses, stream, text_response, tool_call_response


@pytest.fixture
def fake_client(monkeypatch):
    def install(*results):
        responses = FakeResponses(*results)
        monkeypatch.setattr(llm, "client", SimpleNamespace(responses=responses))
        return responses

    return install


@pytest.fixture
def tool_calls(monkeypatch):
    seen = []

    def fake_search(query):
        seen.append(query)
        return {"answer": "Производная — предел отношения приращений.", "sources": [{"chunk_id": 7}]}

    monkeypatch.setattr(llm, "search_study_materials", fake_search)
    return seen


def test_answers_directly_without_tool(fake_client, tool_calls):
    responses = fake_client(text_response("Привет!"))

    assert list(llm.generate_answer("Привет")) == ["Привет!"]
    assert tool_calls == []
    assert len(responses.calls) == 1


def test_first_call_offers_the_tool(fake_client, tool_calls):
    responses = fake_client(text_response("ok"))

    list(llm.generate_answer("Что такое производная?"))

    first = responses.calls[0]
    assert first["input"] == "Что такое производная?"
    assert first["tools"] == llm.TOOLS
    assert first["tool_choice"] == "auto"
    assert "stream" not in first


def test_tool_call_runs_search_and_streams_final_answer(fake_client, tool_calls):
    responses = fake_client(
        tool_call_response('{"query": "определение производной"}'),
        stream("Производная ", "— это предел."),
    )

    chunks = list(llm.generate_answer("Что такое производная?"))

    assert chunks == ["Производная ", "— это предел."]
    assert tool_calls == ["определение производной"]
    assert len(responses.calls) == 2


def test_second_call_replays_tool_call_and_result(fake_client, tool_calls):
    responses = fake_client(
        tool_call_response('{"query": "производная"}', call_id="call_42"),
        stream("ok"),
    )

    list(llm.generate_answer("Что такое производная?"))

    second = responses.calls[1]
    assert second["stream"] is True
    user, call, output = second["input"]
    assert user == {"role": "user", "content": "Что такое производная?"}
    assert call == {
        "type": "function_call",
        "call_id": "call_42",
        "name": "search_study_materials",
        "arguments": '{"query": "производная"}',
    }
    assert output["type"] == "function_call_output"
    assert output["call_id"] == "call_42"
    # Non-ASCII is kept readable for the model rather than \u-escaped.
    assert "Производная" in output["output"]
    assert json.loads(output["output"])["sources"] == [{"chunk_id": 7}]


def test_only_text_deltas_are_streamed(fake_client, tool_calls):
    fake_client(tool_call_response('{"query": "q"}'), stream("a", "b"))

    assert "".join(llm.generate_answer("q")) == "ab"


@pytest.mark.parametrize("arguments", ['{"q": "wrong key"}', "not json", '{"query": 5}'])
def test_invalid_tool_arguments_are_rejected_before_the_tool_runs(fake_client, tool_calls, arguments):
    fake_client(tool_call_response(arguments))

    with pytest.raises(ValidationError):
        list(llm.generate_answer("q"))
    assert tool_calls == []
