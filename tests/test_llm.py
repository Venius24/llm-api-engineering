import json
from types import SimpleNamespace

import httpx
import pytest
from pydantic import ValidationError

from app import config, llm
from app.schemas import RouteDecision
from tests.fakes import FakeResponses, function_call, stream, text_response, tool_call_response


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


def outputs(call_kwargs):
    return [json.loads(i["output"]) for i in call_kwargs["input"] if i.get("type") == "function_call_output"]


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
    assert first["model"] == config.LLM_MODEL
    assert first["tools"] == llm.TOOLS
    assert first["tool_choice"] == "auto"
    assert "stream" not in first


def test_tool_call_runs_search_and_streams_final_answer(fake_client, tool_calls):
    responses = fake_client(
        tool_call_response(function_call('{"query": "определение производной"}')),
        stream("Производная ", "— это предел."),
    )

    chunks = list(llm.generate_answer("Что такое производная?"))

    assert chunks == ["Производная ", "— это предел."]
    assert tool_calls == ["определение производной"]
    assert len(responses.calls) == 2


def test_second_call_replays_tool_call_and_result(fake_client, tool_calls):
    responses = fake_client(
        tool_call_response(function_call('{"query": "производная"}', call_id="call_42")),
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


def test_parallel_calls_in_one_turn_all_run(fake_client, tool_calls):
    responses = fake_client(
        tool_call_response(
            function_call('{"query": "производная"}', call_id="a"),
            function_call('{"query": "интеграл"}', call_id="b"),
        ),
        stream("ok"),
    )

    list(llm.generate_answer("q"))

    assert tool_calls == ["производная", "интеграл"]
    assert [i["call_id"] for i in responses.calls[1]["input"][1:]] == ["a", "a", "b", "b"]


def test_model_can_request_another_tool_round(fake_client, tool_calls):
    responses = fake_client(
        tool_call_response(function_call('{"query": "первый"}', call_id="a")),
        stream("Уточняю… ", calls=[function_call('{"query": "второй"}', call_id="b")]),
        stream("Готово."),
    )

    assert "".join(llm.generate_answer("q")) == "Уточняю… Готово."
    assert tool_calls == ["первый", "второй"]
    # The last request carries the whole tool history in order.
    assert [i["call_id"] for i in responses.calls[2]["input"][1:]] == ["a", "a", "b", "b"]


def test_tool_rounds_are_capped(fake_client, tool_calls, monkeypatch):
    monkeypatch.setattr(config, "MAX_TOOL_ROUNDS", 2)

    def again(call_id):
        return stream(calls=[function_call('{"query": "снова"}', call_id=call_id)])

    responses = fake_client(tool_call_response(function_call('{"query": "q"}')), again("b"), again("c"))

    chunks = list(llm.generate_answer("q"))

    assert chunks[-1] == llm.TOOL_LIMIT_MESSAGE
    assert len(responses.calls) == 3
    assert len(tool_calls) == 2


@pytest.mark.parametrize("arguments", ['{"q": "wrong key"}', "not json", '{"query": 5}'])
def test_invalid_arguments_go_back_to_the_model_as_tool_error(fake_client, tool_calls, arguments):
    responses = fake_client(tool_call_response(function_call(arguments)), stream("Не смог найти."))

    assert "".join(llm.generate_answer("q")) == "Не смог найти."
    assert tool_calls == []
    (result,) = outputs(responses.calls[1])
    assert result["error"] == "invalid arguments"
    assert result["details"]


def test_unknown_tool_is_reported_not_run(fake_client, tool_calls):
    responses = fake_client(tool_call_response(function_call("{}", name="delete_everything")), stream("ok"))

    list(llm.generate_answer("q"))

    assert tool_calls == []
    assert outputs(responses.calls[1]) == [{"error": "unknown tool: delete_everything"}]


def test_rag_failure_becomes_tool_error(fake_client, monkeypatch):
    def broken(query):
        raise httpx.ConnectError("refused")

    monkeypatch.setattr(llm, "search_study_materials", broken)
    responses = fake_client(tool_call_response(function_call('{"query": "q"}')), stream("База недоступна."))

    assert "".join(llm.generate_answer("q")) == "База недоступна."
    assert outputs(responses.calls[1]) == [{"error": "study materials search failed: ConnectError"}]


def test_only_text_deltas_are_streamed(fake_client, tool_calls):
    fake_client(tool_call_response(function_call('{"query": "q"}')), stream("a", "b"))

    assert "".join(llm.generate_answer("q")) == "ab"


def test_route_question_requests_json_schema_and_parses_it(fake_client):
    decision = {"use_rag": True, "search_query": "производная", "reason": "учебный вопрос"}
    responses = fake_client(text_response(json.dumps(decision, ensure_ascii=False)))

    assert llm.route_question("Что такое производная?") == RouteDecision(**decision)
    fmt = responses.calls[0]["text"]["format"]
    assert fmt["type"] == "json_schema"
    assert fmt["schema"] == RouteDecision.model_json_schema()


def test_route_question_rejects_malformed_output(fake_client):
    fake_client(text_response('{"use_rag": "maybe"}'))

    with pytest.raises(ValidationError):
        llm.route_question("q")
