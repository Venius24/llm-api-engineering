import json
import os

import httpx
from openai import OpenAI
from pydantic import ValidationError

from app import config
from app.schemas import RouteDecision, SearchArgs
from app.tools import TOOLS, search_study_materials


client = OpenAI(
    api_key=os.environ["DEEPSEEK_API_KEY"],
    base_url=config.LLM_BASE_URL,
    timeout=config.LLM_TIMEOUT,
    max_retries=config.LLM_MAX_RETRIES,
)


ASSISTANT_INSTRUCTIONS = """
Ты учебный ассистент.

Если вопрос относится к материалам учебной базы,
используй search_study_materials.

После получения результата tool
ответь пользователю на его основе.
"""

ANSWER_INSTRUCTIONS = """
Ответь пользователю на основе результата
search_study_materials.

Если tool вернул ошибку, скажи об этом и не выдумывай ответ.

Отвечай понятно и без лишней воды.
"""

ROUTER_INSTRUCTIONS = """
Ты маршрутизатор вопросов.

Определи, нужно ли искать ответ в нашей учебной базе знаний.

Если нужен поиск:
- use_rag = true
- search_query = хороший поисковый запрос

Если поиск не нужен:
- use_rag = false
- search_query = ""

Кратко объясни решение в reason.
"""

TOOL_LIMIT_MESSAGE = "\n\n[Остановлено: превышен лимит вызовов инструментов]"


def run_tool(call) -> dict:
    """Runs one function call. Problems become a tool error for the model, not an exception."""

    if call.name != "search_study_materials":
        return {"error": f"unknown tool: {call.name}"}

    try:
        args = SearchArgs.model_validate_json(call.arguments)
    except ValidationError as exc:
        return {
            "error": "invalid arguments",
            "details": exc.errors(include_url=False, include_context=False),
        }

    try:
        return search_study_materials(args.query)
    except httpx.HTTPError as exc:
        return {"error": f"study materials search failed: {type(exc).__name__}"}


def tool_round_items(call, result: dict) -> list:
    return [
        {
            "type": "function_call",
            "call_id": call.call_id,
            "name": call.name,
            "arguments": call.arguments,
        },
        {
            "type": "function_call_output",
            "call_id": call.call_id,
            "output": json.dumps(result, ensure_ascii=False, default=str),
        },
    ]


def function_calls(output) -> list:
    return [item for item in output if item.type == "function_call"]


def generate_answer(question: str):

    # 1. Модель решает, нужен ли tool

    response = client.responses.create(
        model=config.LLM_MODEL,
        instructions=ASSISTANT_INSTRUCTIONS,
        input=question,
        tools=TOOLS,
        tool_choice="auto",
    )

    calls = function_calls(response.output)

    # 2. Tool не нужен

    if not calls:
        yield response.output_text
        return

    # 3. Выполняем tools и стримим ответ; модель может запросить ещё раунд

    history = [{"role": "user", "content": question}]

    for _ in range(config.MAX_TOOL_ROUNDS):
        for call in calls:
            history += tool_round_items(call, run_tool(call))

        stream = client.responses.create(
            model=config.LLM_MODEL,
            instructions=ANSWER_INSTRUCTIONS,
            tools=TOOLS,
            input=history,
            stream=True,
        )

        calls = []

        for event in stream:
            if event.type == "response.output_text.delta":
                yield event.delta
            elif (
                event.type == "response.output_item.done"
                and event.item.type == "function_call"
            ):
                calls.append(event.item)

        if not calls:
            return

    yield TOOL_LIMIT_MESSAGE


def route_question(question: str) -> RouteDecision:
    """Structured output: the model must answer with a RouteDecision JSON object."""

    response = client.responses.create(
        model=config.LLM_MODEL,
        instructions=ROUTER_INSTRUCTIONS,
        input=question,
        text={
            "format": {
                "type": "json_schema",
                "name": "route_decision",
                "schema": RouteDecision.model_json_schema(),
            }
        },
    )

    return RouteDecision.model_validate_json(response.output_text)
