import json
import os

from dotenv import load_dotenv
from openai import OpenAI

from app.schemas import SearchArgs
from app.tools import TOOLS, search_study_materials


load_dotenv()


client = OpenAI(
    api_key=os.environ["DEEPSEEK_API_KEY"],
    base_url="https://api.deepseek.com",
)


def generate_answer(question: str):

    # 1. Модель решает, нужен ли tool

    response = client.responses.create(
        model="deepseek-flash",
        instructions="""
Ты учебный ассистент.

Если вопрос относится к материалам учебной базы,
используй search_study_materials.

После получения результата tool
ответь пользователю на его основе.
""",
        input=question,
        tools=TOOLS,
        tool_choice="auto",
    )


    # 2. Ищем function call

    tool_call = None

    for item in response.output:
        if item.type == "function_call":
            tool_call = item
            break


    # 3. Tool не нужен

    if tool_call is None:
        yield response.output_text
        return


    # 4. Валидируем arguments

    args = SearchArgs.model_validate_json(
        tool_call.arguments
    )


    # 5. Выполняем tool

    tool_result = search_study_materials(
        args.query
    )


    # 6. Финальный streaming response

    stream = client.responses.create(
        model="deepseek-flash",
        instructions="""
Ответь пользователю на основе результата
search_study_materials.

Отвечай понятно и без лишней воды.
""",
        tools=TOOLS,
        input=[
            {
                "role": "user",
                "content": question,
            },
            {
                "type": "function_call",
                "call_id": tool_call.call_id,
                "name": tool_call.name,
                "arguments": tool_call.arguments,
            },
            {
                "type": "function_call_output",
                "call_id": tool_call.call_id,
                "output": json.dumps(
                    tool_result,
                    ensure_ascii=False,
                ),
            },
        ],
        stream=True,
    )


    for event in stream:
        if event.type == "response.output_text.delta":
            yield event.delta