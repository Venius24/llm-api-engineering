import json
import os

import httpx
from dotenv import load_dotenv
from openai import OpenAI
from pydantic import BaseModel


load_dotenv()


client = OpenAI(
    api_key=os.environ["DEEPSEEK_API_KEY"],
    base_url="https://api.deepseek.com",
)


# -------------------------
# 1. Аргументы нашего tool
# -------------------------

class SearchArgs(BaseModel):
    query: str


# -------------------------
# 2. Реальная Python-функция
# -------------------------

def search_study_materials(query: str) -> dict:
    response = httpx.post(
        "http://127.0.0.1:8000/ask",
        json={"question": query},
        timeout=60,
    )

    response.raise_for_status()

    return response.json()


# -------------------------
# 3. Описание tool для LLM
# -------------------------

tools = [
    {
        "type": "function",
        "name": "search_study_materials",
        "description": (
            "Search the local study knowledge base. "
            "Use this tool for questions about mathematics "
            "or information that should be answered from study materials."
        ),
        "parameters": SearchArgs.model_json_schema(),
    }
]


question = "Что такое производная?"


# -------------------------
# 4. LLM решает:
#    отвечать самому
#    или вызвать tool
# -------------------------

response = client.responses.create(
    model="deepseek-flash",
    instructions="""
Ты учебный ассистент.

Если вопрос относится к учебным материалам,
используй search_study_materials.

После получения результата tool
ответь пользователю на основе найденной информации.
""",
    input=question,
    tools=tools,
    tool_choice="auto",
)


# -------------------------
# 5. Ищем tool call
# -------------------------

tool_call = None

for item in response.output:
    if item.type == "function_call":
        tool_call = item
        break


# -------------------------
# 6. Tool не понадобился
# -------------------------

if tool_call is None:
    print("MODEL ANSWER:")
    print(response.output_text)

    raise SystemExit


print("TOOL CALL:")
print(tool_call.name)
print(tool_call.arguments)


# -------------------------
# 7. Валидируем arguments
# -------------------------

args = SearchArgs.model_validate_json(
    tool_call.arguments
)

print()
print("VALIDATED:")
print(args)


# -------------------------
# 8. Реально вызываем manual-rag
# -------------------------

tool_result = search_study_materials(
    args.query
)

print()
print("RAG ANSWER:")
print(tool_result["answer"])


# -------------------------
# 9. Возвращаем результат LLM
# -------------------------

final_response = client.responses.create(
    model="deepseek-flash",
    instructions="""
Ты учебный ассистент.

Ответь пользователю на основе результата
search_study_materials.
""",
    tools=tools,
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
)


print()
print("FINAL ANSWER:")
print(final_response.output_text)