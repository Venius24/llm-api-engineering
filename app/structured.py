import os

from dotenv import load_dotenv
from openai import OpenAI
from pydantic import BaseModel


load_dotenv()


class RouteDecision(BaseModel):
    use_rag: bool
    search_query: str
    reason: str


client = OpenAI(
    api_key=os.environ["DEEPSEEK_API_KEY"],
    base_url="https://api.deepseek.com",
)


question = "Что такое производная?"


response = client.responses.create(
    model="deepseek-flash",

    instructions="""
Ты маршрутизатор вопросов.

Определи, нужно ли искать ответ в нашей учебной базе знаний.

Если нужен поиск:
- use_rag = true
- search_query = хороший поисковый запрос

Если поиск не нужен:
- use_rag = false
- search_query = ""

Кратко объясни решение в reason.
""",

    input=question,

    text={
        "format": {
            "type": "json_schema",
            "name": "route_decision",
            "schema": RouteDecision.model_json_schema(),
        }
    },
)


decision = RouteDecision.model_validate_json(response.output_text)

print(decision)
print()
print(decision.model_dump())