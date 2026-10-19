import httpx

from app import config
from app.schemas import SearchArgs


def search_study_materials(query: str) -> dict:
    print(f"[TOOL] manual-rag query: {query}")

    response = httpx.post(
        config.RAG_URL,
        json={"question": query},
        timeout=config.RAG_TIMEOUT,
    )

    response.raise_for_status()

    return response.json()


TOOLS = [
    {
        "type": "function",
        "name": "search_study_materials",
        "description": (
            "Search the local study knowledge base. "
            "Use this tool for questions about mathematics "
            "or information that should come from study materials."
        ),
        "parameters": SearchArgs.model_json_schema(),
    }
]
