import httpx

from app.schemas import SearchArgs


def search_study_materials(query: str) -> dict:
    print(f"[TOOL] manual-rag query: {query}")

    response = httpx.post(
        "http://127.0.0.1:8000/ask",
        json={"question": query},
        timeout=60,
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