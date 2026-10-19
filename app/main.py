from itertools import chain

from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from openai import APIError
from pydantic import ValidationError

from app.llm import generate_answer, route_question
from app.schemas import ChatRequest, RouteDecision


app = FastAPI(
    title="LLM API Engineering",
    version="1.1.0",
)


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/chat")
def chat(request: ChatRequest):
    chunks = generate_answer(request.message)

    # The first LLM call happens before the response starts, so provider
    # errors become a 503 instead of a stream that breaks mid-way.
    try:
        first = next(chunks, "")
    except APIError:
        raise HTTPException(status_code=503, detail="LLM provider unavailable")

    return StreamingResponse(
        chain([first], chunks),
        media_type="text/plain; charset=utf-8",
    )


@app.post("/route", response_model=RouteDecision)
def route(request: ChatRequest):
    try:
        return route_question(request.message)
    except APIError:
        raise HTTPException(status_code=503, detail="LLM provider unavailable")
    except ValidationError:
        raise HTTPException(status_code=502, detail="LLM returned an invalid route decision")
