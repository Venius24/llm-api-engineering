from fastapi import FastAPI
from fastapi.responses import StreamingResponse

from app.llm import generate_answer
from app.schemas import ChatRequest


app = FastAPI(
    title="LLM API Engineering",
    version="1.0.0",
)


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/chat")
def chat(request: ChatRequest):
    return StreamingResponse(
        generate_answer(request.message),
        media_type="text/plain; charset=utf-8",
    )