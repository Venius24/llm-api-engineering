# LLM API Engineering

A small LLM backend demonstrating structured outputs, tool calling, RAG integration, and streaming responses.

## Features

- Structured output with JSON Schema and Pydantic
- LLM function/tool calling
- Tool argument validation
- Integration with an external RAG service
- Streaming LLM responses
- FastAPI REST API
- Separation of API, LLM, tool, and schema layers

## Architecture

```text
User
  │
  ▼
POST /chat
  │
  ▼
DeepSeek
  │
  ├── answer directly
  │
  └── function call
          │
          ▼
  search_study_materials
          │
          ▼
  manual-rag :8000
          │
          ▼
      RAG result
          │
          ▼
       DeepSeek
          │
          ▼
   Streaming response
```

The RAG service is implemented separately in the `manual-rag` project.

## Project Structure

```text
app/
├── __init__.py
├── main.py
├── llm.py
├── tools.py
└── schemas.py
```

## Tech Stack

- Python
- FastAPI
- Pydantic
- OpenAI-compatible Responses API
- DeepSeek
- HTTPX

External tool:

- `manual-rag` running on `localhost:8000`

## Setup

Create a virtual environment:

```bash
python -m venv .venv
```

Activate it on Windows Git Bash:

```bash
source .venv/Scripts/activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Create `.env`:

```env
DEEPSEEK_API_KEY=your_api_key_here
```

## Run

First start the external `manual-rag` service on port `8000`.

Then run this API:

```bash
uvicorn app.main:app --reload --port 8001
```

Health check:

```bash
curl http://127.0.0.1:8001/health
```

## Chat API

Endpoint:

```text
POST /chat
```

Request:

```json
{
  "message": "What is a derivative?"
}
```

The response is streamed as plain text.

Example:

```bash
curl -N http://127.0.0.1:8001/chat --json '{"message":"test"}'
```

## Tool Calling Flow

The model receives the `search_study_materials` tool definition.

When the model decides that information from the study knowledge base is required:

1. The model returns a function call.
2. Tool arguments are validated with Pydantic.
3. The backend calls the external `manual-rag` API.
4. The tool result is returned to the model.
5. The final response is streamed to the client.

## What This Project Demonstrates

This project focuses on LLM application engineering rather than model training.

It demonstrates how to connect an LLM to external systems safely and turn model output into application logic using:

- schemas
- structured outputs
- function calling
- external tools
- validation
- streaming APIs

## Related Project

`manual-rag` — RAG pipeline with embeddings, PostgreSQL/pgvector, hybrid retrieval, reranking, and evaluation.