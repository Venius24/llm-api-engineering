# LLM API Engineering

[![tests](https://github.com/Venius24/llm-api-engineering/actions/workflows/tests.yml/badge.svg)](https://github.com/Venius24/llm-api-engineering/actions/workflows/tests.yml)

A small FastAPI backend around an LLM (DeepSeek, OpenAI-compatible Responses API):
a tool-calling loop that queries an external RAG service, streamed answers, and a
structured-output endpoint validated with Pydantic.

## Features

- **Tool calling loop**: the model can call `search_study_materials` several times,
  including several calls in one turn, up to `MAX_TOOL_ROUNDS` rounds per request.
- **Tool errors go back to the model**: invalid arguments (Pydantic validation), an
  unknown tool name or a failed RAG request become a `{"error": ...}` tool result,
  so the model can explain the problem instead of the request crashing.
- **Streaming**: after a tool round the answer is streamed as text deltas. A direct
  answer (no tool needed) comes from a single non-streamed call and is sent as one chunk.
- **Structured output**: `POST /route` asks the model for a `RouteDecision`
  (JSON Schema from Pydantic) and validates the reply; malformed output is a 502.
- **Provider failures**: client timeout and retries are configurable; if the LLM is
  unreachable, `/chat` and `/route` return 503 before any output is sent.
- Offline test suite (28 tests) and CI.

## Architecture

```text
POST /chat ─► DeepSeek (tools offered)
                 │
                 ├── text ─────────────────────────────► answer (one chunk)
                 │
                 └── function call(s)
                        │
                        ▼
                validate args ─► search_study_materials ─► manual-rag /ask
                        │                     (errors become tool results)
                        ▼
                DeepSeek, streamed ─► text deltas ─► client
                        │
                        └── another function call? repeat, up to MAX_TOOL_ROUNDS

POST /route ─► DeepSeek with json_schema format ─► RouteDecision (Pydantic)
```

The RAG service is the separate [manual-rag](https://github.com/Venius24/manual-rag)
project. Its `/ask` endpoint already generates an answer with an LLM, so a tool call
costs a second LLM round trip; a retrieval-only endpoint would be cheaper.

## Project structure

```text
app/
├── main.py       # FastAPI: /chat, /route, /health
├── llm.py        # client, tool-calling loop, structured route decision
├── tools.py      # search_study_materials and its tool definition
├── schemas.py    # request models, tool arguments, RouteDecision
└── config.py     # settings from environment variables
examples/
├── tool_use.py   # step-by-step tool calling script with printed intermediate steps
└── structured.py # standalone structured output script
tests/            # offline pytest suite
```

## Configuration

Set in `.env` (see `.env.example`) or the environment:

| Variable | Default | Meaning |
|---|---|---|
| `DEEPSEEK_API_KEY` | — (required) | API key |
| `LLM_BASE_URL` | `https://api.deepseek.com` | OpenAI-compatible endpoint |
| `LLM_MODEL` | `deepseek-flash` | model name |
| `LLM_TIMEOUT` / `LLM_MAX_RETRIES` | `30` / `2` | client timeout (s) and retries |
| `RAG_URL` / `RAG_TIMEOUT` | `http://127.0.0.1:8000/ask` / `60` | manual-rag endpoint and timeout (s) |
| `MAX_TOOL_ROUNDS` | `3` | tool rounds per `/chat` request |

## Setup and run

```bash
python -m venv .venv
source .venv/Scripts/activate      # Windows Git Bash; .venv/bin/activate on Linux/macOS
pip install -r requirements.txt
cp .env.example .env               # then set DEEPSEEK_API_KEY
```

Start manual-rag on port 8000, then this API:

```bash
uvicorn app.main:app --reload --port 8001
```

## API

### POST /chat

```bash
curl -N http://127.0.0.1:8001/chat --json '{"message":"What is a derivative?"}'
```

The response is plain text, streamed. `message` must be 1–4000 characters (422 otherwise).

### POST /route

```bash
curl http://127.0.0.1:8001/route --json '{"message":"What is a derivative?"}'
```

```json
{"use_rag": true, "search_query": "derivative definition", "reason": "..."}
```

### GET /health

Returns `{"status": "ok"}`.

## Tests

```bash
pip install -r requirements-dev.txt
pytest
```

The suite runs offline: the LLM client and the manual-rag HTTP call are replaced
with fakes. It covers the direct answer, the tool round trip (each request replays
the full tool history), parallel calls, a second tool round and the round limit,
tool errors for invalid arguments, unknown tools and RAG failures, streaming of
text deltas only, the structured route decision, input validation, and the 502/503
mappings. It does not call the real DeepSeek API.

## Tech stack

Python · FastAPI · Pydantic · OpenAI Python SDK (Responses API) · DeepSeek · HTTPX · pytest
