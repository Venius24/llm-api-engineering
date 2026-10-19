import os

from dotenv import load_dotenv


load_dotenv()


LLM_BASE_URL = os.getenv("LLM_BASE_URL", "https://api.deepseek.com")
LLM_MODEL = os.getenv("LLM_MODEL", "deepseek-flash")
LLM_TIMEOUT = float(os.getenv("LLM_TIMEOUT", "30"))
LLM_MAX_RETRIES = int(os.getenv("LLM_MAX_RETRIES", "2"))

# manual-rag endpoint used by the search_study_materials tool
RAG_URL = os.getenv("RAG_URL", "http://127.0.0.1:8000/ask")
RAG_TIMEOUT = float(os.getenv("RAG_TIMEOUT", "60"))

# How many rounds of tool calls one /chat request may run
MAX_TOOL_ROUNDS = int(os.getenv("MAX_TOOL_ROUNDS", "3"))
