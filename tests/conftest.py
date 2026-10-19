import os

# app.llm builds the client at import time; tests never reach the network.
os.environ.setdefault("DEEPSEEK_API_KEY", "test-key")
