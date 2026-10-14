from pydantic import BaseModel


class ChatRequest(BaseModel):
    message: str


class SearchArgs(BaseModel):
    query: str