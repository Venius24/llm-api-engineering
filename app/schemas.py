from pydantic import BaseModel, Field


class ChatRequest(BaseModel):
    message: str = Field(min_length=1, max_length=4000)


class SearchArgs(BaseModel):
    query: str


class RouteDecision(BaseModel):
    use_rag: bool
    search_query: str
    reason: str
