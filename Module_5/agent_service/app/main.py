from fastapi import FastAPI
from pydantic import BaseModel

from app.agent import ask_agent

app = FastAPI(
    title="Agent Service API",
    description="FastAPI service for LangChain Agent with tools and memory",
    version="1.0.0"
)

class QueryRequest(BaseModel):
    question: str
    session_id: str

class QueryResponse(BaseModel):
    answer: str

@app.post("/ask", response_model=QueryResponse)
def ask(request: QueryRequest):
    answer = ask_agent(
        message=request.question,
        thread_id=request.session_id
    )
    return {"answer": answer}
