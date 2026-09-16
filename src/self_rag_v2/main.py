from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import List
from .rag_engine import run_self_rag_pipeline

app = FastAPI(
    title="Self-RAG Gemini API",
    description="Self-Reflective RAG Backend using Google Gemini and LangGraph with LangSmith Tracing",
    version="1.0.0"
)

class ChatRequest(BaseModel):
    question: str = Field(..., example="Describe NexaAI's company culture.")

class ChatResponse(BaseModel):
    answer: str
    evidence: List[str] = Field(default_factory=list)
    is_useful: bool
    rewrite_tries: int

@app.post("/api/chat", response_model=ChatResponse)
def chat_endpoint(payload: ChatRequest):
    try:
        res = run_self_rag_pipeline(payload.question)
        return ChatResponse(
            answer=res["answer"],
            evidence=res["evidence"],
            is_useful=res["is_useful"],
            rewrite_tries=res["rewrite_tries"]
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
async def health_check():
    return {"status": "healthy"}