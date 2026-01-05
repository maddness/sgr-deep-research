"""HTTP API routes for testing the agent."""
import logging
import time
from typing import Optional

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from bot.src.sgr.agent import DeepResearchAgent, ResearchResult

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api", tags=["chat"])

# Will be set from app.py
_agent: Optional[DeepResearchAgent] = None


def set_agent(agent: DeepResearchAgent) -> None:
    """Set the agent instance for API routes."""
    global _agent
    _agent = agent


class ChatRequest(BaseModel):
    """Request model for chat endpoint."""
    query: str = Field(..., min_length=5, description="Research query")
    user_id: Optional[str] = Field(None, description="User ID for tracing")
    session_id: Optional[str] = Field(None, description="Session ID for tracing")


class ChatResponse(BaseModel):
    """Response model for chat endpoint."""
    result: str = Field(..., description="Research result")
    tools_used: list[str] = Field(default_factory=list, description="Tools used during research")
    iterations: int = Field(0, description="Number of agent iterations")
    duration_seconds: float = Field(0.0, description="Request duration")
    error: Optional[str] = Field(None, description="Error message if failed")


@router.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest) -> ChatResponse:
    """
    Send a research query to the agent.

    Returns structured research result with sources.
    """
    if _agent is None:
        raise HTTPException(status_code=503, detail="Agent not initialized")

    start_time = time.time()
    tools_used: list[str] = []
    result_content = ""
    iterations = 0
    error_msg: Optional[str] = None

    try:
        logger.info(f"API request: {request.query[:50]}...")

        async for result in _agent.research(
            request.query,
            user_id=request.user_id,
            session_id=request.session_id,
        ):
            if result.tools_used:
                tools_used = result.tools_used
            if result.content:
                result_content = result.content
            if result.iterations:
                iterations = result.iterations
            if result.error:
                error_msg = result.error
            if result.needs_clarification:
                result_content = f"Требуется уточнение: {result.clarification_question}"
                break

    except Exception as e:
        logger.error(f"API error: {e}")
        error_msg = str(e)

    duration = time.time() - start_time

    return ChatResponse(
        result=result_content or "No result",
        tools_used=tools_used,
        iterations=iterations,
        duration_seconds=round(duration, 2),
        error=error_msg,
    )


@router.get("/health")
async def health() -> dict:
    """Health check endpoint."""
    return {
        "status": "ok",
        "agent_initialized": _agent is not None,
    }
