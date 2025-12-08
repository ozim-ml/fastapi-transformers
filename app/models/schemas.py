from typing import List, Optional
from pydantic import BaseModel, Field

class ClassifyRequest(BaseModel):
    subject: str = Field(..., description="Email subject")
    body: str = Field(..., description="Email body")

class ChunkResult(BaseModel):
    chunk: str
    top_label: str
    top_score: float

class TaskResult(BaseModel):
    task_index: int
    selected_label: Optional[str]
    selected_score: Optional[float]
    chunks: List[ChunkResult]

class ClassifyResponse(BaseModel):
    results: List[TaskResult]
