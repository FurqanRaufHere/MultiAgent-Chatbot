from pydantic import BaseModel, Field
from typing import Optional, List
from datetime import datetime

class MemoryRecord(BaseModel):
    """Structured record schema for all knowledge persistence."""
    
    # Unique identifier
    id: str = Field(default_factory=lambda: datetime.now().isoformat())
    timestamp: datetime = Field(default_factory=datetime.now) # Required
    
    # Provenance and Metadata
    memory_type: str = Field(description="e.g., 'Conversation', 'Knowledge', 'AgentState'")
    agent_source: str = Field(description="The agent that created this record.")
    topic: str = Field(description="Extracted key topic/keywords for this finding.") # Required
    confidence: Optional[float] = Field(None, ge=0.0, le=1.0) # Confidence score
    source_refs: List[str] = Field(default_factory=list, description="List of original sources (snippets, URLs, queries).")

    # The Core Knowledge
    summary: str = Field(description="A concise, vectorized summary of the stored finding.")
    
    # Tracking
    is_used: bool = False