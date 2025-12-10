"""Cache data schemas.

Defines Pydantic models for agent response cache entries.
"""

from datetime import datetime

from pydantic import BaseModel, Field


class CacheEntry(BaseModel):
    """Schema for cache entries."""

    cache_id: str = Field(..., description="Unique cache entry ID (UUID)")
    query: str = Field(..., description="Original user query")
    response: str = Field(..., description="Agent response")
    agent_type: str = Field(
        ...,
        description="Agent that handled the query",
        examples=["supervisor", "billing", "tech_support", "product", "account"],
    )
    customer_segment: str = Field(
        ...,
        description="Customer segment",
        examples=["Individual", "Family", "Business", "Premium", "Student"],
    )
    formatted_content: str = Field(
        ..., description="Formatted content for embeddings (query + context)"
    )
    hit_count: int = Field(
        default=0, ge=0, description="Number of times this cache entry was matched"
    )
    last_hit_time: datetime = Field(
        ..., description="Last time this entry was accessed/matched"
    )
    created_time: datetime = Field(..., description="When this entry was created")

