"""
Pydantic models for API requests and responses.
"""

from typing import Dict, Any, Optional, List
from pydantic import BaseModel, Field


# ============================================================================
# Request Models
# ============================================================================

class UpdateRequest(BaseModel):
    """Request to update the knowledge base with new content."""
    new_source_url: str = Field(..., description="New URL to scrape")
    max_depth: int = Field(2, description="Maximum depth for following links")
    delay: float = Field(1.0, description="Delay between requests in seconds")


class QueryRequest(BaseModel):
    """Request to query the RAG system."""
    question: str = Field(..., description="Question to ask")
    search_scopes: List[str] = Field(
        default=["documentation", "mail", "slack"],
        description="Search scopes to include (documentation, mail, slack, etc.)"
    )
    search_limit: int = Field(
        10,
        description="Number of search results to return (5, 10, 15, 20, 30, 40, 50, 100)"
    )
    offset: int = Field(0, description="Start offset for retrieval results (deprecated, use search_limit)")
    limit: int = Field(None, description="Page size for retrieval results (deprecated, use search_limit)")


class SearchQueryRequest(BaseModel):
    """Request for search operation."""
    query: str = Field(..., description="Search query")
    offset: int = Field(0, description="Start offset for retrieval results")
    limit: int = Field(10, description="Page size for retrieval results")


class ChannelRequest(BaseModel):
    """Request to create or manage a channel."""
    channel_id: str = Field(..., description="Channel ID")
    channel_name: Optional[str] = Field(None, description="Channel name")


# ============================================================================
# Response Models
# ============================================================================

class QueryResponse(BaseModel):
    """Response from a query operation."""
    question: str
    answer: str
    retrieval_results: Optional[List[Dict[str, Any]]] = Field(
        None, description="Raw retrieval results used for answer generation"
    )
    timestamp: str


class SearchResponse(BaseModel):
    """Response from a search operation."""
    query: str
    results: List[Dict[str, Any]]
    total_results: int
    search_time: float
    timestamp: str


class ChannelResponse(BaseModel):
    """Response with channel information."""
    channel_id: str
    channel_name: str
    message_count: int
    last_activity: str
    created_at: str


class StatusResponse(BaseModel):
    """Response with system status information."""
    pipeline_stats: Dict[str, Any]
    config_loaded: bool
    components_initialized: Dict[str, bool]
    rag_statistics: Optional[Dict[str, Any]]


class SuccessResponse(BaseModel):
    """Generic success response."""
    success: bool = True
    message: str
    data: Dict[str, Any]
    timestamp: str


class ErrorResponse(BaseModel):
    """Generic error response."""
    success: bool = False
    error: Dict[str, Any]
    timestamp: str


class ActionStatsResponse(BaseModel):
    """Response with action statistics."""
    success: bool
    message: str
    data: Dict[str, Any]
    timestamp: str


# ============================================================================
# Mail/Thread Related Models
# ============================================================================

class ThreadInfo(BaseModel):
    """Information about an email thread."""
    url: str = Field(..., description="Thread URL")
    thread_id: str = Field(..., description="Thread ID")
    subject: str = Field(..., description="Thread subject")
    date_active: str = Field(..., description="Thread active date")
    starting_email: str = Field(..., description="Starting email URL")
    emails_url: str = Field(..., description="Emails URL")
    replies_count: int = Field(0, description="Number of replies")
    votes_total: int = Field(0, description="Total votes")


class MessageData(BaseModel):
    """Data for an individual email message."""
    message_id: str = Field(..., description="Unique message ID")
    subject: str = Field(..., description="Message subject")
    content: str = Field(..., description="Message content")
    thread_url: str = Field(..., description="Thread URL")
    parent: Optional[str] = Field(None, description="Parent message URL")
    children: List[str] = Field(
        default_factory=list, description="Children message URLs"
    )
    sender_address: str = Field(..., description="Sender email address")
    from_field: str = Field(..., description="From field")
    date: str = Field(..., description="Message date")
    to: str = Field(..., description="To field")
    cc: str = Field(default="", description="CC field")
    reply_to: str = Field(default="", description="Reply-To field")
    url: str = Field(..., description="Message URL")


class EmailThreadRequest(BaseModel):
    """Request to add an email thread."""
    timestamp: str = Field(..., description="Request timestamp")
    requestId: str = Field(..., description="Unique request identifier")
    thread_info: ThreadInfo = Field(..., description="Thread information")
    messages: List[MessageData] = Field(..., description="List of messages")
    message_count: int = Field(..., description="Total message count")


class EmailMessagesRequest(BaseModel):
    """Request to add multiple email messages."""
    timestamp: str = Field(..., description="Request timestamp")
    requestId: str = Field(..., description="Unique request identifier")
    messages: List[MessageData] = Field(..., description="List of messages")
    message_count: int = Field(..., description="Total message count")


class UpdateMessageRequest(BaseModel):
    """Request to update an existing message."""
    message_id: str = Field(..., description="Message ID to update")
    subject: Optional[str] = Field(None, description="Updated subject")
    content: Optional[str] = Field(None, description="Updated content")
    sender_address: Optional[str] = Field(None, description="Updated sender address")
    from_field: Optional[str] = Field(None, description="Updated from field")
    date: Optional[str] = Field(None, description="Updated date")
    to: Optional[str] = Field(None, description="Updated to field")
    cc: Optional[str] = Field(None, description="Updated CC field")
    reply_to: Optional[str] = Field(None, description="Updated Reply-To field")
    url: Optional[str] = Field(None, description="Updated URL")


class DeleteMessageRequest(BaseModel):
    """Request to delete a message."""
    message_id: str = Field(..., description="Message ID to delete")
    url: str = Field(..., description="Message URL to delete")


class NewThreadRequest(BaseModel):
    """Request to create a new thread."""
    thread_id: str = Field(..., description="Unique thread ID")
    subject: str = Field(..., description="Thread subject")
    url: Optional[str] = Field(None, description="Thread URL")
    date_active: Optional[str] = Field(None, description="Thread active date")
    starting_email: Optional[str] = Field(None, description="Starting email ID")
    emails_url: Optional[str] = Field(None, description="Emails URL")
    replies_count: int = Field(0, description="Number of replies")
    votes_total: int = Field(0, description="Total votes")


# ============================================================================
# Miscellaneous Models
# ============================================================================

class ModelInfo(BaseModel):
    """Information about a model."""
    name: str
    size: int
    parameter_size: str

