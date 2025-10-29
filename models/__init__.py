"""
Data models for the RAG system.
"""

from .data import RetrievalResult
from .api import (
    # Request models
    QueryRequest,
    ChannelRequest,
    # Response models
    QueryResponse,
    ChannelResponse,
    SuccessResponse,
    # Mail/Thread models
    MessageData,
    EmailThreadRequest,
    DeleteMessageRequest,
)

__all__ = [
    # Internal data models
    "RetrievalResult",
    # Request models
    "QueryRequest",
    "ChannelRequest",
    # Response models
    "QueryResponse",
    "ChannelResponse",
    "SuccessResponse",
    # Mail/Thread models
    "MessageData",
    "EmailThreadRequest",
    "DeleteMessageRequest",
]

