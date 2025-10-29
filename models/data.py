"""
Data models for the RAG system.
These represent data structures, not configurations.
"""

from dataclasses import dataclass
from typing import Dict, Any


@dataclass
class RetrievalResult:
    """Result from hybrid retrieval."""
    text: str
    score: float
    metadata: Dict[str, Any]
    retrieval_method: str  # 'embedding', 'bm25', 'graph', 'hybrid', 'hierarchical'
    source_type: str  # 'code', 'text', 'email'
    source_file: str


__all__ = ["RetrievalResult"]

