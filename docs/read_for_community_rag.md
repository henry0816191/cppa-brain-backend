# RAG Service: Community Summary Guide

## Overview

Generates AI-powered summaries of Boost mailing list discussions and displays them on `/community/`.

**Quick Start:**
```bash
python manage.py generate_community_summary --test
# Visit http://localhost:8000/community/
```

## Architecture

```
HyperKitty DB → ChromaDB → Summary Generation → PostgreSQL → Community Page
```

## Data Flow

### 1. Email Collection (Every 30 min)
- **Task:** `sync_new_mails_to_vector_db()` (Celery)
- Syncs emails from HyperKitty to ChromaDB for semantic search
- **Location:** `rag_service/tasks.py`

### 2. Summary Generation (Daily at 1:00 AM)
1. Get recent emails (last 7 days, ~100 emails)
2. Extract topics (thread-based, clustering, or LLM)
3. For each topic: retrieve past emails via RAG → generate summary via LLM
4. Save as `CommunitySummary` to database

**Components:** `WeeklyCommunitySummaryGenerator`, `TopicExtractor`, `MailDataRetriever`, `LLMHelper`  
**Location:** `rag_service/langchain_rag/task/community_task.py`

### 3. Display
`CommunitySummaryView` retrieves active summary → prepares data → renders in `community.html`  
**Location:** `rag_service/views.py`, `templates/community.html`

## Project Structure

```
rag_service/
├── models.py                    # CommunitySummary model
├── views.py                     # CommunitySummaryView
├── tasks.py                     # Celery tasks
├── services.py                  # RAG service initialization
├── s3_utils.py                  # S3 backup utilities
├── management/commands/
│   └── generate_community_summary.py
├── langchain_rag/
│   ├── rag_pipeline.py          # Main RAG pipeline
│   ├── task/
│   │   ├── community_task.py    # Summary generator
│   │   ├── mail_data_retriever.py
│   │   └── topic_extractor.py
│   ├── retrieve/                # Retrieval components
│   ├── preprocessor/            # Data preprocessing
│   └── llm/                     # LLM integration
└── tests/
    ├── test_models.py           # Model tests
    ├── test_views.py            # View tests
    ├── test_tasks.py            # Task tests
    ├── test_commands.py         # Command tests
    └── tests_rag_pipeline/      # RAG pipeline integration tests
        ├── test_pipeline_basic.py      # Initialization & basic ops
        ├── test_pipeline_operations.py # CRUD operations
        ├── test_pipeline_errors.py     # Error handling
        └── test_pipeline_validation.py # Data validation
```

## Testing

```bash
# Run all tests
pytest rag_service/tests/

# RAG pipeline tests
pytest rag_service/tests/tests_rag_pipeline/

# Specific test file
pytest rag_service/tests/tests_rag_pipeline/test_pipeline_basic.py
```

## Usage

**Test Summary (dummy data):**
```bash
python manage.py generate_community_summary --test --deactivate-existing
```

**Real Summary (requires ChromaDB + LLM API keys):**
```bash
python manage.py generate_community_summary --deactivate-existing
```

**Automated:** Daily at 1:00 AM via Celery task `update_summary_data()` (`config/celery.py`)

## Data Structure

**Summary JSON:**
```json
{
  "summary_by_topic": [
    {
      "subject": "Topic Name",
      "assertions": [
        {
          "content": "Key point...",
          "reference_url": ["url1", "url2"]
        }
      ],
      "chronological_summary": [
        {
          "date": "2025-11-06",
          "summary": "What happened...",
          "reference_url": ["url1"]
        }
      ]
    }
  ],
  "overall_stats": {
    "topics_count": 5,
    "recent_emails": 200,
    "date_range": {
      "start": "2025-11-06T00:00:00",
      "end": "2025-11-13T00:00:00"
    }
  }
}
```

### Database Model

```python
CommunitySummary(
    start_date=datetime(...),      # Summary period start
    end_date=datetime(...),        # Summary period end
    summary_data={...},            # Full JSON above
    topics_count=5,
    recent_emails_count=200,
    generated_at=datetime(...),
    is_active=True,                # Only active summary is displayed
    need_review=False              # Review flag
)
```

## Key Components

**WeeklyCommunitySummaryGenerator** (`rag_service/langchain_rag/task/community_task.py`):
- Main orchestrator for summary generation
- Config: `source="chromadb"`, `limit=200`, `max_topics=5`, `fetch_k=30`, `topic_extraction_method="thread"|"clustering"|"llm"`

**CommunitySummaryView** (`rag_service/views.py`):
- Retrieves active summary, normalizes URLs, assigns reference numbers

## Troubleshooting

**No Summary Displayed:**
```bash
# Check if summary exists
python manage.py shell -c "from rag_service.models import CommunitySummary; print(CommunitySummary.objects.filter(need_review=False).count())"

# Generate test summary
python manage.py generate_community_summary --test
```

**Generation Fails:**
- ChromaDB: Ensure emails synced via `sync_new_mails_to_vector_db`
- LLM: Verify API keys and model availability
- Dependencies: Check RAG packages installed

## Summary

1. **Collect** emails: HyperKitty → ChromaDB (every 30 min)
2. **Generate** summaries: recent emails → topics → RAG → LLM (daily at 1 AM)
3. **Store** in PostgreSQL
4. **Display** on `/community/` page

Runs automatically via Celery or manually via management commands.
