# RAG Service: Community Summary Guide
 
## Overview

The RAG service generates AI-powered summaries of Boost mailing list discussions and displays them on the `/community/` page. This document explains how it works.

## Quick Start

**Generate a test summary:**
```bash
python manage.py generate_community_summary --test
```

**View the summary:**
Visit `http://localhost:8000/community/`

## Architecture

```
HyperKitty DB → ChromaDB → Summary Generation → PostgreSQL → Community Page
```

## Data Flow

### 1. Email Collection (Every 30 minutes)

Emails are automatically synced from HyperKitty to ChromaDB:

- **Task:** `sync_new_mails_to_vector_db()` (Celery)
- **Process:** Emails are embedded and stored for semantic search
- **Location:** `rag_service/tasks.py`

### 2. Summary Generation (Daily at 1:00 AM)

The system generates weekly summaries:

1. **Get recent emails** (last 7 days, ~200 emails)
2. **Extract topics** using one of three methods:
   - Thread-based (groups by conversation)
   - Clustering (semantic similarity)
   - LLM (direct extraction)
3. **For each topic:**
   - Retrieve relevant past emails using RAG
   - Generate chronological summary using LLM
4. **Save to database** as `CommunitySummary`

**Components:**
- `WeeklyCommunitySummaryGenerator` - Main orchestrator
- `TopicExtractor` - Extracts topics
- `MailDataRetriever` - Retrieves emails
- `LLMHelper` - Generates summaries

**Location:** `rag_service/langchain_rag/task/community_task.py`

### 3. Display on Community Page

When a user visits `/community/`:

1. `CommunitySummaryView` retrieves the active summary from database
2. Data is prepared for the template
3. `community.html` renders the summary

**Location:** `rag_service/views.py` and `templates/community.html`

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

### Run All Tests

```bash
pytest rag_service/tests/
```

### Run Specific Test Modules

```bash
# RAG pipeline tests
pytest rag_service/tests/tests_rag_pipeline/

# Specific test file
pytest rag_service/tests/tests_rag_pipeline/test_pipeline_basic.py

# Specific test class
pytest rag_service/tests/tests_rag_pipeline/test_pipeline_errors.py::TestRAGPipelineNetworkFailures
```

### Test Coverage

- **Basic Tests** (`test_pipeline_basic.py`): Initialization, singleton pattern, basic operations
- **Operations Tests** (`test_pipeline_operations.py`): CRUD operations, cache, statistics
- **Error Tests** (`test_pipeline_errors.py`): Network failures, LLM errors, retriever errors
- **Validation Tests** (`test_pipeline_validation.py`): Invalid data, edge cases, public methods

## How to Use

### Generate Test Summary

Creates dummy data for UI testing:

```bash
python manage.py generate_community_summary --test --deactivate-existing
```

### Generate Real Summary

Requires:
- ChromaDB with email data
- LLM API keys configured

```bash
python manage.py generate_community_summary --deactivate-existing
```

### Automated Generation

Summaries are generated automatically:
- **Daily at 1:00 AM** via Celery task `update_summary_data()`
- Configured in `config/celery.py`

## Data Structure

### Summary JSON Format

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

### WeeklyCommunitySummaryGenerator

**Location:** `rag_service/langchain_rag/task/community_task.py`

Main orchestrator for generating summaries.

**Configuration:**
```python
generator = WeeklyCommunitySummaryGenerator(
    source="chromadb",
    limit=200,                      # Max recent emails
    max_topics=5,                   # Max topics
    fetch_k=30,                     # Relevant emails per topic
    topic_extraction_method="thread"  # "thread", "clustering", or "llm"
)
```

### CommunitySummaryView

**Location:** `rag_service/views.py`

Django view that:
- Retrieves active summary from database
- Normalizes reference URLs
- Assigns continuous reference numbers
- Prepares data for template

## Troubleshooting

### No Summary Displayed

1. Check if summary exists:
   ```bash
   python manage.py shell -c \
     "from rag_service.models import CommunitySummary; \
      print(CommunitySummary.objects.filter(need_review=False).count())"
   ```

2. Generate test summary:
   ```bash
   python manage.py generate_community_summary --test
   ```

3. Check view: Ensure `CommunitySummaryView` retrieves summaries with `need_review=False`

### Summary Generation Fails

1. **ChromaDB:** Ensure emails are synced via `sync_new_mails_to_vector_db` task
2. **LLM:** Verify API keys and model availability
3. **Dependencies:** Check all RAG packages are installed

## Summary

The RAG service:
1. **Collects** emails from HyperKitty → ChromaDB (every 30 min)
2. **Generates** summaries from recent emails → topics → RAG → LLM (daily)
3. **Stores** summaries in PostgreSQL
4. **Displays** on `/community/` page

Runs automatically via Celery, or manually via management commands. Comprehensive tests ensure reliability across all components.
