# RAG to Django Migration - Master Plan

## Overview

This is the master plan for migrating the RAG system from a separate FastAPI service into the Django website-v2 backend.

**Current Architecture**: Separate FastAPI RAG service (`localhost:8080`) + Django website
**Target Architecture**: RAG integrated directly into Django website-v2
**Migration Approach**: Direct migration - API service will be removed, no maintenance during migration

---

## Quick Start: Implementation Order

**🎯 START HERE: Community Summary**

See **[Task 1: Community Summary](1_CommunitySummary.md)** for detailed implementation.

### Priority Order

1. **[Task 1: Community Page Weekly Summary](1_CommunitySummary.md)** ⭐ START HERE

   - RAG-by-topic approach to reduce hallucination
   - Weekly summary on Community page
   - Runs Sunday via Celery, reviewed before publishing

2. **[Task 2: Button-Triggered Library Summaries](2_LibrarySummaries.md)** (To be created)

   - Users click button to get AI summaries
   - Discussion summaries, Historical context, FAQ generation

3. **[Task 3: AI-Generated FAQs](3_FAQs.md)** (To be created)

   - Lowest-hanging fruit after community summary
   - FAQ generation for libraries

4. **[Task 4: Historical Section](4_HistoricalSection.md)** (To be created)

   - Library acceptance process, major decisions, concerns
   - Separate section on library homepage

5. **[Task 5: Semantic Search](5_SemanticSearch.md)** (To be created)

   - Third search option or replace all search

6. **[Task 6: Wagtail Review Interface](6_WagtailReview.md)** (To be created)
   - PR-like approval workflow
   - FSC committee and general management review

---

## Architecture Overview

### Current Components

**Location**: `rag/langchain_rag/`

- `rag_pipeline.py` - Main RAG pipeline
- `email_classifier.py` - Email classification
- `llm_helper.py` - LLM helper
- `llm_agent.py` - LLM agents (OpenAI, HuggingFace)
- `hybrid_retriever.py` - Hybrid retrieval system
- `data_processor.py` - Data processing
- `caching_telemetry.py` - Caching and telemetry

**Data Storage**:

- ChromaDB vector store (`rag/langchain_rag/chroma_db/`)
- Email data source: PostgreSQL database (HyperKitty) - after migration
- Configuration files (`rag/config/`)

### Target Django App

**Django App**: Create dedicated `rag/` app
**Important**: RAG will extend to documentation and Slack data, so it should NOT be in `mailing_list/` app

- `rag/` app: All RAG functionality (mailing list, documentation, Slack, extensible)
  - Core RAG components (pipeline, classifier, LLM helper, retriever, etc.)
  - Data source processors (mailing_list, documentation, slack)
  - Service wrapper (`RAGService`) that other apps can import
- `mailing_list/` app: Mailing list-specific functionality only
  - Imports `rag.services.RAGService` when needed
  - Does NOT contain RAG code itself

**Directory Structure**:

```
website-v2/
├── rag/  # Dedicated RAG app
│   ├── core/         # Core RAG components
│   ├── data_sources/ # Data source processors
│   ├── processors/   # Post-processors
│   ├── services.py   # RAGService (imported by other apps)
│   └── tasks.py      # Celery tasks
└── mailing_list/     # Mailing list app (uses RAG service)
```

---

## Migration Strategy

### Client's Requirements (from AI-enabled Website Meeting, Oct 31, 2025)

**MVP Priority - Quick Iteration**:

1. Start with Mailing List Data Only (don't wait for GitHub/Slack integration)
2. Button-Triggered Summaries (users click button, not automatic)
3. AI-Generated FAQs (lowest-hanging fruit)
4. Community Page Summary (weekly/monthly, not per-library)
5. Historical Section (library acceptance process)
6. Weekly Generation (runs Sunday, reviewed by FSC)
7. PR-like Approval Workflow (prevents unilateral approvals)
8. Wagtail Integration (for review interface)
9. Semantic Search (third option or replace all)

### Principles

1. **MVP First**: Quick MVP with mailing list data, iterate based on feedback
2. **Direct Migration**: No API service maintenance during migration
3. **Incremental Migration**: Migrate component by component, prioritize MVP features
4. **Data Preservation**: Ensure ChromaDB and email data remain intact
5. **Human Review Required**: All AI-generated content requires human review before publishing

---

## Key Decisions Needed

- Semantic search: third option or replace all?
- Community summary timeframe: 7 days or 30 days?
- Wagtail MVP: Consult Greg (estimated 1 week)
- Approval workflow: Existing Django moderation or custom PR-like system?

---

## Data Migration

### Step 1: Migrate ChromaDB

**Options**:

1. Keep existing ChromaDB: Point Django to existing `rag/langchain_rag/chroma_db/`
2. Copy ChromaDB: Copy to Django project directory
3. Reindex: Reindex from scratch (slowest but cleanest)

**Recommended**: Option 1 initially, migrate later if needed

### Step 2: Configure Email Data Source

**Important**: After migration, email data comes from PostgreSQL database (HyperKitty), not JSON files.

- JSON files were only for initial ChromaDB build
- RAG system reads email data from HyperKitty PostgreSQL database
- No need to copy JSON files to Django project
- Use `rag/data_sources/mailing_list.py` to query HyperKitty database directly

### Step 3: Update Configuration

Copy config files and update paths for Django project structure.

---

## Testing & Validation

### Testing Strategy

1. **Unit Tests**: Test individual components (LLM agent, retriever, etc.)
2. **Integration Tests**: Test Django views and Celery tasks
3. **Performance Testing**: Measure query latency, cache effectiveness
4. **Data Validation**: Verify ChromaDB contents, HyperKitty connection

---

## Deployment

### Pre-Deployment Checklist

- [ ] All RAG code migrated to Django
- [ ] Tests passing
- [ ] ChromaDB migrated
- [ ] HyperKitty database connection configured
- [ ] Dependencies installed
- [ ] Configuration updated
- [ ] Cache configured
- [ ] Celery tasks working
- [ ] Management commands tested

### Deployment Steps

1. Backup existing data (ChromaDB, HyperKitty database)
2. Deploy Django code
3. Initialize RAG pipeline (if needed)
4. Start services (Django, Celery)
5. Verify endpoints

### Direct Migration

**Direct Approach**: No API service maintenance

- Migrate RAG code directly to Django
- Update frontend to use Django endpoints immediately
- Remove FastAPI service code after migration is complete
- No parallel running or gradual cutover needed

---

## Rollback Plan

### Immediate Rollback

If critical issues occur:

1. Revert Django code: `git revert <migration-commit>`
2. Revert Django migration: `python manage.py migrate`

### Partial Rollback

Keep Django RAG but disable problematic features:

- Disable specific endpoints
- Fallback to cached results
- Use keyword-based classification only

---

## Success Criteria

✅ All RAG functionality available in Django
✅ Query performance meets requirements
✅ All tests passing
✅ No data loss
✅ Deployment completed successfully
✅ All RAG functionality integrated into Django

---

## Related Documents

- **[Task 1: Community Summary](1_CommunitySummary.md)** - Detailed implementation guide ⭐ START HERE
- **[Task 2: Library Summaries](2_LibrarySummaries.md)** - Button-triggered AI summaries on library pages
- **[Task 3: FAQs](3_FAQs.md)** - AI-generated FAQs for libraries
- **[Task 4: Historical Section](4_HistoricalSection.md)** - Historical context on library homepage
- **[Task 5: Semantic Search](5_SemanticSearch.md)** - Semantic search implementation
- **[Task 6: Wagtail Review](6_WagtailReview.md)** - PR-like approval workflow for AI summaries

---

## Notes

- **Lazy Loading**: Initialize RAG pipeline on first use (not at Django startup)
- **Caching**: Use Django cache framework for query results
- **Background Tasks**: Use Celery for heavy RAG operations (indexing, syncing)
- **Configuration**: Use Django settings instead of separate config files
- **Logging**: Integrate with Django logging system
- **Error Handling**: Wrap RAG calls with try/except and return Django-friendly errors
