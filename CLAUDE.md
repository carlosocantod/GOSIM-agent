# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Repo layout

Main worktree: `/Users/cocanto/git_repos/GOSIM-agent`

Feature branches are developed in `.claude/worktrees/*`. Git commands for the current branch work directly from the worktree. To commit to `main`, use `git -C /Users/cocanto/git_repos/GOSIM-agent` or work from the main worktree directly. Never commit directly to `main` — use a feature branch and PR.

## Commands

```bash
# Run the app
uv run streamlit run src/streamlit_app.py

# Run with Docker
docker compose up --build

# Add a dependency
uv add <package>
```

No test suite exists yet. The `src/palyground/` directory contains throwaway scripts for manual exploration.

## Environment

Copy `.env.sample` to `.env` and fill in:
- `OPEN_ALEX_KEY` — OpenAlex API key (free, required)
- `URL_BASE` / `LLM_API_KEY` / `LLM_MODEL` — any OpenAI-compatible LLM endpoint
- `TOPIC_MODEL_K`, `TOPIC_MODEL_MAX_DOC_CHARS`, `TOPIC_MODEL_EMBEDDING_MODEL` — optional tuning

## Architecture

The entire application is `src/streamlit_app.py`. The pipeline runs in this order on each query:

1. **Keyword extraction** (`src/utils/llm.py` → `extract_medical_keywords`) — LLM classifies the query as medical/non-medical and returns search keywords.
2. **Paper fetch** (`src/utils/open_alex.py` → `get_openalex_papers_for_period`) — queries OpenAlex for papers matching the keywords within a date window. `last_n_months_date_range` and `previous_period_date_range` compute current and prior windows.
3. **Semantic rerank** (`src/utils/topic_model_llm.py` → `semantic_rerank`) — SBERT embeds abstracts and scores them against the query; keeps top N.
4. **Topic model** (`src/utils/topic_model_llm.py` → `run_topic_model`) — BERTopic with KMeans (no UMAP), followed by LLM-driven topic merging, relevance filtering, and summarisation.
5. **Period comparison** (`compare_topic_periods`) — LLM compares current vs previous period topic sets to surface emerging and disappeared topics.

The SBERT model is loaded once via `@st.cache_resource` in `streamlit_app.py` and passed as an argument into `semantic_rerank` and `run_topic_model` — never loaded inside those functions.

Expensive steps (paper fetch, topic pipeline, period comparison) are cached with `@st.cache_data` keyed on query/keywords/date range. The pipeline runs progressively: current period topics are displayed before the previous period pipeline starts.

## Deployment

Pushing to `main` automatically syncs to Hugging Face Spaces via the GitHub Actions workflow in `.github/workflows/`. The Space runs the Docker image.

## Commit style

Append `[ai]` at the end of the commit subject line for AI-authored commits.
