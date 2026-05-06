---
title: GOSIM Agent
emoji: 🚀
colorFrom: red
colorTo: red
sdk: docker
app_port: 8501
tags:
- streamlit
pinned: false
short_description: Streamlit template space
license: mit
---

## Medical Science Communication Helper Agent

A Streamlit app for the GOSIM hackathon, focused on medical science communication.

### Pre-hackathon setup
Before the hackathon, the following infrastructure was put in place:
- Streamlit app scaffold.
- Automatic deployment to Hugging Face Spaces via GitHub Actions (push to `main` triggers sync).
- OpenAlex API integration for querying recent biomedical literature by keyword.

### Requirements
- Python 3.13+
- `OPEN_ALEX_KEY` set in your `.env` file (get a free key at [openalex.org/settings/api](https://openalex.org/settings/api))

---

## TopicFlow

> **LangGraph-powered research signal tracker and science communication assistant for medical use cases**

🚀 **Live demo:** [huggingface.co/spaces/carlosocantod/GOSIM-agent](https://huggingface.co/spaces/carlosocantod/GOSIM-agent)

---

### The problem

Clinicians and medical researchers are drowning in literature. Approximately 1.5 million biomedical papers are published every year — more than 4,000 per day. A doctor who wants to stay current on treatment advances for a single condition would need to read dozens of papers a week just to keep up, on top of seeing patients. In practice, this is impossible.

**TopicFlow** addresses this by automating the literature scan.
Given a plain-language medical question — *"What are the latest treatments for malaria?"* — the tool:

1. Identifies the dominant research themes in recent publications
2. Compares them to a six-month baseline to surface what is *emerging* and what has *faded*
3. Drafts ready-to-publish science communication content tailored to a chosen audience (clinicians, patients, journalists, etc.) and platform (LinkedIn, Instagram, Twitter/X)

The goal is to compress hours of reading and writing into a single query.

---

### How it works

The pipeline is split into two independent layers so that changing the target audience or platform does not re-run the expensive topic model.

#### Core pipeline (runs once per query, fully cached)

```
User query
    │
    ▼
analyze_query ── not medical ──▶ [error]
    │ medical
    ▼
create_plan
(search focus, inclusion/exclusion criteria, evidence types to prioritise)
    │
    ▼
set_periods
(current window: last 3 months  |  baseline window: prior 6 months)
    │
    ▼
fetch_current  ◀────────────────────────────┐
(OpenAlex API → up to 500 papers            │
 SBERT semantic rerank → top 200)           │
    │                                       │
    ▼                                   refine_search
inspect_search_quality ── retry ──────────▶ (LLM rewrites keywords)
    │ ok             insufficient
    │                    │
    ▼                    ▼
topic_model_current   [warn user]
(BERTopic + KMeans clustering
 LLM merges near-duplicate topics,
 filters irrelevant ones, summarises each)
    │
    ▼
fetch_previous  →  topic_model_previous
(same pipeline on 6-month baseline)
    │
    ▼
compare_periods
(LLM: emerging signals, disappeared signals, narrative)
    │
    ▼
  Dashboard
```

#### Communication layer (runs per audience + platform, independently cached)

```
Core result  →  generate_followups  →  generate_social_content  →  Draft
                (3 next questions)      (LinkedIn post /
                                         Instagram carousel /
                                         Twitter/X thread)
```

The full LangGraph diagram is saved in [`assets/langgraph_core_pipeline.mmd`](assets/langgraph_core_pipeline.mmd) — paste it into [mermaid.live](https://mermaid.live) to render it.

---

### Screenshot

> _Add a screenshot here once the app is running_

![TopicFlow screenshot placeholder](assets/screenshot.png)

---

### Tech stack

| Layer | Technology |
|---|---|
| Orchestration | [LangGraph](https://github.com/langchain-ai/langgraph) `StateGraph` with conditional edges |
| Literature source | [OpenAlex](https://openalex.org) (open, free) |
| Semantic reranking | [Sentence-Transformers](https://www.sbert.net) (`all-MiniLM-L6-v2`) |
| Topic discovery | [BERTopic](https://maartengr.github.io/BERTopic/) with KMeans (deterministic, no UMAP) |
| LLM calls | Any OpenAI-compatible endpoint (GLM, GPT-4, etc.) |
| UI | [Streamlit](https://streamlit.io) |
| Deployment | Docker → Hugging Face Spaces (auto-synced via GitHub Actions) |

---

### Running locally

```bash
cp .env.sample .env   # fill in your keys
uv run streamlit run src/streamlit_app.py
```

Or with Docker:

```bash
docker compose up --build
```

### Environment variables

| Variable | Description | Default |
|---|---|---|
| `OPEN_ALEX_KEY` | OpenAlex API key | required |
| `URL_BASE` | OpenAI-compatible LLM base URL | required |
| `LLM_API_KEY` | LLM API key | required |
| `LLM_MODEL` | Model name | `glm-5` |
| `TOPIC_MODEL_K` | Number of BERTopic clusters | auto |
| `TOPIC_MODEL_MAX_DOC_CHARS` | Max chars per abstract | 1200 |
| `TOPIC_MODEL_EMBEDDING_MODEL` | SBERT model name | `all-MiniLM-L6-v2` |

---

### TODO — future improvements

- **Interactive follow-up chat** — let users drill into a topic with follow-up questions, keeping the LangGraph trace visible as context.
- **LangGraph tracing / observability** — wire up LangSmith or another tracing backend so every node execution is inspectable in production.
- **Richer UI** — topic size-over-time charts, paper-count timelines, keyword clouds.
- **Export** — download results as PDF report or structured JSON for downstream workflows.
- **More sources** — expand beyond OpenAlex to PubMed, bioRxiv, clinical trial registries.
- **More platforms** — Substack newsletter draft, slide deck outline, lay-audience one-pager.
- **Domain expansion** — extend the medical-query guard to support other scientific fields.
