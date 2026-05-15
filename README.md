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


## TopicFlow
Team members: Carlos Ocanto

> **LangGraph-powered research signal tracker and science communication assistant for medical use cases**


🚀 **Live demo:** [huggingface.co/spaces/carlosocantod/GOSIM-agent](https://huggingface.co/spaces/carlosocantod/GOSIM-agent)

👨🏻‍🏫 **Presentation:** [docs/GOSIM_Agent_TopicFlow_Presentation.pdf](docs/GOSIM_Agent_TopicFlow_Presentation.pdf)

🎥 **Talk video:** [YouTube — GOSIM Agent overview](https://www.youtube.com/watch?v=-PEKKNwMFxw)

🎬 **Demo video:** _coming soon_ <!-- replace with YouTube URL when available -->

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

### Pre-hackathon setup
Before the hackathon, the following infrastructure was put in place:
- Streamlit app scaffold.
- Automatic deployment to Hugging Face Spaces via GitHub Actions (push to `main` triggers sync).
- OpenAlex API integration for querying recent biomedical literature by keyword.

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

### Performance & cost

| Metric | Value |
|---|---|
| Typical cost per run | ~**$0.014** |
| Wall-clock time | ~**3 min 30 s** |
| Papers fetched (current period) | up to 500, reranked to top 200 |
| LLM tier | Mini/small models to keep cost low |

The ~3.5 minute runtime feels close to real-time because the UI provides progressive feedback at each pipeline step: users see partial results (current-period topics) before the previous-period pipeline has finished. Langfuse observability adds a small amount of latency; a production version would use Langfuse's async SDK path or a self-hosted instance with lower round-trip overhead.

To keep token spend predictable, not all text from the chat context is forwarded to the LLM on every call.

---

### Observability

Each run is fully traced via **[Langfuse](https://langfuse.com)**. Every LangGraph node, LLM call, and tool invocation appears as a structured span, making it straightforward to debug latency regressions or prompt failures in production.

#### LLM-as-a-judge evaluations

The `inspect_search_quality` node uses an LLM judge to score the retrieved paper set before passing it to the topic model. The evaluations follow the traces end-to-end and are already production-ready as a signal for quality monitoring. **Custom prompts are needed to reach full precision** — the current judge prompts are functional but not yet tuned for edge cases (very narrow queries, rare diseases, multi-lingual abstracts).

To enable tracing, set `LANGFUSE_PUBLIC_KEY`, `LANGFUSE_SECRET_KEY`, and optionally `LANGFUSE_HOST` in `.env` (see [Environment variables](#environment-variables)).

---

### Tech stack

| Layer | Technology |
|---|---|
| Orchestration | [LangGraph](https://github.com/langchain-ai/langgraph) `StateGraph` with conditional edges |
| Literature source | [OpenAlex](https://openalex.org) (open, free) |
| Semantic reranking | [Sentence-Transformers](https://www.sbert.net) (`all-MiniLM-L6-v2`) |
| Topic discovery | [BERTopic](https://maartengr.github.io/BERTopic/) with KMeans (deterministic, no UMAP) |
| LLM calls | Any OpenAI-compatible endpoint (GLM, GPT-4o-mini, DeepSeek, etc.) |
| Observability | [Langfuse](https://langfuse.com) — LangGraph traces + LLM-as-a-judge evals |
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

---

### Environment variables

| Variable | Description | Default |
|---|---|---|
| `OPEN_ALEX_KEY` | OpenAlex API key (get one at [openalex.org/settings/api](https://openalex.org/settings/api)) | required |
| `URL_BASE` | OpenAI-compatible LLM base URL | required |
| `LLM_API_KEY` | LLM API key | required |
| `LLM_MODEL` | Model name | required |
| `TOPIC_MODEL_K` | Number of BERTopic clusters | auto |
| `TOPIC_MODEL_MAX_DOC_CHARS` | Max chars per abstract fed to embeddings | `300` |
| `TOPIC_MODEL_EMBEDDING_MODEL` | SBERT model name | `sentence-transformers/paraphrase-MiniLM-L3-v2` |
| `LANGFUSE_PUBLIC_KEY` | Langfuse public key (optional, for tracing) | — |
| `LANGFUSE_SECRET_KEY` | Langfuse secret key (optional, for tracing) | — |
| `LANGFUSE_HOST` | Langfuse host (optional, defaults to cloud) | `https://cloud.langfuse.com` |

---

### Current state & known limitations

This is an **exploitable MVP**. The core pipeline works end-to-end and produces actionable research signals, but several rough edges remain:

| Area | Status |
|---|---|
| **Output formatting** | Functional but plain; richer visualisations (charts, keyword clouds) are planned |
| **Result diversity** | Aggregation-level diversity assessment across runs is still in progress |
| **Literature source** | OpenAlex is the slowest step in the pipeline; the OpenAlex semantic search beta has not been evaluated; alternative sources (PubMed, bioRxiv) not yet integrated |
| **Query caching** | Similar queries re-run the full pipeline; result caching is a top priority for the next iteration |
| **LLM-as-a-judge tuning** | Judge evaluations follow every trace but custom prompts are needed to hit higher precision |

---

### TODO — next priorities

- [ ] **Cache similar queries** — embed queries and return cached results when cosine similarity exceeds a threshold, avoiding redundant API calls and LLM runs.
- [ ] **Tune LLM-as-a-judge prompts** — improve precision of the search-quality evaluator, especially for narrow or multi-lingual queries.
- [ ] **Richer output** — topic size-over-time charts, paper-count timelines, keyword clouds, PDF/JSON export.
- [ ] **Aggregation diversity** — surface diversity metrics across topics and time windows to assess coverage breadth.
- [ ] **Evaluate alternative literature sources** — benchmark OpenAlex semantic search (beta), PubMed, and bioRxiv for recall and latency.
- [ ] **Langfuse async path** — switch to async Langfuse SDK to reduce added latency in production.
- [ ] **Interactive follow-up chat** — let users drill into a topic with follow-up questions, keeping the LangGraph trace visible as context.
- [ ] **More sources** — PubMed, bioRxiv, clinical trial registries.
