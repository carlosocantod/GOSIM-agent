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

## TopicFlow — LangGraph-powered medical research signal tracker

Doctors and researchers are overwhelmed. The volume of biomedical literature doubles roughly every nine years, and no clinician has time to track emerging treatment signals across thousands of new papers each month. **TopicFlow** automates that scan: given a plain-language medical question, it fetches recent literature, discovers the dominant research themes, compares them to a six-month baseline, and drafts science-communication content ready to publish — all in one run.

### How it works

The pipeline has two layers: a **core research layer** that runs once per query (and is cached), and a **communication layer** that regenerates whenever the audience or platform changes.

```mermaid
flowchart TD
    A([User query]) --> B[analyze_query\nLLM: is this medical?\nextract keywords]
    B -->|not medical| Z([END — show error])
    B -->|medical| C[create_plan\nLLM: search focus,\ninclusion/exclusion criteria]
    C --> D[set_periods\ncompute current 3-month\nand baseline 6-month windows]
    D --> E[fetch_current\nOpenAlex API]
    E --> F[inspect_search_quality\nLLM: enough signal?\nrevise keywords if not]
    F -->|retry with new keywords| E
    F -->|insufficient after retry| Z2([END — warn user])
    F -->|ok| G[topic_model_current\nBERTopic + KMeans\nLLM merge & summarise]
    G --> H[fetch_previous\nOpenAlex API\nbaseline window]
    H --> I[topic_model_previous\nsame pipeline]
    I --> J[compare_periods\nLLM: emerging vs disappeared topics\nnarrative summary]
    J --> Z3([END — show dashboard])

    subgraph Communication layer — runs per audience + platform
        K([core result]) --> L[generate_followups\nLLM: 3 next questions]
        L --> M[generate_social_content\nLLM: LinkedIn post / Instagram\ncarousel / Twitter thread]
        M --> Z4([rendered draft])
    end
```

### Screenshot

> _Screenshot coming soon — add yours here_

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

### TODO — future improvements

- **Interactive follow-up chat** — let users drill into a topic with follow-up questions, keeping the LangGraph trace visible as context.
- **LangGraph tracing / observability** — wire up LangSmith or another tracing backend so every node execution is inspectable.
- **Richer UI** — charts for topic size over time, paper-count timelines, keyword clouds.
- **More platforms** — Substack newsletter draft, slide deck outline, lay-audience summary.
- **Domain expansion** — extend beyond medical queries to other scientific fields.
- **Export** — download results as PDF or structured JSON for downstream workflows.
