import os
from typing import List

import streamlit as st
from dotenv import load_dotenv
from openai import OpenAI

from src.utils.llm import MESSAGE_NOT_MEDICAL
from src.utils.llm import MedicalQueryAnalysis
from src.utils.llm import extract_medical_keywords
from src.utils.open_alex import OpenAlexWork
from src.utils.open_alex import get_openalex_papers_last_months
from sentence_transformers import SentenceTransformer

from src.utils.topic_model_llm import EMBEDDING_MODEL_NAME
from src.utils.topic_model_llm import TopicSummaries
from src.utils.topic_model_llm import run_topic_model
from src.utils.topic_model_llm import semantic_rerank

load_dotenv()

st.set_page_config(
    page_title="Medical Science Communication Helper Agent - GOSIM",
    page_icon="🩺",
    layout="wide",
)
if "results" not in st.session_state:
    st.session_state["results"] = None

# ------------------------------------------------------------------
# Cached lightweight / serializable steps
# ------------------------------------------------------------------

@st.cache_resource(show_spinner="Loading embedding model...")
def _load_sbert_model() -> SentenceTransformer:
    return SentenceTransformer(EMBEDDING_MODEL_NAME)


@st.cache_data(show_spinner=False)
def _analyze_query(query: str) -> MedicalQueryAnalysis:
    return extract_medical_keywords(query)


@st.cache_data(show_spinner=False)
def _fetch_papers(keywords: tuple[str, ...], query: str) -> tuple[List[OpenAlexWork], int, int]:
    search_query = " OR ".join(keywords)
    docs = get_openalex_papers_last_months(search_query, limit=500)
    fetched = len(docs)
    reranked = semantic_rerank(query, docs, embedding_model=_load_sbert_model(), top_n=200)
    return reranked, fetched, len(reranked)


def _run_pipeline(
    abstracts: tuple[str, ...],
    url_base: str,
    api_key: str,
    model: str,
    query: str,
) -> tuple[TopicSummaries, List[int]]:
    client = OpenAI(base_url=url_base, api_key=api_key)
    result = run_topic_model(
        list(abstracts),
        client=client,
        model=model,
        embedding_model=_load_sbert_model(),
        query=query,
    )
    return result.summaries, result.topic_assignments


# ------------------------------------------------------------------
# UI
# ------------------------------------------------------------------

def render_header() -> None:
    st.title("🩺 Medical Science Communication Helper Agent")
    st.caption("Powered by OpenAlex + BERTopic + LLM")


def render_query_input() -> tuple[str, bool]:
    _, center, _ = st.columns([1, 2, 1])

    with center:
        query = st.text_area(
            "Enter your medical research query",
            max_chars=500,
            placeholder="e.g. What are the latest treatments for pediatric malaria?",
            value="What are the latest treatments for malaria?",
            height=100,
        )

        submitted = st.button(
            "Explore literature",
            type="primary",
            use_container_width=True,
        )

    return query, submitted


def render_centered_message(kind: str, message: str) -> None:
    _, center, _ = st.columns([1, 2, 1])

    with center:
        if kind == "success":
            st.success(message)
        elif kind == "info":
            st.info(message)
        elif kind == "warning":
            st.warning(message)
        elif kind == "error":
            st.error(message)


def render_topic_dashboard(
    summaries: TopicSummaries,
    assignments: List[int],
    docs: List[OpenAlexWork],
) -> None:
    st.subheader("Topics identified in recent literature")

    cols = st.columns(2)

    for idx, summary in enumerate(summaries.summaries):
        topic_docs = [
            docs[i]
            for i, topic_id in enumerate(assignments)
            if topic_id == summary.topic_id
        ]

        with cols[idx % 2]:
            show_key = f"docs_{summary.topic_id}"
            showing_papers = st.session_state.get(show_key, False)

            with st.expander(f"**{summary.label}** — {len(topic_docs)} papers", expanded=showing_papers):
                st.markdown(summary.summary)
                st.toggle("Show papers", key=show_key)

                if st.session_state.get(show_key, False):
                    st.divider()
                    for doc in topic_docs:
                        doi_link = (
                            f"[{doc.doi}](https://doi.org/{doc.doi})"
                            if doc.doi
                            else "N/A"
                        )
                        st.markdown(f"- **{doc.title or 'Untitled'}** | DOI: {doi_link}")


# ------------------------------------------------------------------
# Main
# ------------------------------------------------------------------

def run_query(query: str) -> tuple[TopicSummaries, List[int], List[OpenAlexWork]] | None:
    with st.spinner("Analyzing query..."):
        analysis = _analyze_query(query)

    if not analysis.is_medical:
        render_centered_message("error", MESSAGE_NOT_MEDICAL)
        return None

    render_centered_message(
        "success",
        f"Keywords: {', '.join(analysis.keywords)}",
    )

    with st.spinner("Fetching papers from OpenAlex..."):
        all_docs, fetched_count, reranked_count = _fetch_papers(tuple(analysis.keywords), query=query)
        docs_with_abstract = [doc for doc in all_docs if doc.abstract]

    if not docs_with_abstract:
        render_centered_message("warning", "No abstracts found. Try broadening your search.")
        return None

    render_centered_message(
        "info",
        f"Kept {reranked_count} of {fetched_count} papers after semantic reranking — relevant topics filtered next.",
    )

    with st.spinner("Running topic model — this may take a minute..."):
        abstracts = tuple(doc.abstract for doc in docs_with_abstract)

        summaries, assignments = _run_pipeline(
            abstracts,
            url_base=os.getenv("URL_BASE", ""),
            api_key=os.getenv("LLM_API_KEY", ""),
            model=os.getenv("LLM_MODEL", "glm-5"),
            query=query,
        )

    return summaries, assignments, docs_with_abstract


def main() -> None:
    render_header()

    query, submitted = render_query_input()

    if submitted and query.strip():
        result = run_query(query)

        if result is not None:
            st.session_state["results"] = result
            st.session_state["last_query"] = query

    if st.session_state["results"] is not None:
        summaries, assignments, docs = st.session_state["results"]
        render_topic_dashboard(summaries, assignments, docs)


if __name__ == "__main__":
    main()