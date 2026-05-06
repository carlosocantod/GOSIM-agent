import os
from datetime import date
from typing import List

import streamlit as st
from dotenv import load_dotenv
from openai import OpenAI
from sentence_transformers import SentenceTransformer

from src.utils.llm import MESSAGE_NOT_MEDICAL
from src.utils.llm import MedicalQueryAnalysis
from src.utils.llm import extract_medical_keywords
from src.utils.open_alex import OpenAlexWork
from src.utils.open_alex import get_openalex_papers_for_period
from src.utils.open_alex import last_n_months_date_range
from src.utils.open_alex import previous_period_date_range
from src.utils.topic_model_llm import EMBEDDING_MODEL_NAME
from src.utils.topic_model_llm import PeriodComparison
from src.utils.topic_model_llm import TopicSummaries
from src.utils.topic_model_llm import compare_topic_periods
from src.utils.topic_model_llm import run_topic_model
from src.utils.topic_model_llm import semantic_rerank

load_dotenv()

N_MONTHS_CURRENT = 3
N_MONTHS_BASELINE = 6

st.set_page_config(
    page_title="Medical Science Communication Helper Agent - GOSIM",
    page_icon="🩺",
    layout="wide",
)
if "results" not in st.session_state:
    st.session_state["results"] = None

# ------------------------------------------------------------------
# Cached resources / steps
# ------------------------------------------------------------------

@st.cache_resource(show_spinner="Loading embedding model...")
def _load_sbert_model() -> SentenceTransformer:
    return SentenceTransformer(EMBEDDING_MODEL_NAME)


@st.cache_data(show_spinner=False)
def _analyze_query(query: str) -> MedicalQueryAnalysis:
    return extract_medical_keywords(query)


@st.cache_data(show_spinner=False)
def _fetch_papers(
    keywords: tuple[str, ...],
    query: str,
    from_date: str,
    to_date: str,
) -> tuple[List[OpenAlexWork], int, int]:
    search_query = " OR ".join(keywords)
    docs = get_openalex_papers_for_period(search_query, from_date=from_date, to_date=to_date, limit=500)
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


@st.cache_data(show_spinner=False)
def _compare_periods(
    current_summaries_json: str,
    previous_summaries_json: str,
    current_period_label: str,
    previous_period_label: str,
    url_base: str,
    api_key: str,
    model: str,
) -> str:
    client = OpenAI(base_url=url_base, api_key=api_key)
    current = TopicSummaries.model_validate_json(current_summaries_json)
    previous = TopicSummaries.model_validate_json(previous_summaries_json)
    result = compare_topic_periods(current, previous, current_period_label, previous_period_label, client, model)
    return result.model_dump_json()


# ------------------------------------------------------------------
# UI helpers
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
            placeholder="e.g. Which are the latest treatments for pediatric malaria?",
            value="Which are the latest treatments for malaria?",
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


def _format_period_label(from_date: str, to_date: str) -> str:
    from_dt = date.fromisoformat(from_date)
    to_dt = date.fromisoformat(to_date)
    return f"{from_dt.strftime('%b %Y')} – {to_dt.strftime('%b %Y')}"


def render_topic_dashboard(
    summaries: TopicSummaries,
    assignments: List[int],
    docs: List[OpenAlexWork],
    key_prefix: str = "curr",
    emerging_labels: set[str] | None = None,
) -> None:
    cols = st.columns(2)

    for idx, summary in enumerate(summaries.summaries):
        topic_docs = [
            docs[i]
            for i, topic_id in enumerate(assignments)
            if topic_id == summary.topic_id
        ]

        is_emerging = emerging_labels and summary.label in emerging_labels
        label_display = f"🆕 {summary.label}" if is_emerging else summary.label

        with cols[idx % 2]:
            show_key = f"{key_prefix}_docs_{summary.topic_id}"
            expander_key = f"{key_prefix}_exp_{summary.topic_id}"

            with st.expander(f"**{label_display}** — {len(topic_docs)} papers", key=expander_key):
                st.markdown(summary.summary)
                if st.toggle("Show papers", key=show_key):
                    st.divider()
                    for doc in topic_docs:
                        doi_link = (
                            f"[{doc.doi}](https://doi.org/{doc.doi})"
                            if doc.doi
                            else "N/A"
                        )
                        st.markdown(f"- **{doc.title or 'Untitled'}** | DOI: {doi_link}")


def render_period_comparison(
    current_summaries: TopicSummaries,
    current_assignments: List[int],
    current_docs: List[OpenAlexWork],
    previous_summaries: TopicSummaries,
    previous_assignments: List[int],
    previous_docs: List[OpenAlexWork],
    comparison: PeriodComparison,
    current_label: str,
    previous_label: str,
) -> None:
    emerging = set(comparison.emerging_topic_labels)
    disappeared = set(comparison.disappeared_topic_labels)

    # Current period
    st.subheader(f"Current period: {current_label}")
    relevant_topic_ids = {s.topic_id for s in current_summaries.summaries}
    docs_in_relevant = sum(1 for a in current_assignments if a in relevant_topic_ids)
    st.caption(f"{docs_in_relevant} papers across {len(current_summaries.summaries)} topics")
    render_topic_dashboard(current_summaries, current_assignments, current_docs, key_prefix="curr")

    # Comparison narrative
    st.divider()
    st.subheader(f"How does this compare to {previous_label}?")
    st.info(comparison.narrative)
    if emerging:
        st.success(f"**🆕 Emerging this period:** {', '.join(emerging)}")
    if disappeared:
        st.caption(f"**No longer prominent:** {', '.join(disappeared)}")

    # Previous period
    with st.expander(f"Previous period topics: {previous_label}", expanded=False):
        prev_relevant_ids = {s.topic_id for s in previous_summaries.summaries}
        prev_docs_count = sum(1 for a in previous_assignments if a in prev_relevant_ids)
        st.caption(f"{prev_docs_count} papers across {len(previous_summaries.summaries)} topics")
        render_topic_dashboard(previous_summaries, previous_assignments, previous_docs, key_prefix="prev")


# ------------------------------------------------------------------
# Main query runner
# ------------------------------------------------------------------

def run_query(query: str) -> None:
    with st.spinner("Analyzing query..."):
        analysis = _analyze_query(query)

    if not analysis.is_medical:
        render_centered_message("error", MESSAGE_NOT_MEDICAL)
        return

    render_centered_message("success", f"Keywords: {', '.join(analysis.keywords)}")

    keywords_tuple = tuple(analysis.keywords)
    url_base = os.getenv("URL_BASE", "")
    api_key = os.getenv("LLM_API_KEY", "")
    model = os.getenv("LLM_MODEL", "glm-5")

    curr_from, curr_to = last_n_months_date_range(N_MONTHS_CURRENT)
    prev_from, prev_to = previous_period_date_range(N_MONTHS_BASELINE, n_months_current=N_MONTHS_CURRENT)
    current_label = _format_period_label(curr_from, curr_to)
    previous_label = _format_period_label(prev_from, prev_to)

    with st.spinner(f"Fetching latest papers ({current_label})..."):
        curr_docs, curr_fetched, curr_reranked = _fetch_papers(keywords_tuple, query, curr_from, curr_to)
        curr_with_abstract = [d for d in curr_docs if d.abstract]

    if not curr_with_abstract:
        render_centered_message("warning", "No abstracts found for the current period. Try broadening your search.")
        return

    render_centered_message("info", f"Kept {curr_reranked} of {curr_fetched} papers after semantic reranking.")

    with st.spinner(f"Running topic model on latest papers ({current_label})..."):
        curr_summaries, curr_assignments = _run_pipeline(
            tuple(d.abstract for d in curr_with_abstract),
            url_base=url_base, api_key=api_key, model=model, query=query,
        )

    with st.spinner(f"Fetching papers for previous period ({previous_label})..."):
        prev_docs, _, _ = _fetch_papers(keywords_tuple, query, prev_from, prev_to)
        prev_with_abstract = [d for d in prev_docs if d.abstract]

    if prev_with_abstract:
        with st.spinner(f"Running topic model on previous period ({previous_label})..."):
            prev_summaries, prev_assignments = _run_pipeline(
                tuple(d.abstract for d in prev_with_abstract),
                url_base=url_base, api_key=api_key, model=model, query=query,
            )
    else:
        prev_summaries, prev_assignments = TopicSummaries(summaries=[]), []

    with st.spinner("Comparing periods..."):
        comparison_json = _compare_periods(
            curr_summaries.model_dump_json(),
            prev_summaries.model_dump_json(),
            current_label,
            previous_label,
            url_base=url_base, api_key=api_key, model=model,
        )
        comparison = PeriodComparison.model_validate_json(comparison_json)

    st.session_state["results"] = (
        curr_summaries, curr_assignments, curr_with_abstract,
        prev_summaries, prev_assignments, prev_with_abstract,
        comparison, current_label, previous_label,
    )


def main() -> None:
    render_header()

    query, submitted = render_query_input()

    if submitted and query.strip():
        run_query(query)

    if st.session_state["results"] is not None:
        (
            curr_summaries, curr_assignments, curr_docs,
            prev_summaries, prev_assignments, prev_docs,
            comparison, current_label, previous_label,
        ) = st.session_state["results"]

        render_period_comparison(
            curr_summaries, curr_assignments, curr_docs,
            prev_summaries, prev_assignments, prev_docs,
            comparison, current_label, previous_label,
        )


if __name__ == "__main__":
    main()
