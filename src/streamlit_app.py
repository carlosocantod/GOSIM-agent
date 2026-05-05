import os
from typing import List

from dotenv import load_dotenv
from openai import OpenAI
import streamlit as st

load_dotenv()

from src.utils.llm import MESSAGE_NOT_MEDICAL, MedicalQueryAnalysis, extract_medical_keywords
from src.utils.open_alex import OpenAlexWork, get_200_openalex_last_months
from src.utils.topic_model_llm import TopicSummaries, run_topic_model

st.set_page_config(
    page_title="Medical Science Communication Helper Agent - GOSIM",
    page_icon="🩺",
    layout="wide",
)


# ------------------------------------------------------------------
# Cached pipeline steps — same query = no recomputation
# ------------------------------------------------------------------

@st.cache_data(show_spinner=False)
def _analyze_query(query: str) -> MedicalQueryAnalysis:
    return extract_medical_keywords(query)


@st.cache_data(show_spinner=False)
def _fetch_papers(keywords: tuple[str, ...]) -> List[OpenAlexWork]:
    search_query = " OR ".join(keywords)
    return get_200_openalex_last_months(search_query, limit=200)


@st.cache_data(show_spinner=False)
def _run_pipeline(
    abstracts: tuple[str, ...],
    url_base: str,
    api_key: str,
    model: str,
) -> tuple[TopicSummaries, List[int]]:
    client = OpenAI(base_url=url_base, api_key=api_key)
    result = run_topic_model(list(abstracts), client=client, model=model)
    return result.summaries, result.topic_assignments


# ------------------------------------------------------------------
# UI components
# ------------------------------------------------------------------

def render_header():
    st.title("🩺 Medical Science Communication Helper Agent")
    st.caption("Powered by OpenAlex + BERTopic + LLM")


def render_query_input() -> tuple[str, bool]:
    query = st.text_area(
        "Enter your medical research query",
        max_chars=500,
        placeholder="e.g. What are the latest treatments for pediatric malaria?",
        height=100,
    )
    submitted = st.button("Explore literature", type="primary")
    return query, submitted


def render_topic_dashboard(summaries: TopicSummaries, assignments: List[int], docs: List[OpenAlexWork]):
    st.subheader("Topics identified in recent literature")
    for summary in summaries.summaries:
        topic_docs = [docs[i] for i, t in enumerate(assignments) if t == summary.topic_id]
        with st.expander(f"**{summary.label}** — {len(topic_docs)} papers"):
            st.markdown(summary.summary)
            st.markdown(f"*{len(topic_docs)} papers in this topic*")
            show = st.toggle("Show papers", key=f"docs_{summary.topic_id}")
            if show:
                for doc in topic_docs:
                    doi_link = f"[{doc.doi}](https://doi.org/{doc.doi})" if doc.doi else "N/A"
                    st.markdown(f"- **{doc.title or 'Untitled'}** | DOI: {doi_link}")


# ------------------------------------------------------------------
# Main
# ------------------------------------------------------------------

def main():
    render_header()
    query, submitted = render_query_input()

    if submitted and query.strip():
        with st.spinner("Analyzing query..."):
            analysis = _analyze_query(query)

        if not analysis.is_medical:
            st.error(MESSAGE_NOT_MEDICAL)
            st.session_state.pop("results", None)
            return

        st.success(f"Keywords: {', '.join(analysis.keywords)}")

        with st.spinner("Fetching papers from OpenAlex..."):
            all_docs = _fetch_papers(tuple(analysis.keywords))
            docs_with_abstract = [d for d in all_docs if d.abstract]

        if not docs_with_abstract:
            st.warning("No abstracts found. Try broadening your search.")
            st.session_state.pop("results", None)
            return

        st.info(f"Found {len(docs_with_abstract)} papers with abstracts.")

        with st.spinner("Running topic model — this may take a minute..."):
            abstracts = tuple(d.abstract for d in docs_with_abstract)
            summaries, assignments = _run_pipeline(
                abstracts,
                url_base=os.getenv("URL_BASE", ""),
                api_key=os.getenv("LLM_API_KEY", ""),
                model=os.getenv("LLM_MODEL", "glm-5"),
            )

        st.session_state["results"] = (summaries, assignments, docs_with_abstract)

    if "results" in st.session_state:
        summaries, assignments, docs = st.session_state["results"]
        render_topic_dashboard(summaries, assignments, docs)


if __name__ == "__main__":
    main()
