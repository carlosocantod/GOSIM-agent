import os
from typing import List

import streamlit as st
from dotenv import load_dotenv
from openai import OpenAI
from sentence_transformers import SentenceTransformer

from src.utils.open_alex import OpenAlexWork
from src.utils.research_agent import CommunicationResult
from src.utils.research_agent import ResearchAgentResult
from src.utils.research_agent import SocialMediaContent
from src.utils.research_agent import run_communication_layer
from src.utils.research_agent import run_core_pipeline
from src.utils.topic_model_llm import EMBEDDING_MODEL_NAME
from src.utils.topic_model_llm import PeriodComparison
from src.utils.topic_model_llm import TopicSummaries

load_dotenv()

st.set_page_config(
    page_title="TopicFlow",
    page_icon="🔬",
    layout="wide",
)

if "core_result" not in st.session_state:
    st.session_state["core_result"] = None


# ------------------------------------------------------------------
# Cached resources
# ------------------------------------------------------------------

@st.cache_resource(show_spinner="Loading embedding model...")
def _load_sbert_model() -> SentenceTransformer:
    return SentenceTransformer(EMBEDDING_MODEL_NAME)


# ------------------------------------------------------------------
# UI helpers
# ------------------------------------------------------------------

def render_header() -> None:
    st.title("🔬 TopicFlow")
    st.caption("LangGraph-powered deep research and science communication agent for medical use cases — built on an LLM-enhanced topic model.")


def render_query_input() -> tuple[str, str, bool]:
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
            "Explore literature signals",
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


def render_agent_sidebar(result: ResearchAgentResult | None) -> None:
    with st.sidebar:
        st.header("Signal workflow")
        st.caption("What the LLM-enhanced topic model checked during this run.")

        if result is None:
            st.write("Run a query to see the workflow trace.")
            return

        for step in result.agent_steps:
            st.write(step)


def render_plan_and_quality(result: ResearchAgentResult) -> None:
    if result.analysis and result.analysis.keywords:
        render_centered_message("success", f"Keywords used: {', '.join(result.keywords_used or result.analysis.keywords)}")

    if result.plan:
        with st.expander("Literature signal plan", expanded=True):
            st.markdown(f"**Interpreted question:** {result.plan.interpreted_question}")

            cols = st.columns(2)
            with cols[0]:
                st.markdown("**Search focus**")
                for item in result.plan.search_focus:
                    st.markdown(f"- {item}")

                st.markdown("**Prioritized evidence**")
                for item in result.plan.evidence_types_to_prioritize:
                    st.markdown(f"- {item}")

            with cols[1]:
                st.markdown("**Include**")
                for item in result.plan.inclusion_criteria:
                    st.markdown(f"- {item}")

                st.markdown("**Exclude**")
                for item in result.plan.exclusion_criteria:
                    st.markdown(f"- {item}")

    if result.search_quality:
        with st.expander("Search quality check", expanded=False):
            st.markdown(f"**Decision:** `{result.search_quality.status}`")
            st.write(result.search_quality.reason)
            if result.search_quality.revised_keywords:
                st.markdown(f"**Revised keywords:** {', '.join(result.search_quality.revised_keywords)}")


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
            if i < len(docs) and topic_id == summary.topic_id
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

    st.subheader(f"Current literature signals: {current_label}")
    relevant_topic_ids = {s.topic_id for s in current_summaries.summaries}
    docs_in_relevant = sum(1 for a in current_assignments if a in relevant_topic_ids)
    st.caption(f"{docs_in_relevant} papers across {len(current_summaries.summaries)} topics")
    render_topic_dashboard(
        current_summaries,
        current_assignments,
        current_docs,
        key_prefix="curr",
        emerging_labels=emerging,
    )

    st.divider()
    st.subheader(f"What changed vs. {previous_label}?")
    st.info(comparison.narrative)
    if emerging:
        st.success(f"**🆕 Emerging signals:** {', '.join(emerging)}")
    if disappeared:
        st.caption(f"**Less prominent than baseline:** {', '.join(disappeared)}")

    with st.expander(f"Baseline topics: {previous_label}", expanded=False):
        prev_relevant_ids = {s.topic_id for s in previous_summaries.summaries}
        prev_docs_count = sum(1 for a in previous_assignments if a in prev_relevant_ids)
        st.caption(f"{prev_docs_count} papers across {len(previous_summaries.summaries)} topics")
        render_topic_dashboard(previous_summaries, previous_assignments, previous_docs, key_prefix="prev")


def render_social_content(sc: SocialMediaContent) -> None:
    icon = _PLATFORM_ICONS.get(sc.platform, "")
    st.subheader(f"{icon} {sc.platform} draft")
    st.markdown(f"**{sc.headline}**")

    if sc.platform == "LinkedIn":
        st.markdown(sc.post)

    elif sc.platform == "Twitter/X":
        for i, tweet in enumerate(sc.tweets, 1):
            with st.container(border=True):
                st.markdown(f"**Tweet {i}** · {len(tweet)} chars")
                st.write(tweet)

    elif sc.platform == "Instagram":
        st.caption("Carousel slides")
        for i, slide in enumerate(sc.slides, 1):
            with st.expander(f"Slide {i} — {slide.title}", expanded=True):
                st.write(slide.body)
        if sc.caption:
            st.markdown(f"**Caption:** {sc.caption}")

    if sc.hashtags:
        st.caption(" ".join(f"#{h}" for h in sc.hashtags))
    if sc.caution:
        st.warning(f"**Caution:** {sc.caution}")


def render_science_communication(comm: CommunicationResult) -> None:
    if comm.social_content:
        st.divider()
        render_social_content(comm.social_content)

    if comm.followups:
        with st.expander("Suggested next questions", expanded=False):
            for question in comm.followups:
                st.markdown(f"- {question}")


_AUDIENCES = ["General public", "Patients", "Clinicians", "Researchers", "Policy makers", "Journalists"]
_PLATFORMS = ["LinkedIn", "Instagram", "Twitter/X"]
_PLATFORM_ICONS = {"LinkedIn": "💼", "Instagram": "📸", "Twitter/X": "🐦"}


def _run_core_query(query: str) -> None:
    url_base = os.getenv("URL_BASE", "")
    api_key = os.getenv("LLM_API_KEY", "")
    model = os.getenv("LLM_MODEL", "glm-5")

    cache_key = (query, url_base, api_key, model)
    cache = st.session_state.setdefault("_core_cache", {})

    if cache_key in cache:
        st.session_state["core_result"] = cache[cache_key]
        return

    client = OpenAI(base_url=url_base, api_key=api_key)

    with st.status("Analyzing query...", expanded=False) as status:
        result = run_core_pipeline(
            query=query,
            client=client,
            model=model,
            embedding_model=_load_sbert_model(),
            on_step=lambda msg: status.update(label=msg),
        )

    cache[cache_key] = result
    st.session_state["core_result"] = result


def _get_communication(core: ResearchAgentResult, audience: str, platform: str) -> CommunicationResult:
    url_base = os.getenv("URL_BASE", "")
    api_key = os.getenv("LLM_API_KEY", "")
    model = os.getenv("LLM_MODEL", "glm-5")

    cache_key = (core.query, audience, platform, url_base, api_key, model)
    cache = st.session_state.setdefault("_comm_cache", {})

    if cache_key in cache:
        return cache[cache_key]

    client = OpenAI(base_url=url_base, api_key=api_key)

    with st.status("Generating communication draft...", expanded=False) as status:
        comm = run_communication_layer(
            core, audience, client, model,
            platform=platform,
            on_step=lambda msg: status.update(label=msg),
        )

    cache[cache_key] = comm
    return comm


def main() -> None:
    render_header()

    query, submitted = render_query_input()

    if submitted and query.strip():
        st.session_state["core_result"] = None
        _run_core_query(query)

    core = st.session_state.get("core_result")
    render_agent_sidebar(core)

    if core is None:
        return

    if not core.is_medical:
        render_centered_message("error", core.message or "Please enter a medical or biomedical query.")
        return

    render_plan_and_quality(core)

    if core.current_summaries and core.previous_summaries and core.comparison:
        render_period_comparison(
            core.current_summaries,
            core.current_assignments,
            core.current_docs,
            core.previous_summaries,
            core.previous_assignments,
            core.previous_docs,
            core.comparison,
            core.current_label,
            core.previous_label,
        )

        st.divider()
        _, center, _ = st.columns([1, 2, 1])
        with center:
            col1, col2 = st.columns(2)
            with col1:
                audience = st.selectbox("Audience", _AUDIENCES, key="audience_selector")
            with col2:
                platform = st.selectbox("Platform", _PLATFORMS, key="platform_selector")

        comm = _get_communication(core, audience, platform)
        render_science_communication(comm)
    else:
        render_centered_message(
            "warning",
            "Not enough usable abstracts were found for topic modeling. Try a broader biomedical query.",
        )


if __name__ == "__main__":
    main()
