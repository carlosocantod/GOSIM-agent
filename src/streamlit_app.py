import os

from dotenv import load_dotenv
from openai import OpenAI
import streamlit as st

load_dotenv()

from src.utils.llm import MESSAGE_NOT_MEDICAL, extract_medical_keywords
from src.utils.open_alex import get_200_openalex_last_months
from src.utils.topic_model_llm import run_topic_model


st.set_page_config(
    page_title="Medical Science Communication Helper Agent - GOSIM",
    page_icon="🩺",
    layout="wide",
)


def _get_client() -> OpenAI:
    return OpenAI(
        base_url=os.getenv("URL_BASE"),
        api_key=os.getenv("LLM_API_KEY"),
    )


def main():
    st.title("🩺 Medical Science Communication Helper Agent")
    st.caption("Powered by OpenAlex + BERTopic + LLM")

    query = st.text_area(
        "Enter your medical research query",
        max_chars=500,
        placeholder="e.g. What are the latest treatments for pediatric malaria?",
        height=100,
    )

    if not st.button("Explore literature", type="primary") or not query.strip():
        return

    with st.spinner("Analyzing query..."):
        analysis = extract_medical_keywords(query)

    if not analysis.is_medical:
        st.error(MESSAGE_NOT_MEDICAL)
        return

    st.success(f"Keywords identified: {', '.join(analysis.keywords)}")

    with st.spinner("Fetching papers from OpenAlex..."):
        search_query = " OR ".join(analysis.keywords)
        docs_raw = get_200_openalex_last_months(search_query, limit=200)
        docs_with_abstract = [(doc, doc.abstract) for doc in docs_raw if doc.abstract]

    if not docs_with_abstract:
        st.warning("No abstracts found for this query. Try broadening your search.")
        return

    st.info(f"Found {len(docs_with_abstract)} papers with abstracts.")

    with st.spinner("Running topic model — this may take a minute..."):
        abstracts = [a for _, a in [(d, d.abstract) for d, _ in docs_with_abstract]]
        client = _get_client()
        model = os.getenv("LLM_MODEL", "glm-5")
        result = run_topic_model(abstracts, client=client, model=model)

    st.subheader("Topics identified in recent literature")

    doc_objects = [d for d, _ in docs_with_abstract]
    assignments = result.topic_assignments

    for summary in result.summaries.summaries:
        topic_docs = [doc_objects[i] for i, t in enumerate(assignments) if t == summary.topic_id]

        with st.expander(f"**{summary.label}** — {len(topic_docs)} papers"):
            st.markdown(summary.summary)

            if st.checkbox("Show papers", key=f"docs_{summary.topic_id}"):
                for doc in topic_docs:
                    doi_link = f"[{doc.doi}](https://doi.org/{doc.doi})" if doc.doi else "N/A"
                    st.markdown(f"- **{doc.title or 'Untitled'}** | DOI: {doi_link}")


if __name__ == "__main__":
    main()
