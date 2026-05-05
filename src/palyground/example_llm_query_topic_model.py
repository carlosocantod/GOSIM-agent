from sentence_transformers import SentenceTransformer

from src.utils.llm import MESSAGE_NOT_MEDICAL
from src.utils.llm import extract_medical_keywords
from src.utils.open_alex import get_200_openalex_last_months
from src.utils.topic_model_llm import (
    EMBEDDING_MODEL_NAME,
    K,
    build_topics_payload,
    propose_merges,
    summarize_topics_with_pydantic,
)
from bertopic import BERTopic
from bertopic.dimensionality import BaseDimensionalityReduction
from sklearn.cluster import KMeans


def main():
    result = extract_medical_keywords("What are the latest treatments for pediatric malaria?")
    if not result.is_medical:
        print(MESSAGE_NOT_MEDICAL)
        return

    docs_raw = get_200_openalex_last_months(" OR ".join(result.keywords), limit=200)
    docs = [d.abstract for d in docs_raw if d.abstract]
    print(f"Fetched {len(docs)} abstracts")

    sbert_model = SentenceTransformer(EMBEDDING_MODEL_NAME)
    embeddings = sbert_model.encode(docs, show_progress_bar=True, normalize_embeddings=True)

    topic_model = BERTopic(
        embedding_model=sbert_model,
        umap_model=BaseDimensionalityReduction(),
        hdbscan_model=KMeans(n_clusters=K, random_state=0, n_init="auto"),
        calculate_probabilities=False,
        verbose=True,
    )
    topic_model.fit_transform(docs, embeddings=embeddings)

    payload = build_topics_payload(topic_model)
    merges = propose_merges(payload)
    if merges.merges:
        topic_model.merge_topics(docs, merges.merges)

    merged_payload = build_topics_payload(topic_model)
    summaries = summarize_topics_with_pydantic(merged_payload)
    print(summaries.model_dump_json(indent=2))


if __name__ == "__main__":
    main()
