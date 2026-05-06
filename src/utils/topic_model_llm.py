import json
import os
import random
from typing import List, TYPE_CHECKING

import numpy as np
from bertopic import BERTopic
from bertopic.dimensionality import BaseDimensionalityReduction
from bertopic.representation import OpenAI as OpenAIBertTopic
from dotenv import load_dotenv
from openai import OpenAI
from pydantic import BaseModel
from pydantic import Field
from sentence_transformers import SentenceTransformer
from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import CountVectorizer

if TYPE_CHECKING:
    from src.utils.open_alex import OpenAlexWork

# -----------------------------
# Config
# -----------------------------

load_dotenv()

K = int(os.getenv("TOPIC_MODEL_K", "12"))
MAX_DOC_CHARS = int(os.getenv("TOPIC_MODEL_MAX_DOC_CHARS", "1000"))
EMBEDDING_MODEL_NAME = os.getenv("TOPIC_MODEL_EMBEDDING_MODEL", "sentence-transformers/paraphrase-MiniLM-L3-v2")


# -----------------------------
# Pydantic schemas
# -----------------------------

class TopicPayload(BaseModel):
    topic_id: int
    top_words: List[str]
    count: int
    docs: List[str]


class MergeTopics(BaseModel):
    merges: List[List[int]] = Field(
        description="Groups of topic IDs that should be merged. No singletons."
    )


class TopicSummary(BaseModel):
    topic_id: int
    label: str
    summary: str


class TopicSummaries(BaseModel):
    summaries: List[TopicSummary]


class RelevantTopics(BaseModel):
    relevant_topic_ids: List[int]


class PeriodComparison(BaseModel):
    narrative: str
    emerging_topic_labels: List[str]
    disappeared_topic_labels: List[str]


class TopicModelResult(BaseModel):
    summaries: TopicSummaries
    topic_assignments: List[int]
    topic_model: object

    model_config = {"arbitrary_types_allowed": True}


# -----------------------------
# Helpers
# -----------------------------

def truncate_docs(docs: List[str], max_chars: int = MAX_DOC_CHARS) -> List[str]:
    return [
        doc[:max_chars] + "..." if len(doc) > max_chars else doc
        for doc in docs
    ]


def build_topics_payload(topic_model: BERTopic) -> List[TopicPayload]:
    df = topic_model.get_topic_info()

    payload = []
    for _, row in df.iterrows():
        payload.append(
            TopicPayload(
                topic_id=int(row["Topic"]),
                top_words=list(row["Representation"]),
                count=int(row["Count"]),
                docs=truncate_docs(row["Representative_Docs"]),
            )
        )

    return payload


def propose_merges(topics_payload: List[TopicPayload], client: OpenAI, model: str) -> MergeTopics:
    system_prompt = """
    You are an expert topic modeler.

    You are given topics with:
    - topic_id
    - top_words
    - representative documents

    Your job:
    Group topics that are semantically similar and should be merged.

    Rules:
    - Output ONLY a JSON object.
    - Each topic_id must appear at most once.
    - Do NOT include singletons.
    - Do NOT duplicate topic_ids across groups.
    - Only merge if clearly similar.
    - Prioritize merges of smaller, semantically overlapping topics.
    - Do not merge unrelated topics just to reduce the total number.
    Schema: {"merges": [[int]]}
    """

    response = client.chat.completions.create(
        model=model,
        temperature=0,
        messages=[
            {"role": "system", "content": system_prompt},
            {
                "role": "user",
                "content": json.dumps(
                    [topic.model_dump() for topic in topics_payload],
                    ensure_ascii=False,
                ),
            },
        ],
        response_format={"type": "json_object"},
    )

    return MergeTopics.model_validate_json(response.choices[0].message.content)


def summarize_topics_with_pydantic(
    topics_payload: List[TopicPayload], client: OpenAI, model: str,
) -> TopicSummaries:
    system_prompt = """
    You are an expert topic modeler.

    Given topic keywords and representative documents, produce a concise topic label
    and summary for each topic.

    Rules:
    - Output ONLY a JSON object.
    - Label should be short, ideally 3-7 words.
    - Summary should be 1-3 sentences.
    - Never start the summary with "This topic". Start directly with the subject matter.
    - Go straight to the point.
    Schema: {"summaries": [{"topic_id": int, "label": str, "summary": str}]}
    """

    response = client.chat.completions.create(
        model=model,
        temperature=0,
        messages=[
            {"role": "system", "content": system_prompt},
            {
                "role": "user",
                "content": json.dumps(
                    [topic.model_dump() for topic in topics_payload],
                    ensure_ascii=False,
                ),
            },
        ],
        response_format={"type": "json_object"},
    )

    return TopicSummaries.model_validate_json(response.choices[0].message.content)


def filter_relevant_topics(
    topics_payload: List[TopicPayload], query: str, client: OpenAI, model: str
) -> List[TopicPayload]:
    system_prompt = f"""
    The user asked: "{query}"

    You are given a list of topics extracted from a set of biomedical documents retrieved for that query.
    Some topics may be entirely off-topic.

    Your job: return the topic_ids that are relevant or meaningfully related to the user's query.
    Be inclusive — keep topics covering related biology, immunology, epidemiology, drug mechanisms,
    or public health aspects even if not explicitly stated in the query.
    Only exclude topics that are clearly unrelated (different disease area, unrelated field).

    Output ONLY a JSON object.
    Schema: {{"relevant_topic_ids": [int]}}
    """

    response = client.chat.completions.create(
        model=model,
        temperature=0,
        messages=[
            {"role": "system", "content": system_prompt},
            {
                "role": "user",
                "content": json.dumps(
                    [topic.model_dump() for topic in topics_payload],
                    ensure_ascii=False,
                ),
            },
        ],
        response_format={"type": "json_object"},
    )

    result = RelevantTopics.model_validate_json(response.choices[0].message.content)
    relevant_ids = set(result.relevant_topic_ids)
    return [t for t in topics_payload if t.topic_id in relevant_ids]


def compare_topic_periods(
    current_summaries: "TopicSummaries",
    previous_summaries: "TopicSummaries",
    current_period_label: str,
    previous_period_label: str,
    client: OpenAI,
    model: str,
) -> PeriodComparison:
    system_prompt = f"""
You are a biomedical research analyst comparing two consecutive literature periods.

Current period ({current_period_label}):
{json.dumps([{"label": s.label, "summary": s.summary} for s in current_summaries.summaries], ensure_ascii=False)}

Previous period ({previous_period_label}):
{json.dumps([{"label": s.label, "summary": s.summary} for s in previous_summaries.summaries], ensure_ascii=False)}

Your tasks:
1. Write a concise narrative (3-5 sentences) comparing the two periods. What shifted? What is new? What faded?
2. List the labels of current-period topics that have NO clear equivalent in the previous period — these are emerging topics.
3. List the labels of previous-period topics that have NO clear equivalent in the current period — these disappeared.

Be specific and use the actual topic labels. Do not invent new labels.

Output ONLY a JSON object matching this schema:
{{
  "narrative": "string",
  "emerging_topic_labels": ["string"],
  "disappeared_topic_labels": ["string"]
}}
"""

    response = client.chat.completions.create(
        model=model,
        temperature=0,
        messages=[{"role": "user", "content": system_prompt}],
        response_format={"type": "json_object"},
    )

    return PeriodComparison.model_validate_json(response.choices[0].message.content)


def semantic_rerank(
    query: str,
    docs: "List[OpenAlexWork]",
    embedding_model: SentenceTransformer,
    top_n: int = 200,
) -> "List[OpenAlexWork]":
    if len(docs) <= top_n:
        return docs

    texts = [doc.abstract or doc.title or "" for doc in docs]

    query_emb = embedding_model.encode([query], normalize_embeddings=True)
    doc_embs = embedding_model.encode(texts, normalize_embeddings=True, show_progress_bar=False)

    scores = (doc_embs @ query_emb.T).squeeze()
    top_indices = np.argsort(scores)[::-1][:top_n]
    return [docs[i] for i in top_indices]


def run_topic_model(
    docs: List[str],
    client: OpenAI,
    model: str,
    embedding_model: SentenceTransformer,
    query: str | None = None,
    k: int = K,
    max_doc_chars: int = MAX_DOC_CHARS,
) -> TopicModelResult:

    random.seed(42)
    np.random.seed(42)

    embeddings = embedding_model.encode(
        truncate_docs(
            docs,
            max_chars=max_doc_chars,
        ),
        show_progress_bar=True,
        normalize_embeddings=True,
    )

    vectorizer_model = CountVectorizer(
        stop_words="english",
        ngram_range=(1, 3),
        min_df=2,
        max_df=0.9,
        token_pattern=r"(?u)\b[a-zA-Z][a-zA-Z\-]{2,}\b",
    )

    topic_model = BERTopic(
        embedding_model=embedding_model,
        umap_model=BaseDimensionalityReduction(),
        hdbscan_model=KMeans(n_clusters=k, random_state=42, n_init=1),
        vectorizer_model=vectorizer_model,
        calculate_probabilities=False,
        verbose=True,
    )
    assignments, _ = topic_model.fit_transform(docs, embeddings=embeddings)

    payload = build_topics_payload(topic_model)
    merges = propose_merges(payload, client, model)
    if merges.merges:
        topic_model.merge_topics(docs, merges.merges)
        assignments = topic_model.topics_

    final_payload = build_topics_payload(topic_model)
    if query:
        final_payload = filter_relevant_topics(final_payload, query, client, model)

    summaries = summarize_topics_with_pydantic(final_payload, client, model)
    return TopicModelResult(summaries=summaries, topic_assignments=assignments, topic_model=topic_model)


def main() -> None:
    from sklearn.datasets import fetch_20newsgroups
    from sentence_transformers import SentenceTransformer

    # -----------------------------
    # 1. Load docs
    # -----------------------------
    docs = fetch_20newsgroups(
        subset="test",
        remove=("headers", "footers", "quotes"),
    )["data"][:100]


    # -----------------------------
    # 2. Embed once
    # -----------------------------
    sbert_model = SentenceTransformer(EMBEDDING_MODEL_NAME)
    embeddings = sbert_model.encode(
        docs,
        show_progress_bar=True,
        normalize_embeddings=True,
    )


    # -----------------------------
    # 3. Fit BERTopic with fixed K=15
    #    No UMAP: raw embeddings go to KMeans
    # -----------------------------
    empty_dimensionality_model = BaseDimensionalityReduction()

    cluster_model = KMeans(
        n_clusters=K,
        random_state=0,
        n_init="auto",
    )

    topic_model = BERTopic(
        embedding_model=sbert_model,
        umap_model=empty_dimensionality_model,
        hdbscan_model=cluster_model,
        calculate_probabilities=False,
        verbose=True,
    )

    topics, probs = topic_model.fit_transform(docs, embeddings=embeddings)

    df_initial = topic_model.get_topic_info()
    print(df_initial)


    # -----------------------------
    # 4. Ask LLM for merge groups
    # -----------------------------

    initial_payload = build_topics_payload(topic_model)
    merges = propose_merges(initial_payload)

    print("Proposed merges:")
    print(merges.model_dump_json(indent=2))


    # -----------------------------
    # 5. Merge topics
    # -----------------------------

    if merges.merges:
        topic_model.merge_topics(docs, merges.merges)

    df_merged = topic_model.get_topic_info()
    print(df_merged)


    # -----------------------------
    # 6. Pydantic summaries
    # -----------------------------

    merged_payload = build_topics_payload(topic_model)
    topic_summaries = summarize_topics_with_pydantic(merged_payload)

    print(topic_summaries.model_dump_json(indent=2))

    # -----------------------------
    # 7. Optional: also update BERTopic representations
    #    using BERTopic's OpenAI representation model
    # -----------------------------

    summarization_prompt = """
    I have a topic that is described by the following keywords: [KEYWORDS]
    
    In this topic, the following documents are a small but representative subset of all documents in the topic:
    [DOCUMENTS]
    
    Based on the information above, give a description in this format:
    topic: <description>
    
    Rules:
    - Do not mention "This topic".
    - Go straight to the point.
    - Use 2-3 sentences max.
    """
    client = OpenAI(base_url=os.getenv("URL_BASE"), api_key=os.getenv("LLM_API_KEY"))
    model = os.getenv("LLM_MODEL", "glm-5")
    representation_model = OpenAIBertTopic(
        client,
        model=model,
        prompt=summarization_prompt,
    )

    topic_model.update_topics(
        docs=docs,
        representation_model=representation_model,
    )

    df_final = topic_model.get_topic_info()
    print(df_final)


if __name__ == "__main__":
    main()
