import json
from typing import List

from bertopic import BERTopic
from bertopic.dimensionality import BaseDimensionalityReduction
from bertopic.representation import OpenAI as OpenAIBertTopic
from dotenv import load_dotenv
from openai import OpenAI
from pydantic import BaseModel
from pydantic import Field
from sklearn.cluster import KMeans

# -----------------------------
# Config
# -----------------------------

K = 15
MAX_DOC_CHARS = 300
EMBEDDING_MODEL_NAME = "sentence-transformers/paraphrase-MiniLM-L3-v2"
OPENAI_MODEL = "gpt-4.1-mini"

load_dotenv()
client = OpenAI()


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


def propose_merges(topics_payload: List[TopicPayload]) -> MergeTopics:
    system_prompt = """
    You are an expert topic modeler.
    
    You are given topics with:
    - topic_id
    - top_words
    - representative documents
    
    Your job:
    Group topics that are semantically similar and should be merged.
    
    Rules:
    - Output ONLY the requested structured object.
    - Each topic_id must appear at most once.
    - Do NOT include singletons.
    - Do NOT duplicate topic_ids across groups.
    - Only merge if clearly similar.
    - Prioritize merges of smaller, semantically overlapping topics.
    - Do not merge unrelated topics just to reduce the total number.
    """

    response = client.chat.completions.parse(
        model=OPENAI_MODEL,
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
        response_format=MergeTopics,
    )

    return response.choices[0].message.parsed


def summarize_topics_with_pydantic(
    topics_payload: List[TopicPayload],
) -> TopicSummaries:
    system_prompt = """
    You are an expert topic modeler.
    
    Given topic keywords and representative documents, produce a concise topic label
    and summary for each topic.
    
    Rules:
    - Output ONLY the requested structured object.
    - Label should be short, ideally 3-7 words.
    - Summary should be 1-3 sentences.
    - Do not say "This topic is about".
    - Go straight to the point.
    """

    response = client.chat.completions.parse(
        model=OPENAI_MODEL,
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
        response_format=TopicSummaries,
    )

    return response.choices[0].message.parsed

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

    representation_model = OpenAIBertTopic(
        client,
        model=OPENAI_MODEL,
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
