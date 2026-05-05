from src.utils.llm import MESSAGE_NOT_MEDICAL, extract_medical_keywords
from src.utils.open_alex import get_200_openalex_last_months
from src.utils.topic_model_llm import run_topic_model


def main():
    result = extract_medical_keywords("What are the latest treatments for pediatric malaria?")
    if not result.is_medical:
        print(MESSAGE_NOT_MEDICAL)
        return

    docs_raw = get_200_openalex_last_months(" OR ".join(result.keywords), limit=200)
    docs = [d.abstract for d in docs_raw if d.abstract]
    print(f"Fetched {len(docs)} abstracts")

    summaries = run_topic_model(docs)
    print(summaries.model_dump_json(indent=2))


if __name__ == "__main__":
    main()
