import os

from dotenv import load_dotenv
from openai import OpenAI

from src.utils.llm import MESSAGE_NOT_MEDICAL, extract_medical_keywords
from src.utils.open_alex import get_200_openalex_last_months
from src.utils.topic_model_llm import run_topic_model

load_dotenv()


def main():
    client = OpenAI(base_url=os.getenv("URL_BASE"), api_key=os.getenv("LLM_API_KEY"))
    model = os.getenv("LLM_MODEL", "glm-5")

    result = extract_medical_keywords("latest developments in malaria treatment")
    if not result.is_medical:
        print(MESSAGE_NOT_MEDICAL)
        return

    docs_raw = get_200_openalex_last_months(" OR ".join(result.keywords), limit=200)
    docs = [d.abstract for d in docs_raw if d.abstract]
    print(f"Fetched {len(docs)} abstracts")

    topic_result = run_topic_model(docs, client=client, model=model)
    print(topic_result.summaries.model_dump_json(indent=2))
    df = topic_result.topic_model.get_topic_info()


if __name__ == "__main__":
    main()
