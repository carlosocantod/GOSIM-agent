from src.utils.llm import MESSAGE_NOT_MEDICAL
from src.utils.llm import extract_medical_keywords
from src.utils.open_alex import get_200_openalex_last_months


def main():

    result = extract_medical_keywords("What are the latest treatments for pediatric malaria?")
    if not result.is_medical:
        print(MESSAGE_NOT_MEDICAL)
        return
    print(result.keywords)
    query_results = get_200_openalex_last_months("OR ".join(result.keywords), limit=100)
    print([q.title for q in query_results])


if __name__ == "__main__":
    main()
