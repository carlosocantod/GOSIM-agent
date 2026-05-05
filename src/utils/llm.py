from __future__ import annotations

import os

from dotenv import load_dotenv
from openai import OpenAI
from pydantic import BaseModel


class MedicalQueryAnalysis(BaseModel):
    is_medical: bool
    keywords: list[str]


PROMPT_MEDICAL_QUERY_ANALYSIS = """
You are a medical research assistant. Determine if the user query is related to the medical or biomedical field. "
If it is, extract the key medical terms suitable for a literature search. 
If it is not medical, return is_medical=false and an empty keywords list. 
Respond in JSON with the schema: {"is_medical": bool, "keywords": [str]}
"""

MESSAGE_NOT_MEDICAL = "Your query is outside the scope of this app. Please only do queries within the medical domain"

def _get_client() -> OpenAI:
    load_dotenv()
    return OpenAI(
        base_url=os.getenv("URL_BASE"),
        api_key=os.getenv("LLM_API_KEY"),
    )


def extract_medical_keywords(query: str) -> MedicalQueryAnalysis:
    client = _get_client()
    model = os.getenv("LLM_MODEL", "glm-5")

    response = client.chat.completions.create(
        model=model,
        messages=[
            {
                "role": "system",
                "content": PROMPT_MEDICAL_QUERY_ANALYSIS
            },
            {"role": "user", "content": query},
        ],
        # no native pydantic support. we do our best here
        response_format={"type": "json_object"},
    )

    return MedicalQueryAnalysis.model_validate_json(response.choices[0].message.content)


def main() -> None:
    from src.utils.open_alex import get_200_openalex_last_months

    result = extract_medical_keywords("What are the latest treatments for pediatric malaria?")
    if not result.is_medical:
        print(MESSAGE_NOT_MEDICAL)
        return
    print(result.keywords)
    query_results = get_200_openalex_last_months("OR ".join(result.keywords), limit=2)
    print([q.title for q in query_results])


if __name__ == "__main__":
    main()
