from __future__ import annotations

import os

from dotenv import load_dotenv
from openai import OpenAI
from pydantic import BaseModel


class MedicalQueryAnalysis(BaseModel):
    is_medical: bool
    keywords: list[str]


PROMPT_MEDICAL_QUERY_ANALYSIS = """
You are a medical literature-search query analyzer.

Your task:
1. Decide whether the user's query is about medicine, health, biomedical science, public health, clinical care, drugs, diseases, diagnostics, epidemiology, biology relevant to human or animal health, or healthcare systems.
2. If the query is medical/biomedical, generate diverse search keywords that together cast a wide net over the relevant literature.
3. If the query is not medical/biomedical, return is_medical=false and keywords=[].

Keyword rules:
- Return 5 to 10 keywords or short key phrases.
- Keywords MUST be DIVERSE — cover different angles of the topic:
  - Core disease/condition (e.g., "malaria", "Plasmodium falciparum")
  - Treatments and interventions (e.g., "artemisinin combination therapy", "antimalarial drugs")
  - Prevention and public health (e.g., "malaria vector control", "insecticide-treated nets")
  - Immunology and biology (e.g., "malaria immunity", "Plasmodium life cycle")
  - Epidemiology and burden (e.g., "malaria epidemiology", "malaria endemic regions")
  - Diagnostics (e.g., "malaria rapid diagnostic test") if relevant
- Do NOT return synonyms of the same concept. Each keyword must open a different slice of the literature.
- Every keyword must be anchored to a biomedical concept.
- Prefer canonical biomedical phrasing.
- Preserve important qualifiers from the query (pediatric, severe, resistant, etc.).
- Do NOT include filler terms ("latest", "best", "what are").
- Do NOT invent concepts not supported by the query.

Output format:
Return only valid JSON matching exactly this schema:
{
  "is_medical": true | false,
  "keywords": ["string"]
}

Examples:

User: "What are the latest treatments for malaria?"
Response:
{"is_medical": true, "keywords": ["malaria", "artemisinin combination therapy", "antimalarial drug resistance", "malaria vaccine", "malaria vector control", "Plasmodium falciparum treatment", "malaria clinical trials"]}

User: "What are the latest treatments for pediatric malaria?"
Response:
{"is_medical": true, "keywords": ["pediatric malaria", "malaria children treatment", "artemisinin pediatric", "malaria vaccine children", "severe malaria children", "Plasmodium falciparum pediatric", "malaria child mortality"]}

User: "How does climate change affect crop yields?"
Response:
{"is_medical": false, "keywords": []}

User: "Does metformin reduce cardiovascular risk in type 2 diabetes?"
Response:
{"is_medical": true, "keywords": ["metformin type 2 diabetes", "cardiovascular risk diabetes", "metformin cardiovascular outcomes", "diabetes heart disease", "HbA1c cardiovascular events", "insulin resistance cardiovascular"]}

User: "Write me a workout plan"
Response:
{"is_medical": false, "keywords": []}
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
    from src.utils.open_alex import get_openalex_papers_last_months

    result = extract_medical_keywords("What are the latest treatments for pediatric malaria?")
    if not result.is_medical:
        print(MESSAGE_NOT_MEDICAL)
        return
    print(result.keywords)
    query_results = get_openalex_papers_last_months("OR ".join(result.keywords), limit=2)
    print([q.title for q in query_results])


if __name__ == "__main__":
    main()
