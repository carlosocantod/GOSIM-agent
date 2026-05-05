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
2. If the query is medical/biomedical, extract concise search keywords suitable for OpenAlex or PubMed-style literature search.
3. If the query is not medical/biomedical, return is_medical=false and keywords=[].

Keyword rules:
- Return 2 to 8 keywords or short key phrases.
- Each keyword must be independently meaningful as a biomedical literature search query.
- Every keyword must include or be anchored to a biomedical concept (e.g., disease, drug, population, outcome, intervention, diagnostic, or biological process).
- Standalone geographic, demographic, temporal, or generic context terms are NOT allowed (e.g., "Latin America", "children", "2020", "hospital").
- If context such as geography or population is important, it must be combined with a biomedical concept:
  - Good: "dengue Latin America", "pediatric malaria", "cancer survival Europe"
  - Bad: "Latin America", "children", "Europe"
- Prefer canonical biomedical phrasing:
  - "pediatric malaria" instead of "malaria in children"
  - "type 2 diabetes" instead of "diabetes type II"
- Include both condition and intervention when relevant:
  - e.g., "metformin type 2 diabetes", "malaria vaccine efficacy"
- Preserve important qualifiers such as:
  - population (pediatric, neonatal, elderly)
  - severity (severe, resistant)
  - study type (randomized trial, cohort study)
  - purpose (diagnosis, treatment, prevention)
- Avoid filler or vague terms:
  - Do NOT include "latest", "best", "what are", etc.
- Do NOT invent concepts not supported by the query.

Output format:
Return only valid JSON matching exactly this schema:
{
  "is_medical": true | false,
  "keywords": ["string"]
}

Examples:

User: "What are the latest treatments for pediatric malaria?"
Response:
{"is_medical": true, "keywords": ["pediatric malaria", "malaria treatment", "antimalarial therapy"]}

User: "How does climate change affect crop yields?"
Response:
{"is_medical": false, "keywords": []}

User: "Does metformin reduce cardiovascular risk in type 2 diabetes?"
Response:
{"is_medical": true, "keywords": ["metformin type 2 diabetes", "cardiovascular risk diabetes", "metformin cardiovascular outcomes"]}

User: "Can dogs get Lyme disease?"
Response:
{"is_medical": true, "keywords": ["Lyme disease dogs", "canine Lyme disease", "veterinary Lyme disease"]}

User: "Write me a workout plan"
Response:
{"is_medical": false, "keywords": []}

User: "Epidemiology of dengue and malaria in Latin America"
Response:
{"is_medical": true, "keywords": ["dengue epidemiology Latin America", "malaria epidemiology Latin America", "dengue malaria epidemiology"]}
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
