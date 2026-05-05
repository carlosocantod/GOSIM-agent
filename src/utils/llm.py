from __future__ import annotations

import os

from dotenv import load_dotenv
from openai import OpenAI
from pydantic import BaseModel


class MedicalQueryAnalysis(BaseModel):
    is_medical: bool
    keywords: list[str]


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
                "content": (
                    "You are a medical research assistant. "
                    "Determine if the user query is related to the medical or biomedical field. "
                    "If it is, extract the key medical terms suitable for a literature search. "
                    "If it is not medical, return is_medical=false and an empty keywords list. "
                    'Respond in JSON with the schema: {"is_medical": bool, "keywords": [str]}'
                ),
            },
            {"role": "user", "content": query},
        ],
        # no native pydantic support. we do our best here
        response_format={"type": "json_object"},
    )

    return MedicalQueryAnalysis.model_validate_json(response.choices[0].message.content)


def main():
    result = extract_medical_keywords("What are the latest treatments for pediatric malaria?")
    print(result)


if __name__ == "__main__":
    main()
