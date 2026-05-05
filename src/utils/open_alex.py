"""
This code was 95% produced by ChatGPT for querying OpenAlex API
"""
from __future__ import annotations

import os
from datetime import date
from datetime import timedelta
from typing import Any

import requests
from dotenv import load_dotenv
from pydantic import BaseModel
from pydantic import Field


class OpenAlexWork(BaseModel):
    doi: str
    abstract: str
    publication_date: str | None = Field(default=None)
    title: str | None = Field(default=None)


OPENALEX_BASE_URL = "https://api.openalex.org/works"

def _get_api_key() -> str:
    load_dotenv()
    key = os.getenv("OPEN_ALEX_KEY")
    if not key:
        raise RuntimeError("OPEN_ALEX_KEY is not set in environment or .env file")
    return key

def reconstruct_abstract(abstract_inverted_index: dict[str, list[int]] | None) -> str | None:
    if not abstract_inverted_index:
        return None

    max_pos = -1
    for positions in abstract_inverted_index.values():
        if positions:
            max_pos = max(max_pos, max(positions))
    if max_pos < 0:
        return None

    words = [""] * (max_pos + 1)
    for token, positions in abstract_inverted_index.items():
        for pos in positions:
            if 0 <= pos < len(words):
                words[pos] = token

    return " ".join(w for w in words if w).strip() or None


def last_n_months_date_range(n_months: int = 3, today: date | None = None) -> tuple[str, str]:
    today = today or date.today()
    first_of_this_month = today.replace(day=1)
    last_day = first_of_this_month - timedelta(days=1)
    first_day = last_day.replace(day=1)
    for _ in range(n_months - 1):
        first_day = (first_day - timedelta(days=1)).replace(day=1)
    return first_day.isoformat(), last_day.isoformat()


def normalize_doi(raw_doi: str | None) -> str | None:
    if not raw_doi:
        return None
    return raw_doi.removeprefix("https://doi.org/")


def get_100_openalex_last_months(
    keywords: str,
    limit: int = 100,
    mailto: str | None = None,
    n_months: int = 3,
) -> list[OpenAlexWork]:
    from_date, to_date = last_n_months_date_range(n_months=n_months)
    headers = {
        "User-Agent": f"openalex-keyword-script/1.0 ({mailto})" if mailto else "openalex-keyword-script/1.0"
    }

    collected: list[dict[str, Any]] = []
    cursor = "*"
    page_size = min(limit, 100)

    while len(collected) < limit:
        params = {
            "search": keywords,
            "filter": f"from_publication_date:{from_date},to_publication_date:{to_date},has_abstract:true",
            "sort": "publication_date:desc",
            "per-page": page_size,
            "cursor": cursor,
            "api_key": _get_api_key(),
        }

        response = requests.get(OPENALEX_BASE_URL, params=params, headers=headers, timeout=30)
        response.raise_for_status()
        payload = response.json()

        works = payload.get("results", [])
        if not works:
            break

        for work in works:
            if work.get("language") != "en":
                continue

            doi = normalize_doi((work.get("ids") or {}).get("doi"))
            abstract = reconstruct_abstract(work.get("abstract_inverted_index"))
            if not doi or not abstract:
                continue

            collected.append(
                OpenAlexWork(
                    doi=doi,
                    publication_date=work.get("publication_date"),
                    title=work.get("display_name"),
                    abstract=abstract,
                )
            )

            if len(collected) >= limit:
                break
        meta = payload.get("meta", {})
        next_cursor = meta.get("next_cursor")
        if not next_cursor or next_cursor == cursor:
            break
        cursor = next_cursor

    return collected[:limit]

def main():
    results = get_100_openalex_last_months(keywords="pediatrics malaria")
    print(1)

if __name__ == "__main__":
    main()
