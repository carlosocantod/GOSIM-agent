from __future__ import annotations

import json
from datetime import date
from typing import Literal, TypedDict

from openai import OpenAI
from pydantic import BaseModel, Field
from sentence_transformers import SentenceTransformer

from src.utils.llm import MESSAGE_NOT_MEDICAL
from src.utils.llm import MedicalQueryAnalysis
from src.utils.llm import extract_medical_keywords
from src.utils.open_alex import OpenAlexWork
from src.utils.open_alex import get_openalex_papers_for_period
from src.utils.open_alex import last_n_months_date_range
from src.utils.open_alex import previous_period_date_range
from src.utils.topic_model_llm import PeriodComparison
from src.utils.topic_model_llm import TopicSummaries
from src.utils.topic_model_llm import compare_topic_periods
from src.utils.topic_model_llm import run_topic_model
from src.utils.topic_model_llm import semantic_rerank

try:  # LangGraph is optional at import time so local syntax checks still work.
    from langgraph.graph import END, START, StateGraph
except ImportError:  # pragma: no cover - runtime fallback for environments without LangGraph.
    END = "__end__"
    START = "__start__"
    StateGraph = None


N_MONTHS_CURRENT = 3
N_MONTHS_BASELINE = 6
MAX_SEARCH_REFINEMENTS = 1


class ResearchPlan(BaseModel):
    interpreted_question: str
    search_focus: list[str]
    inclusion_criteria: list[str]
    exclusion_criteria: list[str]
    evidence_types_to_prioritize: list[str]
    suggested_followups: list[str]


class SearchQualityDecision(BaseModel):
    status: Literal["good", "broaden", "narrow", "insufficient"]
    reason: str
    revised_keywords: list[str] = Field(default_factory=list)


class FollowupQuestions(BaseModel):
    questions: list[str]


class CommunicationRecommendation(BaseModel):
    headline: str
    recommendation: str
    caution: str
    audience_angle: str


class ResearchAgentResult(BaseModel):
    query: str
    analysis: MedicalQueryAnalysis | None = None
    is_medical: bool = True
    message: str | None = None
    agent_steps: list[str] = Field(default_factory=list)
    plan: ResearchPlan | None = None
    search_quality: SearchQualityDecision | None = None
    keywords_used: list[str] = Field(default_factory=list)
    current_summaries: TopicSummaries | None = None
    current_assignments: list[int] = Field(default_factory=list)
    current_docs: list[OpenAlexWork] = Field(default_factory=list)
    previous_summaries: TopicSummaries | None = None
    previous_assignments: list[int] = Field(default_factory=list)
    previous_docs: list[OpenAlexWork] = Field(default_factory=list)
    comparison: PeriodComparison | None = None
    current_label: str = ""
    previous_label: str = ""
    followups: list[str] = Field(default_factory=list)
    recommendation: CommunicationRecommendation | None = None

    model_config = {"arbitrary_types_allowed": True}


class ResearchAgentState(TypedDict, total=False):
    query: str
    audience: str
    client: OpenAI
    model: str
    embedding_model: SentenceTransformer
    analysis: MedicalQueryAnalysis
    is_medical: bool
    message: str
    agent_steps: list[str]
    plan: ResearchPlan
    search_quality: SearchQualityDecision
    keywords: list[str]
    current_from: str
    current_to: str
    previous_from: str
    previous_to: str
    current_label: str
    previous_label: str
    current_docs: list[OpenAlexWork]
    current_with_abstract: list[OpenAlexWork]
    current_fetched: int
    current_reranked: int
    previous_docs: list[OpenAlexWork]
    previous_with_abstract: list[OpenAlexWork]
    previous_fetched: int
    previous_reranked: int
    current_summaries: TopicSummaries
    current_assignments: list[int]
    previous_summaries: TopicSummaries
    previous_assignments: list[int]
    comparison: PeriodComparison
    followups: list[str]
    recommendation: CommunicationRecommendation
    search_refinements: int


def _step(state: ResearchAgentState, message: str) -> list[str]:
    return [*state.get("agent_steps", []), message]


def _format_period_label(from_date: str, to_date: str) -> str:
    from_dt = date.fromisoformat(from_date)
    to_dt = date.fromisoformat(to_date)
    return f"{from_dt.strftime('%b %Y')} – {to_dt.strftime('%b %Y')}"


def _json_chat(client: OpenAI, model: str, system_prompt: str, payload: object) -> str:
    response = client.chat.completions.create(
        model=model,
        temperature=0,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": json.dumps(payload, ensure_ascii=False)},
        ],
        response_format={"type": "json_object"},
    )
    return response.choices[0].message.content


def _fetch_and_rerank(
    keywords: list[str],
    query: str,
    from_date: str,
    to_date: str,
    embedding_model: SentenceTransformer,
) -> tuple[list[OpenAlexWork], int, int]:
    search_query = " OR ".join(keywords)
    docs = get_openalex_papers_for_period(search_query, from_date=from_date, to_date=to_date, limit=500)
    fetched = len(docs)
    reranked = semantic_rerank(query, docs, embedding_model=embedding_model, top_n=200)
    return reranked, fetched, len(reranked)


def analyze_query_node(state: ResearchAgentState) -> ResearchAgentState:
    analysis = extract_medical_keywords(state["query"])
    if not analysis.is_medical:
        return {
            "analysis": analysis,
            "is_medical": False,
            "message": MESSAGE_NOT_MEDICAL,
            "agent_steps": _step(state, "❌ Query is outside the medical/biomedical scope."),
        }

    return {
        "analysis": analysis,
        "is_medical": True,
        "keywords": analysis.keywords,
        "search_refinements": 0,
        "agent_steps": _step(state, f"✅ Medical query detected. Keywords: {', '.join(analysis.keywords)}"),
    }


def create_plan_node(state: ResearchAgentState) -> ResearchAgentState:
    system_prompt = """
You are helping a science communication team use an LLM-enhanced topic model.
Create a concise research plan for a biomedical literature scan.
Return ONLY JSON matching this schema:
{
  "interpreted_question": "string",
  "search_focus": ["string"],
  "inclusion_criteria": ["string"],
  "exclusion_criteria": ["string"],
  "evidence_types_to_prioritize": ["string"],
  "suggested_followups": ["string"]
}
Keep each list to 3-5 short items.
"""
    payload = {
        "query": state["query"],
        "keywords": state.get("keywords", []),
        "audience": state.get("audience", "General public"),
    }
    plan = ResearchPlan.model_validate_json(_json_chat(state["client"], state["model"], system_prompt, payload))
    return {
        "plan": plan,
        "agent_steps": _step(state, "✅ Created a literature scan plan."),
    }


def set_periods_node(state: ResearchAgentState) -> ResearchAgentState:
    curr_from, curr_to = last_n_months_date_range(N_MONTHS_CURRENT)
    prev_from, prev_to = previous_period_date_range(N_MONTHS_BASELINE, n_months_current=N_MONTHS_CURRENT)
    return {
        "current_from": curr_from,
        "current_to": curr_to,
        "previous_from": prev_from,
        "previous_to": prev_to,
        "current_label": _format_period_label(curr_from, curr_to),
        "previous_label": _format_period_label(prev_from, prev_to),
        "agent_steps": _step(state, "✅ Selected current and baseline literature windows."),
    }


def fetch_current_node(state: ResearchAgentState) -> ResearchAgentState:
    docs, fetched, reranked = _fetch_and_rerank(
        state.get("keywords", []),
        state["query"],
        state["current_from"],
        state["current_to"],
        state["embedding_model"],
    )
    with_abstract = [doc for doc in docs if doc.abstract]
    return {
        "current_docs": docs,
        "current_with_abstract": with_abstract,
        "current_fetched": fetched,
        "current_reranked": reranked,
        "agent_steps": _step(
            state,
            f"✅ Retrieved {fetched} current-period papers; kept {len(with_abstract)} with abstracts after reranking.",
        ),
    }


def inspect_search_quality_node(state: ResearchAgentState) -> ResearchAgentState:
    fetched = state.get("current_fetched", 0)
    with_abstract = len(state.get("current_with_abstract", []))

    if with_abstract == 0:
        decision = SearchQualityDecision(
            status="insufficient",
            reason="No abstracts were available for topic modeling.",
            revised_keywords=state.get("keywords", []),
        )
        return {"search_quality": decision, "agent_steps": _step(state, "⚠️ No usable abstracts found.")}

    sample_docs = state.get("current_with_abstract", [])[:12]
    sample = [
        {
            "title": getattr(doc, "title", None),
            "abstract": (getattr(doc, "abstract", "") or "")[:700],
        }
        for doc in sample_docs
    ]
    system_prompt = """
You are evaluating whether a biomedical literature search is good enough for topic modeling.
Return ONLY JSON matching this schema:
{
  "status": "good" | "broaden" | "narrow" | "insufficient",
  "reason": "string",
  "revised_keywords": ["string"]
}
Rules:
- Use "good" if the sample looks relevant and there are enough abstracts.
- Use "broaden" if the search is too sparse.
- Use "narrow" if many sampled papers seem off-topic.
- Use "insufficient" only when topic modeling should not continue.
- revised_keywords should contain 2-8 biomedical keyword phrases and be empty when status is "good".
"""
    payload = {
        "query": state["query"],
        "current_keywords": state.get("keywords", []),
        "fetched_count": fetched,
        "abstract_count": with_abstract,
        "sample_documents": sample,
    }

    if fetched < 15 or with_abstract < 8:
        # Let the LLM propose broader terms, but make the sparse-search issue explicit.
        payload["quality_hint"] = "Sparse search: prefer broaden unless the sample is completely off-topic."

    decision = SearchQualityDecision.model_validate_json(
        _json_chat(state["client"], state["model"], system_prompt, payload)
    )

    if decision.status in {"broaden", "narrow"} and not decision.revised_keywords:
        decision.revised_keywords = state.get("keywords", [])

    return {
        "search_quality": decision,
        "agent_steps": _step(state, f"✅ Search quality check: {decision.status} — {decision.reason}"),
    }


def refine_search_node(state: ResearchAgentState) -> ResearchAgentState:
    decision = state.get("search_quality")
    revised = decision.revised_keywords if decision and decision.revised_keywords else state.get("keywords", [])
    return {
        "keywords": revised,
        "search_refinements": state.get("search_refinements", 0) + 1,
        "agent_steps": _step(state, f"🔁 Refined search keywords: {', '.join(revised)}"),
    }


def topic_model_current_node(state: ResearchAgentState) -> ResearchAgentState:
    docs = state.get("current_with_abstract", [])
    if not docs:
        return {
            "current_summaries": TopicSummaries(summaries=[]),
            "current_assignments": [],
            "agent_steps": _step(state, "⚠️ Skipped current-period topic modeling because no abstracts were available."),
        }

    result = run_topic_model(
        [doc.abstract for doc in docs if doc.abstract],
        client=state["client"],
        model=state["model"],
        embedding_model=state["embedding_model"],
        query=state["query"],
    )
    return {
        "current_summaries": result.summaries,
        "current_assignments": result.topic_assignments,
        "agent_steps": _step(state, f"✅ Modeled current literature into {len(result.summaries.summaries)} relevant topics."),
    }


def fetch_previous_node(state: ResearchAgentState) -> ResearchAgentState:
    docs, fetched, reranked = _fetch_and_rerank(
        state.get("keywords", []),
        state["query"],
        state["previous_from"],
        state["previous_to"],
        state["embedding_model"],
    )
    with_abstract = [doc for doc in docs if doc.abstract]
    return {
        "previous_docs": docs,
        "previous_with_abstract": with_abstract,
        "previous_fetched": fetched,
        "previous_reranked": reranked,
        "agent_steps": _step(state, f"✅ Retrieved {fetched} baseline-period papers; kept {len(with_abstract)} with abstracts."),
    }


def topic_model_previous_node(state: ResearchAgentState) -> ResearchAgentState:
    docs = state.get("previous_with_abstract", [])
    if not docs:
        return {
            "previous_summaries": TopicSummaries(summaries=[]),
            "previous_assignments": [],
            "agent_steps": _step(state, "⚠️ No baseline abstracts available for topic modeling."),
        }

    result = run_topic_model(
        [doc.abstract for doc in docs if doc.abstract],
        client=state["client"],
        model=state["model"],
        embedding_model=state["embedding_model"],
        query=state["query"],
    )
    return {
        "previous_summaries": result.summaries,
        "previous_assignments": result.topic_assignments,
        "agent_steps": _step(state, f"✅ Modeled baseline literature into {len(result.summaries.summaries)} relevant topics."),
    }


def compare_periods_node(state: ResearchAgentState) -> ResearchAgentState:
    comparison = compare_topic_periods(
        state.get("current_summaries", TopicSummaries(summaries=[])),
        state.get("previous_summaries", TopicSummaries(summaries=[])),
        state.get("current_label", "Current period"),
        state.get("previous_label", "Previous period"),
        state["client"],
        state["model"],
    )
    return {
        "comparison": comparison,
        "agent_steps": _step(state, "✅ Compared current topics with the baseline period."),
    }


def generate_followups_node(state: ResearchAgentState) -> ResearchAgentState:
    system_prompt = """
You are a biomedical science communication assistant.
Generate useful follow-up questions a user might ask after seeing topic-model results.
Return ONLY JSON matching this schema: {"questions": ["string"]}
Rules:
- Return 5 questions.
- Make them actionable for science communication, not generic.
- Include at least one question about uncertainty or evidence strength.
"""
    payload = {
        "query": state["query"],
        "audience": state.get("audience", "General public"),
        "current_topics": [s.model_dump() for s in state.get("current_summaries", TopicSummaries(summaries=[])).summaries],
        "comparison": state.get("comparison").model_dump() if state.get("comparison") else None,
    }
    followups = FollowupQuestions.model_validate_json(_json_chat(state["client"], state["model"], system_prompt, payload))
    return {
        "followups": followups.questions,
        "agent_steps": _step(state, "✅ Suggested communication-focused follow-up questions."),
    }


def generate_recommendation_node(state: ResearchAgentState) -> ResearchAgentState:
    system_prompt = """
You are a biomedical science communication strategist.
Based on topic-model outputs, recommend the strongest communication angle.
Return ONLY JSON matching this schema:
{
  "headline": "string",
  "recommendation": "string",
  "caution": "string",
  "audience_angle": "string"
}
Rules:
- Be accurate and cautious.
- Do not claim clinical efficacy unless the topic summaries support it.
- Mention uncertainty or limitations when appropriate.
- Write for the selected audience.
"""
    payload = {
        "query": state["query"],
        "audience": state.get("audience", "General public"),
        "current_topics": [s.model_dump() for s in state.get("current_summaries", TopicSummaries(summaries=[])).summaries],
        "comparison": state.get("comparison").model_dump() if state.get("comparison") else None,
    }
    recommendation = CommunicationRecommendation.model_validate_json(
        _json_chat(state["client"], state["model"], system_prompt, payload)
    )
    return {
        "recommendation": recommendation,
        "agent_steps": _step(state, "✅ Generated a science communication angle."),
    }


def route_after_analysis(state: ResearchAgentState) -> str:
    return "create_plan" if state.get("is_medical", True) else END


def route_after_quality(state: ResearchAgentState) -> str:
    decision = state.get("search_quality")
    if decision and decision.status == "insufficient":
        return END
    if (
        decision
        and decision.status in {"broaden", "narrow"}
        and state.get("search_refinements", 0) < MAX_SEARCH_REFINEMENTS
    ):
        return "refine_search"
    return "topic_model_current"


def build_research_graph():
    if StateGraph is None:
        return None

    graph = StateGraph(ResearchAgentState)
    graph.add_node("analyze_query", analyze_query_node)
    graph.add_node("create_plan", create_plan_node)
    graph.add_node("set_periods", set_periods_node)
    graph.add_node("fetch_current", fetch_current_node)
    graph.add_node("inspect_search_quality", inspect_search_quality_node)
    graph.add_node("refine_search", refine_search_node)
    graph.add_node("topic_model_current", topic_model_current_node)
    graph.add_node("fetch_previous", fetch_previous_node)
    graph.add_node("topic_model_previous", topic_model_previous_node)
    graph.add_node("compare_periods", compare_periods_node)
    graph.add_node("generate_followups", generate_followups_node)
    graph.add_node("generate_recommendation", generate_recommendation_node)

    graph.add_edge(START, "analyze_query")
    graph.add_conditional_edges("analyze_query", route_after_analysis)
    graph.add_edge("create_plan", "set_periods")
    graph.add_edge("set_periods", "fetch_current")
    graph.add_edge("fetch_current", "inspect_search_quality")
    graph.add_conditional_edges("inspect_search_quality", route_after_quality)
    graph.add_edge("refine_search", "fetch_current")
    graph.add_edge("topic_model_current", "fetch_previous")
    graph.add_edge("fetch_previous", "topic_model_previous")
    graph.add_edge("topic_model_previous", "compare_periods")
    graph.add_edge("compare_periods", "generate_followups")
    graph.add_edge("generate_followups", "generate_recommendation")
    graph.add_edge("generate_recommendation", END)
    return graph.compile()


def _run_linear_fallback(state: ResearchAgentState) -> ResearchAgentState:
    """Fallback with the same node order when LangGraph is unavailable."""
    for node in [analyze_query_node]:
        state.update(node(state))
    if not state.get("is_medical", True):
        return state

    for node in [create_plan_node, set_periods_node, fetch_current_node, inspect_search_quality_node]:
        state.update(node(state))

    route = route_after_quality(state)
    if route == "refine_search":
        state.update(refine_search_node(state))
        state.update(fetch_current_node(state))
        state.update(inspect_search_quality_node(state))
    if state.get("search_quality") and state["search_quality"].status == "insufficient":
        return state

    for node in [
        topic_model_current_node,
        fetch_previous_node,
        topic_model_previous_node,
        compare_periods_node,
        generate_followups_node,
        generate_recommendation_node,
    ]:
        state.update(node(state))
    return state


def run_research_agent(
    query: str,
    client: OpenAI,
    model: str,
    embedding_model: SentenceTransformer,
    audience: str = "General public",
) -> ResearchAgentResult:
    initial_state: ResearchAgentState = {
        "query": query,
        "audience": audience,
        "client": client,
        "model": model,
        "embedding_model": embedding_model,
        "agent_steps": [],
        "search_refinements": 0,
    }

    graph = build_research_graph()
    final_state = graph.invoke(initial_state) if graph is not None else _run_linear_fallback(initial_state)

    return ResearchAgentResult(
        query=query,
        analysis=final_state.get("analysis"),
        is_medical=final_state.get("is_medical", True),
        message=final_state.get("message"),
        agent_steps=final_state.get("agent_steps", []),
        plan=final_state.get("plan"),
        search_quality=final_state.get("search_quality"),
        keywords_used=final_state.get("keywords", []),
        current_summaries=final_state.get("current_summaries"),
        current_assignments=final_state.get("current_assignments", []),
        current_docs=final_state.get("current_with_abstract", []),
        previous_summaries=final_state.get("previous_summaries"),
        previous_assignments=final_state.get("previous_assignments", []),
        previous_docs=final_state.get("previous_with_abstract", []),
        comparison=final_state.get("comparison"),
        current_label=final_state.get("current_label", ""),
        previous_label=final_state.get("previous_label", ""),
        followups=final_state.get("followups", []),
        recommendation=final_state.get("recommendation"),
    )
