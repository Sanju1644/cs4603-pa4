"""Planner node (Task 1.2).

TODO: Implement `make_planner(llm)` returning a node that:
  - reads the user question from state["messages"],
  - asks the LLM (PLANNER_PROMPT) for a JSON list of 2-5 steps,
  - parses it robustly (fallback to a single step on parse failure),
  - returns {"plan": [...], "current_step_index": 0, "step_results": []}.
"""

from __future__ import annotations

import json
import re

from langchain_core.messages import HumanMessage, SystemMessage

from agent.prompts import PLANNER_PROMPT
from agent.state import AnalystState


def _get_question(state: AnalystState) -> str:
    """Pull the most recent human question out of state["messages"].

    Messages may arrive either as LangChain message objects or as plain
    dicts (e.g. {"role": "user", "content": "..."}), depending on how the
    graph was invoked, so we handle both.
    """
    for msg in reversed(state["messages"]):
        if isinstance(msg, HumanMessage):
            return msg.content
        if isinstance(msg, dict) and msg.get("role") == "user":
            return msg["content"]
    # Fallback: last message, whatever it is.
    last = state["messages"][-1]
    return last.content if hasattr(last, "content") else str(last)


def _parse_plan(raw: str, question: str) -> list[str]:
    """Robustly parse the planner LLM output into a list of step strings.

    Falls back to a single-step plan (the original question) if the model's
    output isn't valid JSON — this keeps the graph running end-to-end even
    when the LLM misbehaves, rather than crashing the whole request.
    """
    text = raw.strip()

    # Strip markdown code fences if the model added them despite instructions.
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*", "", text)
        text = re.sub(r"\s*```$", "", text)
        text = text.strip()

    try:
        parsed = json.loads(text)
    except json.JSONDecodeError:
        # Last resort: try to grab the first [...] substring in the output.
        match = re.search(r"\[.*\]", text, re.DOTALL)
        if match:
            try:
                parsed = json.loads(match.group(0))
            except json.JSONDecodeError:
                parsed = None
        else:
            parsed = None

    if isinstance(parsed, list) and all(isinstance(s, str) for s in parsed) and parsed:
        return parsed

    # Fallback: treat the whole question as a single step.
    return [question]


def make_planner(llm):
    def planner(state: AnalystState) -> dict:
        question = _get_question(state)

        response = llm.invoke(
            [
                SystemMessage(content=PLANNER_PROMPT),
                HumanMessage(content=question),
            ]
        )

        plan = _parse_plan(response.content, question)

        return {
            "plan": plan,
            "current_step_index": 0,
            "step_results": [],
        }

    return planner