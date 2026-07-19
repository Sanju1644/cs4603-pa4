"""Supervisor node + routing edge (Task 1.3).

TODO:
  - `make_supervisor(llm)`: if current_step_index >= len(plan) -> next_agent =
    'synthesizer'; else classify the current step to 'rag_agent' or 'mcp_tools'.
  - `route_from_supervisor(state)`: return state["next_agent"] for the
    conditional edge.
"""

from __future__ import annotations

from langchain_core.messages import HumanMessage, SystemMessage

from agent.prompts import SUPERVISOR_PROMPT
from agent.state import AnalystState

RAG = "rag_agent"
MCP = "mcp_tools"
SYNTH = "synthesizer"


def _parse_label(raw: str) -> str:
    """Robustly map the supervisor LLM's output to one of RAG or MCP.

    The prompt asks for a single bare word, but LLMs sometimes wrap it in
    punctuation or a short sentence, so we normalize and substring-match
    rather than requiring an exact match. Defaults to RAG if the output is
    ambiguous or unrecognized, since a wrong retrieval is generally cheaper
    to recover from (the RAG agent can report "not found") than skipping a
    needed calculation.
    """
    text = raw.strip().lower()

    if MCP in text:
        return MCP
    if RAG in text:
        return RAG

    # Fall back to loose keyword hints if the model didn't use our exact labels.
    if any(word in text for word in ("calculat", "comput", "math", "growth", "percent")):
        return MCP

    return RAG


def make_supervisor(llm):
    def supervisor(state: AnalystState) -> dict:
        plan = state["plan"]
        idx = state["current_step_index"]

        if idx >= len(plan):
            return {"next_agent": SYNTH}

        current_step = plan[idx]

        response = llm.invoke(
            [
                SystemMessage(content=SUPERVISOR_PROMPT),
                HumanMessage(content=current_step),
            ]
        )

        label = _parse_label(response.content)

        return {"next_agent": label}

    return supervisor


def route_from_supervisor(state: AnalystState) -> str:
    return state["next_agent"]