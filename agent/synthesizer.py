"""Synthesizer node (Task 1.6).

TODO: Implement `make_synthesizer(llm)` returning a node that combines
step_results into one cited answer and writes it to BOTH `final_answer` AND
the `messages` channel as an AIMessage (required for the OpenAI-compatible
serving contract — see spec Task 1.6).
"""

from __future__ import annotations

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

from agent.prompts import SYNTHESIZER_PROMPT
from agent.state import AnalystState


def _get_question(state: AnalystState) -> str:
    """Pull the original user question out of state["messages"].

    Mirrors agent/planner.py's helper — kept local here (rather than shared)
    since each node's message-parsing needs may diverge later (e.g. if we
    ever need to distinguish the *original* question from follow-ups in a
    multi-turn conversation).
    """
    for msg in reversed(state["messages"]):
        if isinstance(msg, HumanMessage):
            return msg.content
        if isinstance(msg, dict) and msg.get("role") == "user":
            return msg["content"]
    last = state["messages"][-1]
    return last.content if hasattr(last, "content") else str(last)


def _format_step_results(plan: list[str], step_results: list[str]) -> str:
    """Number each step alongside its result for the synthesizer prompt.

    Numbering lets the LLM explicitly reference "step 2" style provenance
    per Task 1.6's requirement, and makes partial failures (a step that came
    back "not found in documents") visible rather than silently dropped.
    """
    lines = []
    for i, result in enumerate(step_results, start=1):
        step_text = plan[i - 1] if i - 1 < len(plan) else "(unknown step)"
        lines.append(f"Step {i}: {step_text}\nResult: {result}")
    return "\n\n".join(lines)


def make_synthesizer(llm):
    def synthesizer(state: AnalystState) -> dict:
        question = _get_question(state)
        plan = state["plan"]
        step_results = state["step_results"]

        formatted_results = _format_step_results(plan, step_results)

        response = llm.invoke(
            [
                SystemMessage(content=SYNTHESIZER_PROMPT),
                HumanMessage(
                    content=(
                        f"Original question: {question}\n\n"
                        f"Step results:\n{formatted_results}"
                    )
                ),
            ]
        )

        final_answer = response.content.strip()

        return {
            "final_answer": final_answer,
            "messages": [AIMessage(content=final_answer)],
        }

    return synthesizer