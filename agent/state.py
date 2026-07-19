"""State schema for the Document Analyst graph (Task 1.1).

TODO: Define `AnalystState` as a TypedDict with the fields from the spec table:
  messages, plan, current_step_index, step_results, next_agent, final_answer.
Use `Annotated[list, add_messages]` for `messages`.
"""

from __future__ import annotations

from typing import Annotated, TypedDict

from langgraph.graph.message import add_messages


class AnalystState(TypedDict):
    """Shared state threaded through every node in the Document Analyst graph.

    messages:            Chat history; the entry/exit channel for the deployed
                          endpoint (see DEPLOYMENT_GUIDE.md §5 — "messages in,
                          messages out"). Reducer `add_messages` appends rather
                          than overwrites.
    plan:                Ordered list of natural-language sub-steps produced by
                          the Planner (Task 1.2), e.g.
                          ["Retrieve 2023 net income from the annual report",
                           "Compute 15% of that figure"].
    current_step_index:  Index into `plan` of the step the Supervisor is about
                          to dispatch. Incremented after each step completes.
    step_results:         Results collected so far, one entry per completed step,
                          in the same order as `plan`. The Synthesizer reads this
                          list to compose the final answer.
    next_agent:           The Supervisor's routing decision — the name of the
                          node to invoke next (e.g. "rag_agent", "calculator",
                          "synthesizer"). Read by the graph's conditional edge.
    final_answer:         The Synthesizer's finished response text. Also mirrored
                          into `messages` as an AIMessage so the OpenAI-compatible
                          serving endpoint returns it correctly.
    """

    messages: Annotated[list, add_messages]
    plan: list[str]
    current_step_index: int
    step_results: list[str]
    next_agent: str
    final_answer: str