"""Offline smoke test for the Document Analyst graph (Bonus A test target).

This is the target the Bonus A CI pipeline runs to prove the graph wires up
before any deploy. Fill it in once your nodes are implemented.

TODO (Task 1.7 / Bonus A):
  - Build fake LLM / retriever / tool objects (no Databricks, no network).
  - Call `build_graph(llm=FakeLLM(), retriever=FakeRetriever(), tools=[FakeTool()])`.
  - Invoke it on a combined retrieval+calculation query and assert that a plan was
    produced, both specialists ran, and the final answer surfaced on messages[-1].

Run:  uv run pytest -q
"""

from __future__ import annotations

import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from agent.prompts import (  # noqa: E402
    PLANNER_PROMPT,
    RAG_EXTRACT_PROMPT,
    SUPERVISOR_PROMPT,
    SYNTHESIZER_PROMPT,
)


def test_graph_module_imports():
    """Minimal collection guard: the graph module must import cleanly."""
    from agent.graph import build_graph  # noqa: F401


# ─── Fakes ───────────────────────────────────────────────────────────────────


class _FakeMessage:
    """Minimal stand-in for an AIMessage: just .content and .tool_calls."""

    def __init__(self, content: str = "", tool_calls=None):
        self.content = content
        self.tool_calls = tool_calls or []


class FakeDoc:
    """Minimal stand-in for a LangChain Document."""

    def __init__(self, page_content: str, metadata: dict):
        self.page_content = page_content
        self.metadata = metadata


class FakeRetriever:
    """Always returns one chunk containing a fake revenue figure."""

    def invoke(self, query: str):
        return [
            FakeDoc(
                page_content="Meridian's net revenue in FY2023 was $100 million.",
                metadata={"source": "annual_report.pdf", "page": 1},
            )
        ]


class FakeTool:
    """A fake 'calculate' tool matching the real MCP tool's name/interface."""

    name = "calculate"

    async def ainvoke(self, args: dict) -> str:
        return "100 * 1.10 = 110"


class _FakeToolBoundLLM:
    """Returned by FakeLLM.bind_tools(); always calls exactly one tool."""

    def __init__(self, tools):
        self._tool_name = tools[0].name if tools else "calculate"

    def invoke(self, messages):
        return _FakeMessage(
            content="",
            tool_calls=[
                {
                    "name": self._tool_name,
                    "args": {"expression": "100 * 1.10"},
                    "id": "call_1",
                }
            ],
        )


class FakeLLM:
    """Routes on the system prompt's content to fake each node's LLM call.

    The same instance is reused for the planner, supervisor, RAG extraction,
    and synthesizer nodes, exactly like the real ChatOpenAI client is in
    build_graph() — so this fake must handle all four prompt types.
    """

    def invoke(self, messages):
        system_content = messages[0].content if messages else ""

        if system_content == PLANNER_PROMPT:
            plan = [
                "Retrieve Meridian's net revenue for fiscal year 2023",
                "Compute a 10% increase on that revenue figure",
            ]
            return _FakeMessage(content=json.dumps(plan))

        if system_content == SUPERVISOR_PROMPT:
            step_text = messages[-1].content.lower()
            if "comput" in step_text or "calculat" in step_text or "increase" in step_text:
                return _FakeMessage(content="mcp_tools")
            return _FakeMessage(content="rag_agent")

        if system_content == RAG_EXTRACT_PROMPT:
            return _FakeMessage(
                content="Revenue was $100 million [source: annual_report.pdf, p.1]"
            )

        if system_content == SYNTHESIZER_PROMPT:
            return _FakeMessage(
                content=(
                    "Meridian's FY2023 revenue was $100 million "
                    "[source: annual_report.pdf, p.1]. A 10% increase would "
                    "bring it to $110 million (100 * 1.10 = 110)."
                )
            )

        return _FakeMessage(content="")

    def bind_tools(self, tools):
        return _FakeToolBoundLLM(tools)


# ─── Test ────────────────────────────────────────────────────────────────────


def test_combined_query_end_to_end():
    """Drive planner -> supervisor -> rag_agent -> supervisor -> mcp_tools ->
    supervisor -> synthesizer with fakes, and check the state contract holds.
    """
    from agent.graph import build_graph

    graph = build_graph(llm=FakeLLM(), retriever=FakeRetriever(), tools=[FakeTool()])

    result = graph.invoke(
        {
            "messages": [
                {
                    "role": "user",
                    "content": (
                        "What was Meridian's net revenue in FY2023, and what "
                        "would it be after a 10% increase?"
                    ),
                }
            ]
        }
    )

    # A plan was produced with more than one step.
    assert len(result["plan"]) >= 2

    # Both specialists ran: one step result should carry a citation (rag_agent),
    # another should carry the tool's arithmetic output (mcp_tools).
    assert len(result["step_results"]) == len(result["plan"])
    joined_results = " ".join(result["step_results"])
    assert "source: annual_report.pdf" in joined_results
    assert "100 * 1.10 = 110" in joined_results

    # The synthesizer wrote to both final_answer and messages[-1] (the
    # OpenAI-compatible serving contract from DEPLOYMENT_GUIDE.md §5).
    assert result["final_answer"]
    last_message = result["messages"][-1]
    last_content = (
        last_message.content if hasattr(last_message, "content") else last_message["content"]
    )
    assert last_content == result["final_answer"]