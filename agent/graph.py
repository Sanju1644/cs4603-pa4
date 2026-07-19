"""Full Document Analyst graph (Tasks 1.5 + 1.7).

TODO:
  - `load_mcp_tools(server_path=None)`: connect the GIVEN MCP server over stdio
    (see langchain-mcp-adapters) and return its tools.
  - `make_mcp_node(tools, llm)`: execute one calculation step by letting the LLM
    call exactly one MCP tool, then append the result and increment the index.
  - `build_graph(llm=None, retriever=None, tools=None)`: assemble
    planner -> supervisor -> {rag_agent | mcp_tools} -> ... -> synthesizer.
    Inject dependencies so the graph can be unit-tested offline with fakes.
"""

from __future__ import annotations

import asyncio
import os
import threading

from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.graph import END, START, StateGraph

from agent.planner import make_planner
from agent.prompts import MCP_STEP_PROMPT
from agent.rag_agent import make_rag_agent
from agent.state import AnalystState
from agent.supervisor import MCP, RAG, SYNTH, make_supervisor, route_from_supervisor
from agent.synthesizer import make_synthesizer


def _default_server_path() -> str:
    """Resolve tools/mcp_server.py relative to the project root.

    agent/graph.py lives at <root>/agent/graph.py, so the project root is one
    directory up from this file's directory.
    """
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    return os.path.join(root, "tools", "mcp_server.py")


def _has_real_fileno(stream) -> bool:
    """Check whether a stream is backed by a real OS file descriptor."""
    try:
        stream.fileno()
        return True
    except Exception:
        return False


# Module-level handle kept alive for the life of the process — needed because
# mcp's stdio_client() binds errlog=sys.stderr as a *default parameter value*,
# evaluated once at function-definition time (i.e. at import time). Both
# Jupyter's ipykernel (OutStream) and Databricks' serving container
# (StreamToLogger) replace sys.stderr with objects that lack a real
# fileno(), which breaks subprocess creation both locally and in the
# container. Falling back to sys.__stderr__ isn't reliable either — it can
# be None or itself lack a fileno() in a sandboxed container. Opening a
# dedicated handle to os.devnull guarantees a real, valid file descriptor
# everywhere.
_devnull_stderr = None


def _get_fileno_backed_stream():
    global _devnull_stderr
    if _devnull_stderr is None:
        _devnull_stderr = open(os.devnull, "w")
    return _devnull_stderr


def _run_async(coro):
    """Run an async coroutine from sync code, regardless of whether a loop is
    already running in the calling context.

    Jupyter/IPython kernels (and some serving runtimes) always run their own
    event loop, so both asyncio.run() and loop.run_until_complete() fail with
    "cannot be called from a running event loop" when invoked directly.
    Running the coroutine in a dedicated background thread (with its own
    fresh event loop) sidesteps this: the new thread has no pre-existing loop
    to conflict with, and we simply block the calling thread until it
    finishes.

    Note: this does NOT need to patch sys.stderr — that fix must happen at
    *import time*, before mcp/langchain_mcp_adapters is first imported (see
    load_mcp_tools), since stdio_client()'s errlog default is bound once,
    at function-definition time, not per call.
    """
    result_box: dict = {}
    error_box: dict = {}

    def runner():
        try:
            result_box["value"] = asyncio.run(coro)
        except Exception as exc:  # noqa: BLE001 - re-raised on the calling thread below
            error_box["error"] = exc

    thread = threading.Thread(target=runner)
    thread.start()
    thread.join()

    if "error" in error_box:
        raise error_box["error"]
    return result_box["value"]


def load_mcp_tools(server_path: str | None = None):
    """Connect to the MCP server and return its tools as LangChain tools.

    Loaded once at graph-build time (see build_graph) rather than per-request,
    per the caveat in DEPLOYMENT_GUIDE.md and spec Task 1.5.

    If MCP_SERVER_URL is set, connects over streamable HTTP instead (Bonus C
    groundwork). Otherwise spawns the given/default server script over stdio.
    """
    import mcp.client.stdio as _stdio_mod
    from langchain_mcp_adapters.client import MultiServerMCPClient

    # CRITICAL: mcp's stdio_client() binds errlog=sys.stderr as a *default
    # parameter value*, evaluated once when mcp.client.stdio is first
    # imported anywhere in the process — not per call. In Databricks model
    # serving, the harness pre-imports packages from pip_requirements
    # (including mcp/langchain-mcp-adapters) during its own env validation,
    # BEFORE build_graph()/load_mcp_tools() ever runs. By then errlog is
    # already bound to sys.stderr, which serving replaces with a
    # StreamToLogger object that has no fileno() — breaking subprocess
    # creation. Patching sys.stderr here is too late, since the default was
    # already captured. Instead, rebind the bad default directly on the
    # function object, which works regardless of import order.
    #
    # stdio_client is decorated with @asynccontextmanager, and
    # functools.wraps does NOT copy __defaults__ — so the *outer* wrapper's
    # __defaults__ is None even though the real default lives on
    # stdio_client.__wrapped__. Target that instead, and skip gracefully if
    # this mcp version has no default (or no errlog) to patch at all.
    _target = getattr(_stdio_mod.stdio_client, "__wrapped__", _stdio_mod.stdio_client)

    if _target.__defaults__:
        _target.__defaults__ = tuple(
            _get_fileno_backed_stream() if not _has_real_fileno(d) else d
            for d in _target.__defaults__
        )

    mcp_url = os.environ.get("MCP_SERVER_URL")

    if mcp_url:
        token = os.environ.get("DATABRICKS_TOKEN", "")
        connections = {
            "analyst": {
                "url": f"{mcp_url}/mcp",
                "transport": "streamable_http",
                "headers": {"Authorization": f"Bearer {token}"} if token else {},
            }
        }
    else:
        path = server_path or _default_server_path()
        connections = {
            "analyst": {
                "command": "python",
                "args": [path],
                "transport": "stdio",
            }
        }

    client = MultiServerMCPClient(connections)
    return _run_async(client.get_tools())


def make_mcp_node(tools, llm):
    tools_by_name = {t.name: t for t in tools}
    llm_with_tools = llm.bind_tools(tools)

    def mcp_tools(state: AnalystState) -> dict:
        plan = state["plan"]
        idx = state["current_step_index"]
        step = plan[idx]
        prior_results = state["step_results"]

        context = "\n".join(
            f"Step {i + 1} result: {r}" for i, r in enumerate(prior_results)
        )

        response = llm_with_tools.invoke(
            [
                SystemMessage(content=MCP_STEP_PROMPT),
                HumanMessage(
                    content=(
                        f"Step: {step}\n\n"
                        f"Known facts so far:\n{context if context else '(none yet)'}"
                    )
                ),
            ]
        )

        tool_calls = getattr(response, "tool_calls", None) or []

        if not tool_calls:
            result = f"Step '{step}': no tool call was made ({response.content.strip()})"
        else:
            call = tool_calls[0]
            tool = tools_by_name.get(call["name"])

            if tool is None:
                result = f"Step '{step}': model requested unknown tool '{call['name']}'"
            else:
                try:
                    tool_output = _run_async(tool.ainvoke(call["args"]))
                except Exception as exc:  # noqa: BLE001 - surface any tool failure as a step result
                    tool_output = f"Error calling {call['name']}: {exc}"

                result = f"Step '{step}': {tool_output}"

        step_results = prior_results + [result]

        return {
            "step_results": step_results,
            "current_step_index": idx + 1,
        }

    return mcp_tools


def build_graph(llm=None, retriever=None, tools=None):
    """Assemble and compile the full Document Analyst graph.

    All three dependencies are injected so this can be unit-tested offline
    with fakes (see tests/test_smoke.py) without touching Databricks or
    spawning the MCP subprocess. Defaults are only constructed if the caller
    doesn't supply them, keeping import-time side effects at zero.
    """
    if llm is None:
        from config import get_chat_llm

        llm = get_chat_llm()

    if retriever is None:
        from rag.store import get_retriever

        retriever = get_retriever()

    if tools is None:
        tools = load_mcp_tools()

    planner = make_planner(llm)
    supervisor = make_supervisor(llm)
    rag_agent = make_rag_agent(retriever, llm)
    mcp_tools_node = make_mcp_node(tools, llm)
    synthesizer = make_synthesizer(llm)

    builder = StateGraph(AnalystState)
    builder.add_node("planner", planner)
    builder.add_node("supervisor", supervisor)
    builder.add_node("rag_agent", rag_agent)
    builder.add_node("mcp_tools", mcp_tools_node)
    builder.add_node("synthesizer", synthesizer)

    builder.add_edge(START, "planner")
    builder.add_edge("planner", "supervisor")
    builder.add_conditional_edges(
        "supervisor",
        route_from_supervisor,
        {RAG: "rag_agent", MCP: "mcp_tools", SYNTH: "synthesizer"},
    )
    builder.add_edge("rag_agent", "supervisor")
    builder.add_edge("mcp_tools", "supervisor")
    builder.add_edge("synthesizer", END)

    return builder.compile()