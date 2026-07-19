"""MLflow models-from-code definition (Task 2.1).

TODO: Make this file self-contained so MLflow can serialise it:
  - validate DATABRICKS_HOST/TOKEN/MODEL at import time (clear error if missing),
  - rebuild the graph with production clients (LLM, Vector Search retriever,
    MCP tools),
  - end with `mlflow.models.set_model(graph)`.

Must import cleanly:  python -c "import deployment.agent_model"
"""

from __future__ import annotations

import os

import mlflow

from agent.graph import build_graph

# ─── Validate required environment variables up front ───────────────────────
# Fail loudly and specifically here, at import time, rather than letting a
# missing env var surface as an opaque error deep inside build_graph() or,
# worse, only at first-request time in the serving container. This is what
# DEPLOYMENT_GUIDE.md §9 means by "match the traceback" — a clear message
# here saves a debugging loop in the Serving Logs tab.

_REQUIRED_ENV_VARS = [
    "DATABRICKS_HOST",
    "DATABRICKS_TOKEN",
    "DATABRICKS_MODEL",
]

_missing = [name for name in _REQUIRED_ENV_VARS if not os.environ.get(name)]
if _missing:
    raise OSError(
        "deployment/agent_model.py: missing required environment variable(s): "
        f"{', '.join(_missing)}. "
        "Set these in your .env (local) or as endpoint environment_vars / "
        "secret references (deployed) before building the model."
    )

# Vector Search vars aren't strictly required to import this file, but the
# retriever will fail at build_graph() time without them — surface that here
# too so the cause is obvious in Serving Logs rather than a stack trace from
# inside rag/store.py.
_VS_ENV_VARS = ["VECTOR_SEARCH_ENDPOINT", "VECTOR_SEARCH_INDEX"]
_missing_vs = [name for name in _VS_ENV_VARS if not os.environ.get(name)]
if _missing_vs:
    raise OSError(
        "deployment/agent_model.py: missing required Vector Search "
        f"environment variable(s): {', '.join(_missing_vs)}. "
        "Without these, rag/store.py cannot connect to the retrieval index."
    )


# ─── Rebuild the graph with production clients ──────────────────────────────
# build_graph() with no arguments constructs get_chat_llm(), get_retriever(),
# and load_mcp_tools() internally (see agent/graph.py) — exactly the
# production clients this deployed model needs, using the same env vars we
# just validated.

graph = build_graph()

# ─── Tell MLflow what to serve ──────────────────────────────────────────────
mlflow.models.set_model(graph)