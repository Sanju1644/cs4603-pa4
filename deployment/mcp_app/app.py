"""Bonus C — standalone MCP server as a Databricks App (deployment/mcp_app/app.py).

Reuses the exact same tool definitions from tools/mcp_server.py (the GIVEN
stdio server) but serves them over HTTP instead, so the tool server can be
deployed, scaled, and monitored independently of the model (see PA4 spec,
Bonus C, and the comparison diagram in DEPLOYMENT_GUIDE.md-style tradeoffs).

Run standalone for a local smoke test:

    uv run python deployment/mcp_app/app.py

On Databricks Apps, the platform sets $DATABRICKS_APP_PORT and starts this
via the command in app.yaml.
"""

from __future__ import annotations

import os
import sys

# tools/mcp_server.py lives two directories up from this file
# (deployment/mcp_app/app.py -> deployment/mcp_app -> deployment -> root -> tools).
_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from tools.mcp_server import mcp  # noqa: E402  — reuse the GIVEN tool definitions

if __name__ == "__main__":
    # streamable-http is the transport the agent's MultiServerMCPClient expects
    # when MCP_SERVER_URL is set (see agent/graph.py's load_mcp_tools()).
    # Databricks Apps provides the port via $DATABRICKS_APP_PORT; FastMCP reads
    # host/port from its own settings, so we pass them through explicitly.
    port = int(os.environ.get("DATABRICKS_APP_PORT", "8000"))
    mcp.run(transport="streamable-http", host="0.0.0.0", port=port)