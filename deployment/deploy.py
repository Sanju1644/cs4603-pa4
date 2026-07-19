"""Log, register, and serve the Document Analyst (Tasks 2.2 + 2.3).

Run:  uv run python deployment/deploy.py

TODO:
  - `log_and_register()`: set registry uri to 'databricks-uc', log the model via
    `mlflow.langchain.log_model(lc_model="deployment/agent_model.py", name=...,
    code_paths=[...], pip_requirements=[...], input_example={...})`, then
    `mlflow.register_model(...)` into $UC_CATALOG.$UC_SCHEMA.<model>.
  - `create_or_update_endpoint(uc_name, version)`: create/update a Model Serving
    endpoint with `WorkspaceClient().serving_endpoints`, workload_size='Small',
    scale_to_zero_enabled=True, and environment_vars supplied as secret refs
    ({{secrets/cs4603-deploy/...}}). Wait for READY and print the URL.
"""

from __future__ import annotations

import os
import time

import mlflow
from dotenv import load_dotenv

load_dotenv()

# ─── Fixed configuration for this deployment ────────────────────────────────

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_NAME = os.environ.get("PA4_MODEL_NAME", "pa4_document_analyst")
ENDPOINT_NAME = os.environ.get("PA4_ENDPOINT_NAME", "pa4-document-analyst")
SECRET_SCOPE = "cs4603-deploy"

PIP_REQUIREMENTS = [
    "mlflow>=2.16.0",
    "langgraph>=0.2.0",
    "langchain>=0.3.0",
    "langchain-core>=0.3.0",
    "langchain-openai>=0.2.0",
    "databricks-langchain>=0.1.0",
    "databricks-vectorsearch>=0.40",
    "databricks-sdk>=0.23.0",
    "mcp>=1.0.0",
    "langchain-mcp-adapters>=0.0.5",
    "openai>=1.40.0",
    "python-dotenv>=1.0.0",
    "httpx>=0.27.0",
]


def log_and_register():
    """Log the model via models-from-code and register it in Unity Catalog."""
    catalog = os.environ["UC_CATALOG"]
    schema = os.environ["UC_SCHEMA"]
    uc_name = f"{catalog}.{schema}.{MODEL_NAME}"

    mlflow.set_tracking_uri("databricks")
    mlflow.set_registry_uri("databricks-uc")
    mlflow.set_experiment(f"/Shared/{MODEL_NAME}")

    with mlflow.start_run():
        model_info = mlflow.langchain.log_model(
            lc_model="deployment/agent_model.py",   # relative, not ROOT-based
            name="agent",
            code_paths=[                             # relative, not ROOT-based
                "agent",
                "rag",
                "tools",
                "config.py",
            ],
            pip_requirements=PIP_REQUIREMENTS,
            input_example={
                "messages": [{"role": "user", "content": "What was the revenue?"}]
            },
        )

    registered = mlflow.register_model(model_info.model_uri, uc_name)
    print(f"Registered model: {uc_name} version {registered.version}")

    return uc_name, registered.version


def create_or_update_endpoint(uc_name: str, version: str) -> str:
    """Create the serving endpoint if it doesn't exist, else update it to the
    given model version. Waits for READY and returns the endpoint URL.
    """
    from databricks.sdk import WorkspaceClient
    from databricks.sdk.service.serving import (
        EndpointCoreConfigInput,
        ServedEntityInput,
    )

    client = WorkspaceClient()

    served_entities = [
        ServedEntityInput(
            entity_name=uc_name,
            entity_version=version,
            workload_size="Small",
            scale_to_zero_enabled=True,
            environment_vars={
                "DATABRICKS_HOST": f"{{{{secrets/{SECRET_SCOPE}/DATABRICKS_HOST}}}}",
                "DATABRICKS_TOKEN": f"{{{{secrets/{SECRET_SCOPE}/DATABRICKS_TOKEN}}}}",
                "DATABRICKS_MODEL": f"{{{{secrets/{SECRET_SCOPE}/DATABRICKS_MODEL}}}}",
                # Not secrets — the retriever needs these to reach the index.
                "VECTOR_SEARCH_ENDPOINT": os.environ["VECTOR_SEARCH_ENDPOINT"],
                "VECTOR_SEARCH_INDEX": os.environ["VECTOR_SEARCH_INDEX"],
                "EMBEDDINGS_ENDPOINT": os.environ.get(
                    "EMBEDDINGS_ENDPOINT", "databricks-gte-large-en"
                ),
            },
        )
    ]

    existing = None
    try:
        existing = client.serving_endpoints.get(ENDPOINT_NAME)
    except Exception:
        existing = None

    if existing is None:
        client.serving_endpoints.create(
            name=ENDPOINT_NAME,
            config=EndpointCoreConfigInput(
                name=ENDPOINT_NAME,
                served_entities=served_entities
            ),
        )
        print(f"Creating endpoint '{ENDPOINT_NAME}'...")
    else:
        client.serving_endpoints.update_config(
            name=ENDPOINT_NAME,
            served_entities=served_entities,
        )
        print(f"Updating endpoint '{ENDPOINT_NAME}' to version {version}...")

    # Poll for READY.
    timeout_seconds = 900
    poll_interval = 15
    elapsed = 0
    status = client.serving_endpoints.get(ENDPOINT_NAME)

    while status.state.ready.value != "READY" and elapsed < timeout_seconds:
        time.sleep(poll_interval)
        elapsed += poll_interval
        status = client.serving_endpoints.get(ENDPOINT_NAME)
        print(f"  ...waiting ({elapsed}s elapsed), state={status.state.ready}")

    if status.state.ready.value != "READY":
        raise TimeoutError(
            f"Endpoint '{ENDPOINT_NAME}' did not reach READY within "
            f"{timeout_seconds}s (last state: {status.state.ready})"
        )

    host = os.environ["DATABRICKS_HOST"].rstrip("/")
    url = f"{host}/serving-endpoints/{ENDPOINT_NAME}/invocations"
    print(f"Endpoint READY: {ENDPOINT_NAME}")
    print(f"Invocation URL: {url}")
    return url


if __name__ == "__main__":
    name, ver = log_and_register()
    create_or_update_endpoint(name, ver)