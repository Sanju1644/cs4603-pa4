"""Bonus B — deploy via the databricks-agents SDK (deployment/deploy_agents.py).

TODO: Log + register the model (reuse the pattern from deploy.py), then call
`databricks.agents.deploy(model_name=..., model_version=...)` to provision the
serving endpoint AND the Review App in one call. Print the endpoint + review URL.
"""

from __future__ import annotations

import os

import mlflow

from deployment.deploy import PIP_REQUIREMENTS, ROOT

MODEL_NAME = os.environ.get("PA4_MODEL_NAME", "pa4_document_analyst")


def main() -> None:
    # ─── Log + register (identical pattern to deploy.py's log_and_register) ───
    catalog = os.environ["UC_CATALOG"]
    schema = os.environ["UC_SCHEMA"]
    uc_name = f"{catalog}.{schema}.{MODEL_NAME}"

    mlflow.set_registry_uri("databricks-uc")
    mlflow.set_experiment(f"/Shared/{MODEL_NAME}")

    with mlflow.start_run():
        model_info = mlflow.langchain.log_model(
            lc_model=os.path.join(ROOT, "deployment", "agent_model.py"),
            name="agent",
            code_paths=[
                os.path.join(ROOT, "agent"),
                os.path.join(ROOT, "rag"),
                os.path.join(ROOT, "tools"),
                os.path.join(ROOT, "config.py"),
            ],
            pip_requirements=PIP_REQUIREMENTS,
            input_example={
                "messages": [{"role": "user", "content": "What was the revenue?"}]
            },
        )

    registered = mlflow.register_model(model_info.model_uri, uc_name)
    print(f"Registered model: {uc_name} version {registered.version}")

    # ─── Deploy with the databricks-agents SDK ─────────────────────────────
    # One call provisions both the serving endpoint and a Review App, and
    # handles auth for the endpoint automatically — no WorkspaceClient, no
    # EndpointCoreConfigInput, no manual secret scope references (contrast
    # with deployment/deploy.py's create_or_update_endpoint()).
    from databricks import agents

    deployment = agents.deploy(
        model_name=uc_name,
        model_version=registered.version,
        scale_to_zero=True,
    )

    print(f"Endpoint name: {deployment.endpoint_name}")
    print(f"Review App URL: {deployment.review_app_url}")


if __name__ == "__main__":
    main()