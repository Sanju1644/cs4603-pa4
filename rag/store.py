"""Vector Search retriever factory (Task 1.4 support / rag/store.py).

TODO: Implement `get_retriever(k=4)` that returns a LangChain retriever over the
Databricks Vector Search index built by `ingest.py`, using
`DatabricksVectorSearch` from `databricks_langchain`. Read endpoint/index names
from config.get_settings(). This exact retriever is reused by the deployed model.
"""

from __future__ import annotations

from functools import lru_cache

from config import get_settings

TEXT_COLUMN = "chunk_to_retrieve"
CITATION_COLUMNS = ["chunk_id", "source", "page"]


@lru_cache(maxsize=1)
def get_vector_store():
    """Build a DatabricksVectorSearch handle over the index from Task 0.3.

    Cached with lru_cache so repeated calls (e.g. one per RAG step within a
    single request, or across requests inside the same serving container)
    reuse the same connection rather than re-authenticating each time.

    Reads endpoint/index names from config.get_settings() so the exact same
    code runs locally and inside the deployed model (see DEPLOYMENT_GUIDE.md
    §6) — no separate embedding path for deployment.
    """
    from databricks_langchain import DatabricksVectorSearch

    settings = get_settings()

    if not settings["vs_endpoint"] or not settings["vs_index"]:
        raise OSError(
            "VECTOR_SEARCH_ENDPOINT and VECTOR_SEARCH_INDEX must be set "
            "(in .env locally, or as endpoint environment_vars when deployed) "
            "for rag/store.py to connect to the Vector Search index."
        )

    # NOTE: text_column is intentionally omitted. The index was created with
    # embedding_source_column="chunk_to_retrieve" already configured at
    # index-creation time (Task 0.3), so DatabricksVectorSearch auto-detects
    # the source column from the index itself. Passing text_column explicitly
    # conflicts with that and raises a ValueError.
    return DatabricksVectorSearch(
        endpoint=settings["vs_endpoint"],
        index_name=settings["vs_index"],
        columns=CITATION_COLUMNS,
    )


def get_retriever(k: int = 4):
    """Return a top-k LangChain retriever over the Vector Search index.

    Used both by agent/rag_agent.py at request time and by pa4.ipynb for
    ad-hoc similarity-search testing (Task 0.3 step 4).
    """
    store = get_vector_store()
    return store.as_retriever(search_kwargs={"k": k})