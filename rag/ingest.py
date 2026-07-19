"""Corpus ingestion into Databricks Vector Search (Task 0.3 / rag/ingest.py).

Run inside a Databricks notebook (needs Spark + ai_parse_document/ai_prep_search).
Mirror PA2 Part 1:

TODO:
  - `build_chunks_table(spark, volume_path, chunks_table)`: parse the PDF with
    ai_parse_document, chunk with ai_prep_search into a Delta table with columns
    chunk_id, chunk_to_retrieve, chunk_to_embed, source, page. Enable Change Data
    Feed on the table.
  - `create_index()`: create a STANDARD Vector Search endpoint and a TRIGGERED
    Delta Sync index (primary_key='chunk_id',
    embedding_source_column='chunk_to_retrieve',
    embedding_model_endpoint_name=$EMBEDDINGS_ENDPOINT).
"""

from __future__ import annotations

from config import get_settings


def build_chunks_table(spark, volume_path: str, chunks_table: str) -> None:
    """Parse `volume_path` and chunk it into the Delta table `chunks_table`.

    Steps (mirrors PA2 Part 1):
      1. `ai_parse_document` reads the PDF from the UC volume and produces
         structured parsed content (text blocks with page numbers).
      2. `ai_prep_search` chunks that parsed content into retrieval-sized
         pieces, producing `chunk_to_retrieve` (the text used for embedding /
         similarity search) and `chunk_to_embed` (kept distinct in case you
         later want a different embedding-vs-display text; here they're the
         same content, matching Task 0.3's schema).
      3. We attach `source` (the original filename) and `page` (page number)
         as metadata columns so RAG results can be cited exactly like
         "[source: annual_report.pdf, p.4]" (see Analysis.md's example).
      4. Change Data Feed is enabled on the table so the Vector Search
         Delta Sync index (TRIGGERED pipeline) can detect new/changed rows.

    Args:
        spark: the active SparkSession (provided by the Databricks notebook).
        volume_path: UC volume path to the source PDF,
            e.g. "/Volumes/main/default/pa4/annual_report.pdf".
        chunks_table: fully-qualified Delta table name to write chunks into,
            e.g. "main.default.ali_analyst_chunks".
    """
    source_name = volume_path.rsplit("/", 1)[-1]

    # Step 1: parse the document with Databricks' AI Functions SQL, which
    # returns one row per parsed page/block with its page number and text.
    parsed_df = spark.sql(
        """
        SELECT
            parsed.page AS page,
            parsed.content AS content
        FROM (
            SELECT explode(ai_parse_document(:volume_path).pages) AS parsed
        )
        """,
        {"volume_path": volume_path},
    )
    parsed_df.createOrReplaceTempView("pa4_parsed_pages")

    # Step 2: chunk each page's content with ai_prep_search. This UDF splits
    # long text into retrieval-sized chunks, and we assign a deterministic
    # chunk_id (source + page + chunk index within the page) as the primary
    # key required by the Vector Search Delta Sync index.
    chunks_df = spark.sql(
        """
        SELECT
            concat_ws('-', :source_name, cast(page as string), cast(pos as string)) AS chunk_id,
            chunk AS chunk_to_retrieve,
            chunk AS chunk_to_embed,
            :source_name AS source,
            page AS page
        FROM (
            SELECT
                page,
                posexplode(ai_prep_search(content)) AS (pos, chunk)
            FROM pa4_parsed_pages
        )
        """,
        {"source_name": source_name},
    )

    # Step 3: write to the Delta table, then enable Change Data Feed so the
    # TRIGGERED Vector Search sync can detect inserts/updates incrementally.
    chunks_df.write.mode("overwrite").saveAsTable(chunks_table)
    spark.sql(
        f"ALTER TABLE {chunks_table} SET TBLPROPERTIES (delta.enableChangeDataFeed = true)"
    )


def create_index() -> None:
    """Create the Vector Search endpoint and Delta Sync index for PA4.

    Reads endpoint/index names and the embeddings model from
    config.get_settings() so this matches whatever `.env` / secret-scope
    values the rest of the project (rag/store.py, deployment) uses.

    - Endpoint type STANDARD (per Task 0.3 step 3).
    - Index is a Delta Sync index (`pipeline_type="TRIGGERED"`) with managed
      embeddings: Vector Search computes embeddings itself from
      `embedding_source_column` using `embedding_model_endpoint_name`, so we
      never have to embed text ourselves in Python.
    """
    from databricks.vector_search.client import VectorSearchClient

    settings = get_settings()
    catalog_schema_table = settings.get("chunks_table") or _default_chunks_table()

    client = VectorSearchClient()

    # Create the endpoint if it doesn't already exist.
    existing_endpoints = {ep["name"] for ep in client.list_endpoints().get("endpoints", [])}
    if settings["vs_endpoint"] not in existing_endpoints:
        client.create_endpoint(name=settings["vs_endpoint"], endpoint_type="STANDARD")

    # Create the Delta Sync index with managed (server-side) embeddings.
    client.create_delta_sync_index(
        endpoint_name=settings["vs_endpoint"],
        index_name=settings["vs_index"],
        source_table_name=catalog_schema_table,
        pipeline_type="TRIGGERED",
        primary_key="chunk_id",
        embedding_source_column="chunk_to_retrieve",
        embedding_model_endpoint_name=settings["embeddings"],
    )


def _default_chunks_table() -> str:
    """Fallback fully-qualified chunks table name if not present in settings."""
    import os

    catalog = os.environ.get("UC_CATALOG", "main")
    schema = os.environ.get("UC_SCHEMA", "default")
    return f"{catalog}.{schema}.pa4_analyst_chunks"