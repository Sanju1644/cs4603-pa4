"""RAG agent node (Task 1.4) — retrieves from Databricks Vector Search.

TODO: Implement `make_rag_agent(retriever, llm)` returning a node that:
  - retrieves top-k chunks for the current step,
  - formats them with [source: file, p.N] citations,
  - extracts a single cited fact via the LLM (or 'not found in documents'),
  - appends the fact to step_results and increments current_step_index.
Reuse `rag/store.py::get_retriever()` so local and deployed retrieval match.
"""

from __future__ import annotations

from langchain_core.messages import HumanMessage, SystemMessage

from agent.prompts import RAG_EXTRACT_PROMPT
from agent.state import AnalystState

NOT_FOUND_SENTINEL = "NOT_FOUND_IN_DOCUMENT"


def format_docs(docs) -> str:
    """Format retrieved chunks with [source: file, p.N] style citations.

    Each chunk is numbered so the extraction prompt can refer to "chunk 2"
    etc. if needed, and so we can tell at a glance in logs/notebooks which
    chunk contributed which fact.
    """
    if not docs:
        return ""

    blocks = []
    for i, doc in enumerate(docs, start=1):
        source = doc.metadata.get("source", "unknown")
        page = doc.metadata.get("page", "?")
        blocks.append(
            f"[Chunk {i} | source: {source}, p.{page}]\n{doc.page_content}"
        )
    return "\n\n".join(blocks)


def make_rag_agent(retriever, llm):
    def rag_agent(state: AnalystState) -> dict:
        plan = state["plan"]
        idx = state["current_step_index"]
        step = plan[idx]

        docs = retriever.invoke(step)
        formatted = format_docs(docs)

        if not formatted:
            # No chunks at all — don't bother calling the LLM, we already
            # know the answer.
            result = f"Step '{step}': not found in documents (no chunks retrieved)"
        else:
            response = llm.invoke(
                [
                    SystemMessage(content=RAG_EXTRACT_PROMPT),
                    HumanMessage(
                        content=(
                            f"Step: {step}\n\n"
                            f"Retrieved chunks:\n{formatted}"
                        )
                    ),
                ]
            )
            extracted = response.content.strip()

            if extracted == NOT_FOUND_SENTINEL:
                result = f"Step '{step}': not found in documents"
            else:
                result = extracted

        step_results = state["step_results"] + [result]

        return {
            "step_results": step_results,
            "current_step_index": idx + 1,
        }

    return rag_agent