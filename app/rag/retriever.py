"""
Gives every node access to the same vector store retriever without
re-loading Chroma from disk on every call.
"""

import os
from langchain_chroma import Chroma
from app.config import EMBEDDING_MODEL, VECTOR_STORE_DIR

# Module-level cache: Chroma reads its index from disk, which is slow-ish.
# We only want to pay that cost once per process, not once per node call.
# Starts as None; get_retriever() fills it in the first time it's called.
_vector_store = None


def get_retriever(k: int = 4):
    """
    Returns a LangChain retriever backed by the local Chroma vector store.

    k: how many policy clause chunks to pull back per query. 4 is a
       reasonable default for a single applicant query — enough context
       for the reasoning step without flooding the prompt with irrelevant
       clauses.
    """
    global _vector_store  # we're reassigning the module-level cache, not a local copy

    if _vector_store is None:
        if not os.path.isdir(VECTOR_STORE_DIR):
            # On a fresh deployment (e.g. Streamlit Cloud), the vector store
            # won't exist yet since it's gitignored -- build it on the fly
            # the first time it's needed, rather than failing outright.
            from app.rag.ingest import ingest
            ingest()

        # Re-open the same persisted Chroma collection that ingest.py wrote to.
        # embedding_function must match what was used at ingest time (the same
        # local sentence-transformers model), or similarity search breaks
        # silently — vectors from two different models aren't comparable.
        _vector_store = Chroma(
            persist_directory=VECTOR_STORE_DIR,
            embedding_function=EMBEDDING_MODEL,
            collection_name="credit_policy",
        )

    # .as_retriever() wraps the vector store in LangChain's standard retriever
    # interface, so nodes can just call retriever.invoke(query) without
    # knowing anything about Chroma specifically.
    return _vector_store.as_retriever(search_kwargs={"k": k})
