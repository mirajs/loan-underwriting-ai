import os
from langchain_chroma import Chroma
from app.config import EMBEDDING_MODEL, VECTOR_STORE_DIR

_vector_store = None


def get_retriever(k: int = 4):
    global _vector_store
    if _vector_store is None:
        if not os.path.isdir(VECTOR_STORE_DIR):
            raise RuntimeError(
                "Vector store not found. Run `python app/rag/ingest.py` first."
            )
        _vector_store = Chroma(
            persist_directory=VECTOR_STORE_DIR,
            embedding_function=EMBEDDING_MODEL,
            collection_name="credit_policy",
        )
    return _vector_store.as_retriever(search_kwargs={"k": k})
