"""
Builds the local Chroma vector store from the credit policy documents in
data/policy_docs/. Run this once (and again whenever policy docs change):

    python app/rag/ingest.py

Uses local sentence-transformers embeddings -- no API calls, no cost,
and policy text never leaves the machine at this stage.
"""

import os
import sys

# This script is meant to be runnable directly (`python app/rag/ingest.py`),
# not just imported. When run directly, Python doesn't automatically know
# where the project root is, so `from app.config import ...` below would
# fail with a ModuleNotFoundError. This line manually adds the project root
# (two levels up from app/rag/) to Python's import search path, fixing that.
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import TextLoader
from langchain_chroma import Chroma
from app.config import EMBEDDING_MODEL, VECTOR_STORE_DIR, POLICY_DOCS_DIR


def ingest():
    """
    One-time (or "run whenever policy docs change") setup step: reads every
    policy .md file, splits it into small chunks, embeds each chunk, and
    saves the result to disk as a Chroma index. policy_rag_node then just
    reads this pre-built index at query time — nothing here runs during a
    normal underwriting request, which is why retrieval is fast.
    """
    if not os.path.isdir(POLICY_DOCS_DIR):
        raise FileNotFoundError(f"No policy docs directory at {POLICY_DOCS_DIR}")

    docs = []
    # sorted() makes the ingestion order deterministic — otherwise clause_id
    # numbering below could differ between runs on different machines/OSes,
    # which would make it harder to reference a specific clause consistently.
    for fname in sorted(os.listdir(POLICY_DOCS_DIR)):
        if not fname.endswith(".md"):
            continue
        path = os.path.join(POLICY_DOCS_DIR, fname)
        # TextLoader reads the whole file as one Document. .load() returns a
        # list (some loaders split into multiple documents per file), but
        # for a single .md file this list always has exactly one item.
        loaded = TextLoader(path, encoding="utf-8").load()
        for d in loaded:
            # Stamp the filename onto the document's metadata now, before
            # splitting, so every chunk derived from it inherits this tag —
            # this is what lets retrieved clauses show which file they came from.
            d.metadata["source_doc"] = fname
        docs.extend(loaded)

    if not docs:
        raise FileNotFoundError(f"No .md policy documents found in {POLICY_DOCS_DIR}")

    # Splits each full document into ~800-character chunks (with 120
    # characters of overlap between consecutive chunks, so a sentence that
    # falls right on a chunk boundary isn't cut in a way that loses meaning).
    # The `separators` list tells the splitter to prefer breaking at
    # markdown headers first, then paragraph breaks, then line breaks, only
    # falling back to splitting mid-sentence as a last resort — this keeps
    # each chunk as one coherent policy clause rather than an arbitrary
    # slice of text.
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=120,
        separators=["\n## ", "\n### ", "\n\n", "\n", " "],
    )
    chunks = splitter.split_documents(docs)

    # Assigns each chunk a stable, human-readable ID like
    # "personal_loan_policy.md-3" — this is the clause_id that shows up in
    # retrieved_clauses and, eventually, in the reasoning node's citations.
    for i, c in enumerate(chunks):
        c.metadata["clause_id"] = f"{c.metadata.get('source_doc', 'doc')}-{i}"

    print(f"Ingesting {len(chunks)} chunks from {len(docs)} document(s)...")

    # This single call does the actual embedding work: for each chunk, it
    # runs EMBEDDING_MODEL over the text to get a vector, then writes both
    # the text and its vector to disk at VECTOR_STORE_DIR. This is the slow
    # step (especially the first time, while the embedding model downloads)
    # — expect it to take anywhere from a few seconds to a minute depending
    # on how much policy text you have.
    Chroma.from_documents(
        documents=chunks,
        embedding=EMBEDDING_MODEL,
        persist_directory=VECTOR_STORE_DIR,
        collection_name="credit_policy",
    )

    print(f"Vector store built at {VECTOR_STORE_DIR}")


# This guard means `ingest()` only runs when the file is executed directly
# (`python app/rag/ingest.py`), not if some other module imports something
# from this file — standard Python convention for scripts that are also
# importable modules.
if __name__ == "__main__":
    ingest()
