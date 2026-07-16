"""
Policy RAG node.

Retrieves the credit policy clauses most relevant to this applicant's
profile (loan purpose, amount, credit score band). No LLM call here --
pure retrieval -- keeping the free-tier LLM quota for the steps that
actually need reasoning.
"""

from datetime import datetime, timezone
from app.state import UnderwritingState
from app.rag.retriever import get_retriever


def policy_rag_node(state: UnderwritingState) -> UnderwritingState:
    """
    The "R" in RAG: turns the applicant's profile into a search query, and
    pulls back the most similar chunks of policy text from the local Chroma
    index (built ahead of time by app/rag/ingest.py).

    No LLM call here either — this is pure vector similarity search, which
    is fast, free, and doesn't touch any API quota at all.
    """
    applicant = state["applicant"]

    # This sentence becomes the thing we embed and search with. Writing it
    # as natural language (rather than just concatenating raw field values)
    # matters because the embedding model was trained on natural text — a
    # well-formed sentence retrieves more relevant clauses than a bag of
    # numbers would.
    query = (
        f"Underwriting policy for a {applicant.get('loan_purpose')} loan of "
        f"amount {applicant.get('loan_amount')}, applicant credit score "
        f"{applicant.get('credit_score')}, debt-to-income ratio "
        f"{applicant.get('debt_to_income')}."
    )

    # k=4: pull back the top 4 most similar chunks. See retriever.py for why
    # 4 is a reasonable default (context size vs. relevance tradeoff).
    retriever = get_retriever(k=4)

    # .invoke() runs: embed the query text -> compare against every stored
    # chunk's embedding -> return the k closest matches. Each `doc` is a
    # LangChain Document object with .page_content (the text) and .metadata
    # (whatever ingest.py attached — clause_id, source_doc).
    results = retriever.invoke(query)

    # Convert LangChain's Document objects into plain dicts matching our
    # PolicyCitation shape (see state.py), so the rest of the graph doesn't
    # need to know anything about LangChain's internal types.
    retrieved_clauses = [
        {
            "clause_id": doc.metadata.get("clause_id", "unknown"),
            "source_doc": doc.metadata.get("source_doc", "unknown"),
            "text": doc.page_content,
        }
        for doc in results
    ]

    audit_entry = {
        "node": "policy_rag",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "detail": f"Retrieved {len(retrieved_clauses)} policy clause(s) for query: {query}",
    }

    return {
        **state,
        "retrieved_clauses": retrieved_clauses,
        "audit_log": state.get("audit_log", []) + [audit_entry],
    }
