"""
Intake node.

Takes whatever the raw application payload looks like (a dict from a form,
an uploaded JSON, etc.) and normalizes it into the fields the rest of the
graph expects. Kept deterministic (no LLM call) since this is a structural
step, not a reasoning one -- no point spending free-tier quota on it.
"""

from datetime import datetime, timezone
from app.state import UnderwritingState

# The fields every downstream node (RAG query, risk scoring, reasoning)
# assumes will exist on `applicant`. Centralized here as a list rather than
# scattered checks elsewhere, so adding a new required field only means
# editing this one line.
REQUIRED_FIELDS = [
    "applicant_name",
    "loan_amount",
    "loan_purpose",
    "annual_income",
    "credit_score",
    "existing_debt",
    "employment_years",
]


def intake_node(state: UnderwritingState) -> UnderwritingState:
    """
    First node in the graph. Every LangGraph node is just a function that
    takes the current state and returns a (partial) updated state — LangGraph
    merges the return value back into the running state automatically.

    This node does no LLM call: normalizing a dict's fields is a mechanical
    step, not a reasoning one, so there's no reason to spend free-tier quota
    on it.
    """
    raw = state["raw_application"]

    # .get(f) returns None for any field missing from the raw input instead
    # of raising a KeyError — we'd rather flag missing fields in the audit
    # log than crash the whole pipeline on a malformed application.
    missing = [f for f in REQUIRED_FIELDS if f not in raw]
    applicant = {f: raw.get(f) for f in REQUIRED_FIELDS}

    # Derived field: debt-to-income ratio. Computed once here rather than in
    # every downstream node that needs it (risk scoring, policy RAG query
    # construction both use this).
    if applicant["annual_income"] and applicant["annual_income"] > 0:
        applicant["debt_to_income"] = round(
            applicant["existing_debt"] / applicant["annual_income"], 3
        )
    else:
        # Guards against a ZeroDivisionError if income is 0 or missing —
        # better to leave the ratio as None (visibly "unknown") than crash.
        applicant["debt_to_income"] = None

    audit_entry = {
        "node": "intake",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "detail": f"Parsed application for {applicant.get('applicant_name', 'unknown')}."
        + (f" Missing fields: {missing}" if missing else " All required fields present."),
    }

    # The **state spread keeps every existing key untouched and only adds/
    # overwrites `applicant` and `audit_log`. This pattern repeats in every
    # node in the graph — it's how state accumulates across the whole run
    # without each node needing to know about fields it doesn't touch.
    return {
        **state,
        "applicant": applicant,
        "audit_log": state.get("audit_log", []) + [audit_entry],
    }
