"""
Shared state schema passed between LangGraph nodes.

Keeping this as a single typed dict (rather than passing loose args between
nodes) is what makes the graph inspectable and the audit trail possible --
every node reads its inputs from and writes its outputs to this one object,
so the full history of a case can be logged and replayed.
"""

from typing import TypedDict, List, Dict, Optional


class PolicyCitation(TypedDict):
    """
    One retrieved chunk of policy text, tagged with where it came from.
    Used both for what the RAG step retrieves and for what the reasoning
    step ultimately cites — same shape, so we can filter one down to the
    other without any conversion.
    """
    clause_id: str      # e.g. "personal_loan_policy.md-3" — unique per chunk, set at ingest time
    source_doc: str      # the filename the clause came from, for traceability
    text: str            # the actual chunk of policy text


# total=False means none of these keys are required to be present at once —
# the state starts nearly empty (just raw_application) and each node adds
# its own keys as the graph progresses. This matches how LangGraph actually
# uses the state: it's built up incrementally, not constructed all at once.
class UnderwritingState(TypedDict, total=False):
    # --- Intake ---
    raw_application: Dict            # unparsed input as received (whatever the UI/API sent)
    applicant: Dict                  # structured applicant fields, filled in by intake_node

    # --- Policy RAG ---
    # Every clause retrieved for this applicant, whether or not it ends up
    # being cited in the final recommendation. Kept separate from
    # `citations` below so we can audit what was available vs. what was used.
    retrieved_clauses: List[PolicyCitation]

    # --- Risk scoring --- (filled in by risk_scoring_node, via Groq)
    risk_flags: List[str]            # short strings, e.g. "high debt-to-income"
    risk_score: float                # 0.0 (low risk) - 1.0 (high risk)
    risk_rationale: str              # one or two sentence explanation from the model

    # --- Reasoning / recommendation --- (filled in by reasoning_node, via Gemini)
    recommendation: str              # "approve" | "reject" | "refer" — the AI's draft decision
    recommendation_rationale: str    # explanation, expected to reference clause_id values
    citations: List[PolicyCitation]  # subset of retrieved_clauses the model actually relied on

    # --- Human-in-the-loop --- (filled in after the interrupt() resumes)
    human_decision: Optional[str]    # the human's final call — may match or override `recommendation`
    human_notes: Optional[str]       # free-text notes the human typed in the review step

    # --- Audit trail ---
    # Every node appends one entry here rather than overwriting anything, so
    # by the time decision_node runs, this list is the full step-by-step
    # history of the case — this is what gets shown in the "Full audit
    # trail" expander in the Streamlit UI.
    audit_log: List[Dict]            # append-only list of {node, timestamp, detail}
