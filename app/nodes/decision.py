"""
Decision node.

Runs after the human-in-the-loop interrupt has been resumed. Finalizes the
case using whatever the human decided (approve the AI recommendation as-is,
override it, or send it for further review) and appends the closing audit
entry.
"""

from datetime import datetime, timezone
from app.state import UnderwritingState


def decision_node(state: UnderwritingState) -> UnderwritingState:
    """
    Last node in the graph. By the time this runs, the human_review
    interrupt in graph.py has already paused execution, waited for a human
    to click a button in the UI, and resumed with `human_decision` (and
    optionally `human_notes`) filled into the state.

    This node doesn't make any decision itself — it just finalizes whatever
    the human chose and writes the closing audit entry.
    """
    # `or` here means: if a human explicitly made a call, use that; otherwise
    # fall back to the AI's original recommendation. In the Streamlit UI this
    # branch is effectively unreachable (the UI always sends a human_decision
    # before reaching this node), but it's a safe default if this graph is
    # ever driven by something other than that UI.
    final_decision = state.get("human_decision") or state.get("recommendation")

    audit_entry = {
        "node": "decision",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "detail": (
            f"Final decision: {final_decision}. "
            f"AI recommendation was: {state.get('recommendation')}. "
            f"Human notes: {state.get('human_notes') or 'none'}."
        ),
    }

    return {
        **state,
        "human_decision": final_decision,
        "audit_log": state.get("audit_log", []) + [audit_entry],
    }
