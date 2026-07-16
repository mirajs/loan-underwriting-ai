"""
Streamlit UI for the Loan Underwriting Copilot.

Run with: streamlit run streamlit_app.py

Demonstrates the full agentic RAG flow, including a genuine LangGraph
interrupt: the graph execution pauses after the reasoning node and
resumes only once a human clicks Approve / Override below.
"""

import json
import os
import glob
import streamlit as st
from langgraph.types import Command

from app.graph import build_graph

st.set_page_config(page_title="Loan Underwriting Copilot", layout="wide")
st.title("🏦 Intelligent Loan Underwriting Copilot")
st.caption(
    "Agentic RAG demo — LangGraph + Groq (fast scoring) + Gemini (reasoning) "
    "+ local Chroma retrieval. Human-in-the-loop approval required before any "
    "final decision."
)

# Streamlit re-runs this entire script top-to-bottom on every interaction
# (every button click, every widget change). st.session_state is the one
# thing that survives across those re-runs — without it, the compiled
# graph would get rebuilt and every result would be lost on each click.
# Each `if "x" not in st.session_state` block below is a one-time
# initialization: it only actually runs on the very first load.
if "graph" not in st.session_state:
    # build_graph() (from app/graph.py) compiles the LangGraph pipeline
    # once per browser session, not once per request — rebuilding it on
    # every click would be wasteful and would also reset the checkpointer.
    st.session_state.graph = build_graph()
if "thread_id" not in st.session_state:
    # LangGraph uses "thread_id" to know which conversation/case a given
    # invoke() belongs to — it's how the checkpointer knows which paused
    # state to resume when we later send Command(resume=...).
    st.session_state.thread_id = None
if "interrupted_payload" not in st.session_state:
    # Holds the AI's draft recommendation while we're waiting on the human
    # review step. None means "nothing pending review right now".
    st.session_state.interrupted_payload = None
if "final_state" not in st.session_state:
    # Holds the completed case (after a human decision) so the "Final
    # Decision" section below stays visible even after later re-runs.
    st.session_state.final_state = None

# Populates the "Load a sample application" dropdown with whatever .json
# files exist in data/sample_applications/, plus a manual-entry option.
sample_files = sorted(glob.glob("data/sample_applications/*.json"))
sample_names = ["-- manual entry --"] + [os.path.basename(f) for f in sample_files]

with st.sidebar:
    st.header("Application input")
    choice = st.selectbox("Load a sample application", sample_names)

    # If a sample was picked, load its values as defaults for every form
    # field below; otherwise fall back to a hardcoded blank-ish profile.
    # Either way, `defaults` just pre-fills the widgets — the user can still
    # edit any field before clicking submit.
    if choice != "-- manual entry --":
        with open(os.path.join("data/sample_applications", choice)) as f:
            defaults = json.load(f)
    else:
        defaults = {
            "applicant_name": "",
            "loan_amount": 10000,
            "loan_purpose": "debt consolidation",
            "annual_income": 50000,
            "credit_score": 680,
            "existing_debt": 10000,
            "employment_years": 3,
        }

    # Each st.* widget here returns the CURRENT value on every re-run (not
    # just when changed) — that's how Streamlit forms work. The variable
    # names on the left (applicant_name, loan_amount, ...) are what get
    # bundled into raw_application below when the submit button is pressed.
    applicant_name = st.text_input("Applicant name", defaults["applicant_name"])
    loan_amount = st.number_input("Loan amount ($)", value=int(defaults["loan_amount"]))
    loan_purpose = st.selectbox(
        "Loan purpose",
        ["debt consolidation", "auto purchase", "home improvement", "medical", "other"],
        index=0,
    )
    annual_income = st.number_input("Annual income ($)", value=int(defaults["annual_income"]))
    credit_score = st.slider("Credit score", 300, 850, int(defaults["credit_score"]))
    existing_debt = st.number_input("Existing debt ($)", value=int(defaults["existing_debt"]))
    employment_years = st.number_input(
        "Employment (years)", value=int(defaults["employment_years"])
    )

    submit = st.button("Run underwriting analysis", type="primary")

if submit:
    # A fresh, unique thread_id per submission — this is the key the
    # checkpointer uses to track this specific case's paused/resumed state.
    # Reusing the same thread_id across different applicants would let their
    # states collide.
    st.session_state.thread_id = f"case-{applicant_name or 'anon'}-{loan_amount}"
    # LangGraph expects thread_id nested inside a "configurable" dict — this
    # exact shape is required by the checkpointer API, not a stylistic choice.
    config = {"configurable": {"thread_id": st.session_state.thread_id}}

    # Bundles the form values into the shape intake_node expects to find at
    # state["raw_application"] — see app/nodes/intake.py's REQUIRED_FIELDS.
    raw_application = {
        "applicant_name": applicant_name,
        "loan_amount": loan_amount,
        "loan_purpose": loan_purpose,
        "annual_income": annual_income,
        "credit_score": credit_score,
        "existing_debt": existing_debt,
        "employment_years": employment_years,
    }

    with st.spinner("Running intake → policy retrieval → risk scoring → reasoning..."):
        # .invoke() runs the graph starting from intake, straight through
        # policy_rag, risk_scoring, and reasoning, and then INTO human_review
        # — where interrupt() fires and execution stops. This call blocks
        # until that happens (or, on the second invoke() after resume, until
        # the graph reaches END).
        result = st.session_state.graph.invoke(
            {"raw_application": raw_application}, config=config
        )

    # When a graph run stops on an interrupt rather than reaching END,
    # LangGraph doesn't return the normal state dict — instead the result
    # contains a special "__interrupt__" key holding a list of Interrupt
    # objects. [0] because our graph only ever has one interrupt() call
    # active at a time; .value is the payload dict we passed into
    # interrupt(...) back in graph.py's human_review_node.
    if "__interrupt__" in result:
        st.session_state.interrupted_payload = result["__interrupt__"][0].value
        st.session_state.final_state = None
    else:
        # No interrupt means the graph ran all the way to END in one go —
        # in this app that only happens on the SECOND invoke (the resume
        # call below), never on the first, since human_review always pauses.
        st.session_state.final_state = result
        st.session_state.interrupted_payload = None

# This whole block only renders once there's a pending interrupt to show —
# i.e. after the first invoke() above found "__interrupt__" in the result.
# It stays visible across re-runs (e.g. while the user is typing in the
# notes box) because interrupted_payload lives in session_state, not a
# local variable.
if st.session_state.interrupted_payload:
    payload = st.session_state.interrupted_payload
    st.subheader("🧠 AI Draft Recommendation — awaiting human review")

    # Two-column layout: wider column for the recommendation + citations,
    # narrower column for the at-a-glance risk metrics.
    col1, col2 = st.columns([2, 1])
    with col1:
        st.markdown(f"**Recommendation:** `{payload['recommendation']}`")
        st.markdown(f"**Rationale:** {payload['rationale']}")
        st.markdown("**Cited policy clauses:**")
        # Each citation gets its own collapsible expander so the reviewer
        # can check the exact policy wording without the page getting
        # cluttered by default — this is the human-readable version of the
        # same traceability the audit_log captures programmatically.
        for c in payload.get("citations", []):
            with st.expander(f"{c['clause_id']} — {c['source_doc']}"):
                st.write(c["text"])
    with col2:
        st.metric("Risk score", f"{payload.get('risk_score', 0):.2f}")
        st.markdown("**Risk flags:**")
        for flag in payload.get("risk_flags", []):
            st.write(f"- {flag}")

    st.divider()
    st.markdown("### Human decision")
    human_notes = st.text_area("Notes (optional)")
    c1, c2, c3 = st.columns(3)
    # `decision` stays None unless one of the three buttons was actually
    # clicked on this re-run — Streamlit buttons return True only on the
    # exact re-run triggered by their own click, so this correctly detects
    # "did the user just click something" rather than "was this ever clicked".
    decision = None
    if c1.button("✅ Approve AI recommendation"):
        # Accepts the AI's own recommendation string as-is (could be
        # "approve", "reject", or "refer").
        decision = payload["recommendation"]
    if c2.button("✏️ Override → Approve loan"):
        decision = "approve"
    if c3.button("✋ Override → Reject loan"):
        decision = "reject"

    if decision:
        # Must reuse the SAME thread_id from the original invoke() — this is
        # what tells the checkpointer which paused case to resume, rather
        # than starting a brand new graph run from scratch.
        config = {"configurable": {"thread_id": st.session_state.thread_id}}
        # Command(resume=...) is LangGraph's mechanism for feeding a value
        # back into the exact interrupt() call that's currently paused.
        # Whatever dict we pass here is what human_review_node's
        # `payload = interrupt(...)` line receives as its return value.
        # This invoke() resumes execution at human_review, then continues
        # on to decision_node, then reaches END — completing the graph.
        result = st.session_state.graph.invoke(
            Command(resume={"human_decision": decision, "human_notes": human_notes}),
            config=config,
        )
        st.session_state.final_state = result
        st.session_state.interrupted_payload = None
        # Forces Streamlit to immediately re-run the script from the top,
        # so the UI switches from "awaiting review" to "Final Decision"
        # without waiting for the next natural interaction.
        st.rerun()

# Renders once a case has actually completed (final_state is only ever set
# after decision_node has run, either just above via resume, or from an
# earlier completed case still held in session_state).
if st.session_state.final_state:
    state = st.session_state.final_state
    st.subheader("✅ Final Decision")
    st.markdown(f"**Decision:** `{state.get('human_decision')}`")
    st.markdown(f"**AI's original recommendation:** `{state.get('recommendation')}`")

    # The full step-by-step history assembled by every node along the way —
    # this is UnderwritingState["audit_log"] from state.py, rendered as a
    # simple timeline. Useful both as a demo talking point ("here's the full
    # traceability") and for actually debugging a run that went wrong.
    with st.expander("Full audit trail"):
        for entry in state.get("audit_log", []):
            st.markdown(f"**[{entry['node']}]** {entry['timestamp']}")
            st.write(entry["detail"])
            st.divider()
