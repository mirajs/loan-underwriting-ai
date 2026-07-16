"""
Wires the underwriting graph together.

Flow:
  intake -> policy_rag -> risk_scoring -> reasoning -> [HUMAN REVIEW] -> decision

The human review step uses LangGraph's `interrupt()` -- the graph execution
pauses after `reasoning`, returns control to the caller (the Streamlit app),
and resumes only when a human supplies a decision via `Command(resume=...)`.
This is the actual mechanism, not a UI-only illusion: state is checkpointed
and the graph genuinely halts mid-run.
"""

from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver
from langgraph.types import interrupt

from app.state import UnderwritingState
from app.nodes.intake import intake_node
from app.nodes.policy_rag import policy_rag_node
from app.nodes.risk_scoring import risk_scoring_node
from app.nodes.reasoning import reasoning_node
from app.nodes.decision import decision_node


def human_review_node(state: UnderwritingState) -> UnderwritingState:
    """
    Pauses the graph and surfaces the AI's draft recommendation + citations
    to a human. Resumes with whatever decision/notes the human supplies.

    How interrupt() actually works: calling it inside a node does two things
    in one step. First, it raises a special internal exception that LangGraph
    catches — this unwinds execution and returns control all the way back to
    whoever called graph.invoke(), carrying the dict passed to interrupt()
    as the payload (that's the `state.get("recommendation")` etc. below).
    The graph's progress up to this point is saved by the checkpointer, so
    nothing is lost. Second, when the caller later invokes the graph again
    with `Command(resume=some_value)`, LangGraph resumes execution from
    exactly this line — and `interrupt(...)` now returns `some_value`
    instead of raising. That's why `payload` below holds whatever dict the
    Streamlit app passed into Command(resume=...), even though textually it
    looks like it's just the return value of the interrupt() call above it.
    """
    payload = interrupt(
        {
            "recommendation": state.get("recommendation"),
            "rationale": state.get("recommendation_rationale"),
            "citations": state.get("citations"),
            "risk_score": state.get("risk_score"),
            "risk_flags": state.get("risk_flags"),
        }
    )
    # This line only runs AFTER resume — on the first pass through this
    # node, execution never reaches here at all.
    return {
        **state,
        "human_decision": payload.get("human_decision"),
        "human_notes": payload.get("human_notes"),
    }


def build_graph():
    """
    Assembles the StateGraph: registers every node function, wires the
    edges between them (which node runs after which), and compiles it into
    a runnable graph with checkpointing enabled.
    """
    # UnderwritingState (from state.py) tells StateGraph the shape of the
    # data flowing through every node — this is what makes each node's
    # `**state` spread pattern type-consistent across the whole graph.
    graph = StateGraph(UnderwritingState)

    # Registers each node under a string name. The name is what add_edge
    # below refers to, and also what shows up in the audit log / any
    # LangGraph tracing tooling.
    graph.add_node("intake", intake_node)
    graph.add_node("policy_rag", policy_rag_node)
    graph.add_node("risk_scoring", risk_scoring_node)
    graph.add_node("reasoning", reasoning_node)
    graph.add_node("human_review", human_review_node)
    graph.add_node("decision", decision_node)

    # This is the actual flow of the graph: a straight linear chain from
    # START to END, passing through every node in order. START and END are
    # LangGraph's built-in sentinel nodes marking the entry and exit points.
    # (A more complex agent might use conditional edges here — e.g. routing
    # straight to "decision" if risk_score is very low, skipping human
    # review — but a linear chain keeps this version easy to follow and
    # guarantees a human always reviews every case, which is the safer
    # default for a banking demo.)
    graph.add_edge(START, "intake")
    graph.add_edge("intake", "policy_rag")
    graph.add_edge("policy_rag", "risk_scoring")
    graph.add_edge("risk_scoring", "reasoning")
    graph.add_edge("reasoning", "human_review")
    graph.add_edge("human_review", "decision")
    graph.add_edge("decision", END)

    # The checkpointer is what makes interrupt()/resume actually work: it's
    # what persists the graph's state while execution is paused waiting for
    # a human. MemorySaver keeps checkpoints in RAM, which is fine for a
    # single-process demo but is lost if the app restarts. A persistent
    # checkpointer (Postgres/SQLite) would let a paused case survive a
    # restart — worth mentioning as a "production readiness" step in interviews.
    checkpointer = MemorySaver()

    # .compile() turns the node/edge definitions into an actual runnable
    # object with .invoke() — everything above this line just describes the
    # graph; nothing executes until compile() (and then .invoke()) is called.
    return graph.compile(checkpointer=checkpointer)
