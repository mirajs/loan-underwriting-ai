"""
Risk scoring node.

Uses the fast tier (Groq / Llama 3.3 70B) to turn applicant data + retrieved
policy clauses into a structured risk score. This step is high-frequency
and latency-sensitive relative to the final reasoning step, which is why
it's routed to Groq rather than Gemini.
"""

import json
import re
from datetime import datetime, timezone
from app.state import UnderwritingState
from app.config import get_reasoning_llm, call_with_fallback, extract_text

# The system prompt is where we constrain the model to a fixed, parseable
# output shape. Being explicit about "ONLY a JSON object" and "no markdown
# code fences" cuts down on how often the model wraps its answer in ```json
# blocks or adds a sentence of preamble before the JSON — both of which
# would break the parser below if we didn't guard against them anyway.
SYSTEM_PROMPT = """You are a credit risk scoring assistant for a bank.
Given an applicant's financial profile and relevant policy clauses, output
ONLY a JSON object with these exact keys:
  "risk_score": float between 0.0 (very low risk) and 1.0 (very high risk)
  "risk_flags": list of short strings, each a specific concern (e.g. "high debt-to-income")
  "risk_rationale": one or two sentence explanation

No prose outside the JSON. No markdown code fences."""


def _build_user_prompt(applicant: dict, clauses: list) -> str:
    """
    Assembles the actual data the model needs to score this specific
    applicant: their profile plus whatever policy clauses were retrieved
    upstream by policy_rag_node.
    """
    # [clause_id] prefix on each clause lets the model (and us, reading the
    # logs) refer back to a specific clause unambiguously later.
    clause_text = "\n\n".join(
        f"[{c['clause_id']}] {c['text']}" for c in clauses
    ) or "No policy clauses retrieved."

    return f"""Applicant profile:
{json.dumps(applicant, indent=2)}

Relevant policy clauses:
{clause_text}

Score this applicant's risk."""


def _parse_json_response(text: str) -> dict:
    """
    LLMs are unreliable about following "output ONLY JSON" to the letter —
    they'll sometimes wrap the answer in ```json fences or add a stray
    sentence. This function defends against both rather than trusting the
    prompt alone.
    """
    # Strip any ```json or ``` fences the model added despite being told not to.
    cleaned = re.sub(r"```(json)?", "", text).strip()
    # re.DOTALL lets `.` match newlines, so this grabs everything between the
    # first `{` and the last `}` even if the JSON spans multiple lines.
    match = re.search(r"\{.*\}", cleaned, re.DOTALL)
    if not match:
        # Fail loudly with a snippet of the actual response — much easier to
        # debug a bad prompt or a misbehaving model than a bare "invalid JSON".
        raise ValueError(f"Could not parse JSON from risk scoring response: {text[:200]}")
    return json.loads(match.group(0))


def risk_scoring_node(state: UnderwritingState) -> UnderwritingState:
    """
    Uses the fast Groq tier to produce a structured risk score. This is the
    first LLM call in the graph, and the first place free-tier rate limits
    could realistically be hit, hence the call_with_fallback wrapper.
    """
    applicant = state["applicant"]
    clauses = state.get("retrieved_clauses", [])

    user_prompt = _build_user_prompt(applicant, clauses)

    # This inner function is the "closure" mentioned in config.py's
    # call_with_fallback docstring: it captures user_prompt from the
    # enclosing scope, and describes exactly what to do with WHICHEVER llm
    # object gets passed in — primary or fallback, the logic is identical.
    def _call(llm):
        response = llm.invoke(
            [("system", SYSTEM_PROMPT), ("human", user_prompt)]
        )
        return _parse_json_response(extract_text(response.content))

    # NOTE: Originally routed to Groq (get_fast_llm) as the fast tier in a
    # two-model cost/speed architecture. Temporarily routed through Gemini
    # instead, since Groq's free tier stopped granting access to Llama
    # models and their Developer (pay-as-you-go) tier is currently closed
    # to new upgrades. Switch back to get_fast_llm() once Groq's tier
    # reopens -- see app/config.py's get_fast_llm docstring.
    # get_reasoning_llm() is called once, eagerly, to build the primary client;
    # call_with_fallback only swaps to get_fallback_llm() internally if
    # _call(primary) raises.
    result = call_with_fallback(_call, get_reasoning_llm())

    audit_entry = {
        "node": "risk_scoring",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "detail": f"Risk score {result.get('risk_score')} — flags: {result.get('risk_flags')}",
    }

    return {
        **state,
        "risk_score": result.get("risk_score"),
        "risk_flags": result.get("risk_flags", []),
        "risk_rationale": result.get("risk_rationale", ""),
        "audit_log": state.get("audit_log", []) + [audit_entry],
    }
