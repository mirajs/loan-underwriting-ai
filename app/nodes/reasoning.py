"""
Reasoning node.

Uses the deeper-context tier (Gemini 2.5 Flash) to synthesize the applicant
profile, risk score, and retrieved policy clauses into a final underwriting
recommendation with explicit citations. This is the step where answer
quality matters more than latency, hence the model tier switch.
"""

import json
import re
from datetime import datetime, timezone
from app.state import UnderwritingState
from app.config import get_reasoning_llm, call_with_fallback, extract_text

# "You must ground every claim in the provided policy clauses -- never
# invent a policy rule" is the single most important line in this prompt:
# it's what makes citations trustworthy rather than the model just
# hallucinating a plausible-sounding rule. Still worth spot-checking real
# outputs against the actual policy doc — prompting reduces hallucination,
# it doesn't eliminate it.
SYSTEM_PROMPT = """You are an underwriting recommendation assistant for a bank.
You must ground every claim in the provided policy clauses -- never invent a
policy rule. Given the applicant profile, the risk assessment, and the
retrieved policy clauses, output ONLY a JSON object with these exact keys:
  "recommendation": one of "approve", "reject", "refer"
  "recommendation_rationale": a short paragraph explaining the recommendation,
      explicitly referencing clause_id values where a policy rule is invoked
  "citations": list of clause_id strings actually relied upon

No prose outside the JSON. No markdown code fences."""


def _build_user_prompt(applicant: dict, risk_score, risk_flags, clauses: list) -> str:
    """
    Combines everything upstream nodes have produced so far — applicant
    profile, the Groq risk score, and the retrieved policy clauses — into
    one prompt for the reasoning model to synthesize.
    """
    # Including source_doc here (unlike risk_scoring.py's version) gives the
    # model a second way to refer to a clause in its rationale, and helps a
    # human reviewer sanity-check citations without cross-referencing clause_id.
    clause_text = "\n\n".join(
        f"[{c['clause_id']}] ({c['source_doc']}) {c['text']}" for c in clauses
    ) or "No policy clauses retrieved."

    return f"""Applicant profile:
{json.dumps(applicant, indent=2)}

Risk assessment:
  risk_score: {risk_score}
  risk_flags: {risk_flags}

Relevant policy clauses:
{clause_text}

Draft the underwriting recommendation."""


def _parse_json_response(text: str) -> dict:
    # Same defensive parsing pattern as risk_scoring.py — see that file's
    # comments for why this is necessary rather than trusting the prompt alone.
    cleaned = re.sub(r"```(json)?", "", text).strip()
    match = re.search(r"\{.*\}", cleaned, re.DOTALL)
    if not match:
        raise ValueError(f"Could not parse JSON from reasoning response: {text[:200]}")
    return json.loads(match.group(0))


def reasoning_node(state: UnderwritingState) -> UnderwritingState:
    """
    The most important node in the graph: synthesizes everything gathered
    so far into a draft recommendation, using the Gemini reasoning tier
    rather than the fast Groq tier, because getting the citations right
    matters more here than speed.
    """
    applicant = state["applicant"]
    clauses = state.get("retrieved_clauses", [])

    user_prompt = _build_user_prompt(
        applicant, state.get("risk_score"), state.get("risk_flags"), clauses
    )

    def _call(llm):
        response = llm.invoke(
            [("system", SYSTEM_PROMPT), ("human", user_prompt)]
        )
        return _parse_json_response(extract_text(response.content))

    result = call_with_fallback(_call, get_reasoning_llm())

    # The model returns a list of clause_id strings it says it relied on.
    # Here we cross-reference those IDs against the clauses we actually
    # retrieved, so `citations` only ever contains real, verifiable clauses
    # — if the model hallucinates a clause_id that was never retrieved, it
    # silently gets dropped rather than shown to the human reviewer as if
    # it were real.
    cited_ids = set(result.get("citations", []))
    citations = [c for c in clauses if c["clause_id"] in cited_ids]

    audit_entry = {
        "node": "reasoning",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "detail": f"Draft recommendation: {result.get('recommendation')} "
        f"(citing {len(citations)} clause(s))",
    }

    return {
        **state,
        "recommendation": result.get("recommendation"),
        "recommendation_rationale": result.get("recommendation_rationale", ""),
        "citations": citations,
        "audit_log": state.get("audit_log", []) + [audit_entry],
    }
