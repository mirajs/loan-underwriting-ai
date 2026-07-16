# Intelligent Loan Underwriting Copilot

An agentic RAG system for loan underwriting, built on **LangGraph**, with a
human-in-the-loop approval gate. Designed to run entirely on **free-tier LLM
APIs** — no paid inference required.

## Why this project

Banks underwrite loans against two things: the applicant's numbers, and a
pile of internal credit policy documents nobody wants to re-read for every
case. This copilot automates the first pass — retrieving the relevant policy
clauses, scoring risk, and drafting a recommendation — while keeping a human
underwriter as the final decision-maker. Every recommendation is traceable
back to the exact policy clause that supports it.

## Architecture

```
Applicant Data ─┐
                 ▼
          ┌─────────────┐
          │ Intake Node │  structures raw application into a typed schema
          └──────┬──────┘
                 ▼
          ┌─────────────────┐
          │ Policy RAG Node │  retrieves relevant clauses (Chroma + local embeddings)
          └──────┬──────────┘
                 ▼
          ┌────────────────────┐
          │ Risk Scoring Node  │  Groq / Llama 3.3 70B — fast structured scoring
          └──────┬─────────────┘
                 ▼
          ┌────────────────────┐
          │ Reasoning Node     │  Gemini 2.5 Flash — drafts recommendation + citations
          └──────┬─────────────┘
                 ▼
          ┌───────────────────────────┐
          │ Human-in-the-Loop Gate    │  LangGraph interrupt() — pauses for approval
          └──────┬────────────────────┘
                 ▼
          ┌─────────────────┐
          │ Decision Node   │  logs final decision + full audit trail
          └─────────────────┘
```

**Two-model cost/quality tiering:** fast, cheap steps (structured risk
scoring, classification) run on Groq's free Llama 3.3 70B tier for speed;
the final recommendation — where reasoning quality and citation accuracy
matter most — runs on Gemini 2.5 Flash's free tier, which has a larger
context window for reasoning across multiple retrieved policy clauses.
Both are swappable via `app/config.py`, and OpenRouter free models are
wired in as a fallback if either provider is rate-limited.

**Privacy-by-design:** embeddings run locally via `sentence-transformers`
— applicant and policy data never leaves your machine for the retrieval
step. Only the final, already-anonymizable reasoning prompt goes to a
third-party LLM. This is a deliberate architecture choice worth
highlighting in interviews, not just a cost-saving hack.

## Stack

| Layer | Tool | Cost |
|---|---|---|
| Orchestration | LangGraph | Free |
| Fast agent steps | Groq (Llama 3.3 70B) | Free tier |
| Reasoning / drafting | Google AI Studio (Gemini 2.5 Flash) | Free tier |
| Fallback | OpenRouter free models | Free tier |
| Embeddings | sentence-transformers (bge-small-en) | Free, local |
| Vector store | Chroma (local, persistent) | Free |
| UI | Streamlit | Free |

## Setup

```bash
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt

cp .env.example .env
# Fill in GROQ_API_KEY and GOOGLE_API_KEY (both free, no credit card):
#   Groq:   https://console.groq.com
#   Google: https://aistudio.google.com

python app/rag/ingest.py        # builds the local vector store from data/policy_docs
streamlit run streamlit_app.py
```

## Project structure

```
loan-underwriting-copilot/
├── app/
│   ├── config.py           # model clients (Groq, Gemini, OpenRouter fallback)
│   ├── state.py             # LangGraph state schema
│   ├── graph.py              # graph wiring: nodes, edges, interrupt
│   ├── nodes/
│   │   ├── intake.py
│   │   ├── policy_rag.py
│   │   ├── risk_scoring.py
│   │   ├── reasoning.py
│   │   └── decision.py
│   └── rag/
│       ├── ingest.py         # builds Chroma index from policy docs
│       └── retriever.py
├── data/
│   ├── policy_docs/          # sample credit policy documents (.md)
│   └── sample_applications/  # sample loan applications (.json)
├── streamlit_app.py          # UI with human-in-the-loop approval step
├── requirements.txt
└── .env.example
```

## Roadmap / talking points for interviews

- Swap Chroma for a managed vector DB (Pinecone/Qdrant Cloud) for a
  "production-readiness" story.
- Add a second retriever pass for bureau-style structured data (mocked here).
- Add LangSmith tracing (free tier) for observability — strong for the
  "how do you monitor agentic systems in production" question.
- Extend the audit log to a proper structured store (Postgres) for
  compliance-grade traceability.
