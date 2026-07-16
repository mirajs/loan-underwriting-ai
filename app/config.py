"""
Central place for model clients.

Design: two-tier routing.
  - FAST_LLM   -> Groq / Llama 3.3 70B   (free, ~700 tok/s) for structured,
                  low-latency steps: classification, risk scoring.
  - REASONING_LLM -> Gemini 2.5 Flash (free, large context) for the final
                  underwriting recommendation, where citation accuracy and
                  multi-document reasoning matter more than speed.

Both are swappable. If a key is missing, config.py raises a clear error
rather than failing deep inside a graph node.
"""

import os
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_huggingface import HuggingFaceEmbeddings

# Reads the .env file in the project root and loads GROQ_API_KEY etc. into
# os.environ. Must run before os.getenv() below, or the keys read as None
# even if .env is filled in correctly.
load_dotenv()

# os.getenv returns None (not an error) if a variable isn't set — that's
# intentional here, so we can check for it ourselves below and raise a clear
# message instead of a cryptic auth error from deep inside a provider's SDK.
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")

# os.path.dirname(__file__) = the app/ folder this file lives in.
# ".." goes up one level to the project root, then into data/, so these
# paths resolve correctly no matter what directory you run a script from.
VECTOR_STORE_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "vector_store")
POLICY_DOCS_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "policy_docs")

# Local, free, no rate limits — applicant/policy data never leaves the machine
# for the embedding step. This downloads the model (~130MB) the first time
# it runs and caches it locally; every run after that is instant and offline.
EMBEDDING_MODEL = HuggingFaceEmbeddings(model_name="BAAI/bge-small-en-v1.5")

def extract_text(content) -> str:
    """
    Normalizes an LLM response's .content into a plain string.

    Most providers return response.content as a plain string. Some Gemini
    responses (and some other providers) instead return a LIST of content
    blocks -- e.g. [{"type": "text", "text": "..."}] -- especially as
    provider SDKs evolve. Downstream JSON-parsing code only knows how to
    work with a string, so every node that reads .content should pass it
    through this function first rather than assuming it's already a string.
    """
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for block in content:
            if isinstance(block, str):
                parts.append(block)
            elif isinstance(block, dict):
                parts.append(block.get("text", str(block)))
            else:
                parts.append(str(block))
        return "".join(parts)
    return str(content)

def get_fast_llm(temperature: float = 0.0):
    """
    Groq Llama 3.3 70B — fast, free, used for structured scoring steps.

    temperature=0.0 by default: for risk scoring we want consistent,
    repeatable output (same applicant in -> same score out), not creative
    variation. Raise it only if you deliberately want more varied phrasing.
    """
    if not GROQ_API_KEY:
        # Fail here, at the moment the LLM is requested, rather than letting
        # a downstream API call throw a vague "401 Unauthorized" that doesn't
        # say which key is missing or where to get one.
        raise RuntimeError(
            "GROQ_API_KEY not set. Get a free key (no card) at https://console.groq.com"
        )
    return ChatGroq(
        model="llama-3.3-70b-versatile",
        temperature=temperature,
        api_key=GROQ_API_KEY,
    )


def get_reasoning_llm(temperature: float = 0.1):
    """
    Gemini 3.5 Flash — larger context, used for the final recommendation.

    temperature=0.1 (slightly above 0): a touch of variation keeps the
    written rationale from reading as robotically identical across similar
    cases, while staying close to deterministic for consistency.
    """
    if not GOOGLE_API_KEY:
        raise RuntimeError(
            "GOOGLE_API_KEY not set. Get a free key (no card) at https://aistudio.google.com"
        )
    return ChatGoogleGenerativeAI(
        model="gemini-3.1-flash-lite",
        temperature=temperature,
        google_api_key=GOOGLE_API_KEY,
    )


def get_fallback_llm(temperature: float = 0.0):
    """
    OpenRouter free model — used only if primary providers are rate-limited.

    The import is placed inside the function (not at the top of the file) so
    that people who never configure OpenRouter don't need the
    langchain_openai package installed just to use Groq/Gemini.
    """
    from langchain_openai import ChatOpenAI

    if not OPENROUTER_API_KEY:
        raise RuntimeError(
            "OPENROUTER_API_KEY not set. Get a free key at https://openrouter.ai"
        )
    return ChatOpenAI(
        model="meta-llama/llama-3.3-70b-instruct:free",
        temperature=temperature,
        api_key=OPENROUTER_API_KEY,
        # OpenRouter exposes an OpenAI-compatible endpoint, so we can reuse
        # LangChain's OpenAI client class just by pointing base_url elsewhere.
        base_url="https://openrouter.ai/api/v1",
    )


def call_with_fallback(fn, *args, **kwargs):
    """
    Wraps a node's LLM call. If the primary call raises (commonly a 429 from
    a free-tier rate limit), retries once against the OpenRouter fallback.

    Usage: call_with_fallback(lambda llm: llm.invoke(prompt), get_fast_llm())

    Why a lambda instead of passing the prompt directly: each node's
    prompt-building and response-parsing logic differs, so `fn` is the whole
    "call this llm and parse its output" closure — that way the exact same
    parsing logic runs whether we ended up on Groq, Gemini, or the fallback.
    """
    # args[0] is the primary LLM client (whatever get_fast_llm() or
    # get_reasoning_llm() returned). kwargs["llm"] is supported too for
    # flexibility, though every call site in this project uses positional args.
    primary_llm = args[0] if args else kwargs.get("llm")
    try:
        return fn(primary_llm)
    except Exception as e:
        # Broad except is deliberate: ANY failure from the primary provider
        # (rate limit, timeout, transient 5xx) should trigger the fallback
        # rather than crashing the whole graph run.
        print(f"[fallback] primary LLM call failed ({e}); retrying on OpenRouter free tier")
        return fn(get_fallback_llm())
