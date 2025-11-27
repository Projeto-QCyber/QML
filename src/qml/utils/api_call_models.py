from crewai import LLM
import os

# -----------------------------------------------------------------------------
# OpenAI GPT-5.1 Codex Mini (low reasoning) configuration
# Requires OPENAI_API_KEY environment variable
# -----------------------------------------------------------------------------

_OPENAI_MODEL = "gpt-5.1-codex-mini"
_OPENAI_BASE_URL = os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1")


def _require_openai_api_key() -> str:
    api_key = os.getenv("OPENAI_API_KEY", "").strip()
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY is required to run the crews.")
    return api_key


def _llm_default() -> LLM:
    return LLM(
        model=_OPENAI_MODEL,
        base_url=_OPENAI_BASE_URL,
        api_key=_require_openai_api_key(),
        max_tokens=2048,
        reasoning_effort="low",
    )


def _llm_leader() -> LLM:
    return LLM(
        model=_OPENAI_MODEL,
        base_url=_OPENAI_BASE_URL,
        api_key=_require_openai_api_key(),
        max_tokens=4096,
        reasoning_effort="low",
    )