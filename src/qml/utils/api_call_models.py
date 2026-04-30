import os
from typing import Any, Dict, Optional

from crewai import LLM


OPENAI_BASE_URL = "https://api.openai.com/v1"
DEFAULT_OPENAI_MODEL = "gpt-5-nano"
DEFAULT_OLLAMA_MODEL = "ollama/llama3.1"


class OpenAICompletionTokensLLM(LLM):
    """CrewAI LLM variant that sends OpenAI's GPT-5 token limit parameter."""

    def __init__(self, *args, max_completion_tokens: Optional[int] = None, **kwargs):
        self._openai_max_completion_tokens = max_completion_tokens
        super().__init__(*args, **kwargs)

    def _prepare_completion_params(self, *args, **kwargs) -> Dict[str, Any]:
        params = super()._prepare_completion_params(*args, **kwargs)
        params.pop("max_tokens", None)
        if self._openai_max_completion_tokens is not None:
            params["max_completion_tokens"] = self._openai_max_completion_tokens
        return params


def _get_ollama_base_url() -> str:
    """
    Returns the appropriate Ollama base URL depending on the environment.
    - Inside Docker containers: use service name 'ollama'
    - Outside Docker containers: use 'localhost'
    """
    if os.path.exists('/.dockerenv') or os.environ.get('DOCKER_CONTAINER'):
        return "http://ollama:11434"
    return "http://localhost:11434"


def _llm_ollama(max_tokens: int) -> LLM:
    return LLM(
        model=os.getenv("QML_OLLAMA_MODEL", DEFAULT_OLLAMA_MODEL),
        base_url=os.getenv("OLLAMA_BASE_URL", _get_ollama_base_url()),
        api_key="ollama",
        temperature=float(os.getenv("QML_LLM_TEMPERATURE", "0.2")),
        max_tokens=max_tokens,
    )


def _openai_api_key() -> str:
    api_key = os.getenv("OPENAI_API_KEY", "").strip()
    if not api_key:
        raise RuntimeError(
            "OPENAI_API_KEY is not configured. Set it in your environment or .env "
            "before using QML_LLM_PROVIDER=openai."
        )
    return api_key


def _llm_openai(max_tokens: int) -> LLM:
    kwargs = {
        "model": os.getenv("QML_OPENAI_MODEL", DEFAULT_OPENAI_MODEL),
        "base_url": os.getenv("OPENAI_BASE_URL", OPENAI_BASE_URL),
        "api_key": _openai_api_key(),
        "max_completion_tokens": max_tokens,
    }
    if os.getenv("QML_LLM_TEMPERATURE"):
        kwargs["temperature"] = float(os.getenv("QML_LLM_TEMPERATURE", "1"))

    return OpenAICompletionTokensLLM(
        **kwargs,
    )


def _llm_by_provider(max_tokens: int) -> LLM:
    provider = os.getenv("QML_LLM_PROVIDER", "ollama").strip().lower()
    if provider == "openai":
        return _llm_openai(max_tokens=max_tokens)
    if provider == "ollama":
        return _llm_ollama(max_tokens=max_tokens)
    raise ValueError("QML_LLM_PROVIDER must be either 'ollama' or 'openai'.")


def _llm_default() -> LLM:
    return _llm_by_provider(max_tokens=128)


def _llm_leader() -> LLM:
    return _llm_by_provider(max_tokens=256)


def _llm_openai_default() -> LLM:
    """Explicit OpenAI factory for quick latency tests without changing callers."""
    return _llm_openai(max_tokens=128)


def _llm_openai_leader() -> LLM:
    """Explicit OpenAI factory for quick latency tests without changing callers."""
    return _llm_openai(max_tokens=256)
