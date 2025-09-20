from crewai import LLM
import os
# -----------------------------------------------------------------------------
# Helper factories to avoid duplication
# -----------------------------------------------------------------------------

_OPENAI_MODEL: str = "gpt-5-mini"
_OPENAI_API_KEY: str = "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"
_OPENAI_BASE_URL: str = "https://api.openai.com/v1"



def _get_ollama_base_url() -> str:
    """
    Returns the appropriate Ollama base URL depending on the environment.
    - Inside Docker containers: use service name 'ollama'
    - Outside Docker containers: use 'localhost'
    """
    # Check if we're running inside Docker by looking for common indicators
    if os.path.exists('/.dockerenv') or os.environ.get('DOCKER_CONTAINER'):
        return "http://ollama:11434"
    else:
        return "http://localhost:11434"


def _llm_default() -> LLM:
    return LLM(
        model="ollama/qwen3:8b-q4_K_M",
        base_url=_get_ollama_base_url(),
        api_key="ollama",
        temperature=0.7,
        max_tokens=1024,
        extra_body={"keep_alive": -1, "stop": ["\n\n", "\n###", "```"]},
    )

def _llm_leader() -> LLM:
    return LLM(
        model="ollama/qwen3:8b-q4_K_M",
        base_url=_get_ollama_base_url(),
        api_key="ollama",
        temperature=0.7,
        max_tokens=2048,
        extra_body={"keep_alive": -1, "stop": ["\n\n", "\n###", "```"]},
    )
'''

from crewai import LLM
import os

# -----------------------------------------------------------------------------
# OpenAI-based factories
# -----------------------------------------------------------------------------

def _llm_default() -> LLM:
    return LLM(
        model="gpt-5-nano",
        base_url="https://api.openai.com/v1",
        api_key=_OPENAI_API_KEY,
    )

def _llm_leader() -> LLM:
    return LLM(
        model="gpt-5-nano",
        base_url="https://api.openai.com/v1",
        api_key=_OPENAI_API_KEY,
    )
'''