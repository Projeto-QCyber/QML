import os
import logging
from dataclasses import dataclass
from typing import Literal
from dotenv import load_dotenv
from crewai import LLM

load_dotenv()

Environment = Literal["TEST", "DEV", "PROD"]
logger = logging.getLogger(__name__)

@dataclass(frozen=True)
class LLMEnvironmentConfig:
    default_model: str
    leader_model: str
    coder_model: str
    default_max_tokens: int
    leader_max_tokens: int
    coder_max_tokens: int


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


def _get_env_mode() -> Environment:
    env = (
        os.getenv("QCYBER_ENV")
        or os.getenv("APP_ENV")
        or os.getenv("ENV_MODE")
        or os.getenv("ENVIRONMENT")
        or "DEV"
    ).strip().upper()

    if env not in {"TEST", "DEV", "PROD"}:
        raise ValueError("Ajuste o ambiente para suportar apenas valores: 'TEST', 'DEV' e 'PROD'.")
    return env

def _get_first_env(*names: str, default: str | None = None) -> str | None:
    for name in names:
        value = os.getenv(name)
        if value:
            return value.strip()
    return default


def _get_int_env(name: str, default: int) -> int:
    value = os.getenv(name)
    if value is None or value.strip() == "":
        return default
    try:
        return int(value)
    except ValueError as exc:
        raise ValueError(f"{name} deve ser um inteiro, valor atual: {value!r}") from exc


def _env_flag(name: str, default: bool = False) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "y", "on"}


def _require_prod_value(value: str | None, env_name: str) -> str:
    if value:
        return value
    raise ValueError(f"Configure {env_name} no .env antes de rodar em PROD.")


def _env_mode_select_model_llm(env: Environment | None = None) -> LLMEnvironmentConfig:
    selected_env = env or _get_env_mode()

    if selected_env == "TEST":
        default_model = _get_first_env(
            "LLM_MODEL_TEST_LLM_DEFAULT",
            "LLM_MODEL_TEST_DEFAULT",
            "LLM_MODEL_TEST2",
            default="ollama/qwen3:3b-q4_K_M",
        )
        leader_model = _get_first_env(
            "LLM_MODEL_TEST_LLM_LEADER",
            "LLM_MODEL_TEST_LEADER",
            "LLM_MODEL_TEST2",
            default="ollama/qwen3:8b-q4_K_M",
        )
        coder_model = _get_first_env(
            "LLM_MODEL_TEST_LLM_CODER",
            "LLM_MODEL_TEST_CODER",
            "LLM_MODEL_TEST3",
            default="ollama/qwen2.5-coder:7b",
        )
        default_tokens = _get_int_env("MAX_TOKENS_TEST_LLM_DEFAULT", 128)
        leader_tokens = _get_int_env("MAX_TOKENS_TEST_LLM_LEADER", 256)
        coder_tokens = _get_int_env("MAX_TOKENS_TEST_LLM_CODER", 512)
    elif selected_env == "DEV":
        default_model = _get_first_env(
            "LLM_MODEL_DEV_LLM_DEFAULT",
            "LLM_MODEL_DEV_DEFAULT",
            "LLM_MODEL_DEV3",
            default="ollama/qwen3:3b-q4_K_M",
        )
        leader_model = _get_first_env(
            "LLM_MODEL_DEV_LLM_LEADER",
            "LLM_MODEL_DEV_LEADER",
            "LLM_MODEL_DEV3",
            default="ollama/qwen3:8b-q4_K_M",
        )
        coder_model = _get_first_env(
            "LLM_MODEL_DEV_LLM_CODER",
            "LLM_MODEL_DEV_CODER",
            "LLM_MODEL_DEV2",
            default="ollama/qwen2.5-coder:7b",
        )
        default_tokens = _get_int_env("MAX_TOKENS_DEV_LLM_DEFAULT", 256)
        leader_tokens = _get_int_env("MAX_TOKENS_DEV_LLM_LEADER", 512)
        coder_tokens = _get_int_env("MAX_TOKENS_DEV_LLM_CODER", 1024)
    elif selected_env == "PROD":
        default_model = _require_prod_value(
            _get_first_env("LLM_MODEL_PROD_LLM_DEFAULT", "LLM_MODEL_PROD_DEFAULT", "LLM_MODEL_PROD1"),
            "LLM_MODEL_PROD_LLM_DEFAULT",
        )
        leader_model = _require_prod_value(
            _get_first_env("LLM_MODEL_PROD_LLM_LEADER", "LLM_MODEL_PROD_LEADER", "LLM_MODEL_PROD2"),
            "LLM_MODEL_PROD_LLM_LEADER",
        )
        coder_model = _require_prod_value(
            _get_first_env("LLM_MODEL_PROD_LLM_CODER", "LLM_MODEL_PROD_CODER", "LLM_MODEL_PROD3"),
            "LLM_MODEL_PROD_LLM_CODER",
        )
        default_tokens = _get_int_env("MAX_TOKENS_PROD_LLM_DEFAULT", 256)
        leader_tokens = _get_int_env("MAX_TOKENS_PROD_LLM_LEADER", 512)
        coder_tokens = _get_int_env("MAX_TOKENS_PROD_LLM_CODER", 1024)
    else:
        raise ValueError("Ajuste o env para suportar apenas valores: 'TEST', 'DEV' e 'PROD'.")

    return LLMEnvironmentConfig(
        default_model=default_model,
        leader_model=leader_model,
        coder_model=coder_model,
        default_max_tokens=default_tokens,
        leader_max_tokens=leader_tokens,
        coder_max_tokens=coder_tokens,
    )


def _debug_llm_config(role: str, model_name: str, max_tokens: int, base_url: str) -> None:
    if not _env_flag("QCYBER_DEBUG_LLM_CONFIG"):
        return

    env = _get_env_mode()
    message = (
        "[QCyber LLM config] "
        f"env={env} role={role} model={model_name} max_tokens={max_tokens} base_url={base_url}"
    )
    logger.info(message)
    print(message)


def _llm_default(model_name: str | None = None, max_tokens: int | None = None) -> LLM:
    config = _env_mode_select_model_llm()
    selected_model = model_name or config.default_model
    selected_max_tokens = max_tokens or config.default_max_tokens
    base_url = _get_ollama_base_url()
    #_debug_llm_config("default", selected_model, selected_max_tokens, base_url)

    return LLM(
        model=selected_model,
        base_url=base_url,
        api_key="ollama",
        temperature=0.1,
        max_completion_tokens=selected_max_tokens,
    )


def _llm_leader(model_name: str | None = None, max_tokens: int | None = None) -> LLM:
    config = _env_mode_select_model_llm()
    selected_model = model_name or config.leader_model
    selected_max_tokens = max_tokens or config.leader_max_tokens
    base_url = _get_ollama_base_url()
    #_debug_llm_config("leader", selected_model, selected_max_tokens, base_url)

    return LLM(
        model=selected_model,
        base_url=base_url,
        api_key="ollama",
        temperature=0.1,
        max_completion_tokens=selected_max_tokens,
    )


def _llm_coder_response(model_name: str | None = None, max_tokens: int | None = None) -> LLM:
    config = _env_mode_select_model_llm()
    selected_model = model_name or os.getenv("QML_CODER_MODEL") or config.coder_model
    selected_max_tokens = max_tokens or config.coder_max_tokens
    base_url = _get_ollama_base_url()
    #_debug_llm_config("coder_response", selected_model, selected_max_tokens, base_url)

    return LLM(
        model=selected_model,
        base_url=base_url,
        api_key="ollama",
        temperature=0.2,
        max_completion_tokens=selected_max_tokens,
    )
