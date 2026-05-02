import os

def env_flag(name: str, default: bool = False) -> bool:
    raw_value = os.getenv(name)
    if raw_value is None:
        return default
    return raw_value.strip().lower() in {"1", "true", "yes", "y", "on"}


def crew_verbose(default: bool = False) -> bool:
    return env_flag("QCYBER_CREW_VERBOSE", default=default)


def default_crew_enabled() -> bool:
    env = (
        os.getenv("QCYBER_ENV")
        or os.getenv("APP_ENV")
        or os.getenv("ENV_MODE")
        or os.getenv("ENVIRONMENT")
        or "DEV"
    ).strip().upper()
    return env != "TEST"
