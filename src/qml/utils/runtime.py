import os

def env_flag(name: str, default: bool = False) -> bool:
    raw_value = os.getenv(name)
    if raw_value is None:
        return default
    return raw_value.strip().lower() in {"1", "true", "yes", "y", "on"}


def crew_verbose(default: bool = False) -> bool:
    return env_flag("QCYBER_CREW_VERBOSE", default=default)
