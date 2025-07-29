import os


def get_env_var(var_name: str) -> str:
    if env_var := os.environ.get(var_name):
        return env_var
    raise Exception(f"Variável de ambiente ausente: {var_name}")