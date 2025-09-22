import os


def get_env_var(var_name: str, default:str=None) -> str:
    if default:
        return default
    if env_var := os.environ.get(var_name):
        return env_var
    raise Exception(f"Variável de ambiente ausente: {var_name}")