import os


def get_env_var(var_name: str, default:str=None) -> str:
    if env_var := os.environ.get(var_name):
        return env_var
    if default is not None:
        return default
    raise Exception(f"Variável de ambiente ausente: {var_name}")
