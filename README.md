## Setup
### Instalar o uv
```
# Linux e MacOS
curl -LsSf https://astral.sh/uv/install.sh | sh
```

```
# On Windows.
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
```

### Configurando o ambiente (.venv) com uv

1- Criar um ambiente virtual (python-version: >=3.10,< 3.14)
```
uv venv --python=3.13
```

2- Entrar no ambiente recém criado
```
# Linux e MacOS
source .venv/bin/activate
```

```
# On Windows.
.venv/Scripts/activate
```

3- Sincronizar as bibliotecas
```
uv sync
```

### Estrutura do .env:

```
# Para o SGBD
MYSQL_ROOT_PASSWORD=
MYSQL_USER=
MYSQL_PASSWORD=

# Para as APIs
APP_MYSQL_USER=
APP_MYSQL_DB=
APP_SECRET_KEY=
APP_MYSQL_PORT=
```