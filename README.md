## Setup
### 1. Instalar o uv
```
# Linux e MacOS
curl -LsSf https://astral.sh/uv/install.sh | sh
```

```
# On Windows.
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
```
<br>

### 2. Configurando o ambiente virtual Python (.venv) com uv

#### 2.1. Criar um ambiente virtual e sincronizar as bibliotecas (python-version: >=3.10,< 3.14 - vide arquivo [pyproject.toml](./pyproject.toml))

```bash
# "uv sync --no-dev" para produção!
uv sync
```
OBS: No linux, antes instale os requisitos para o mysqlclient:
```bash
sudo apt update
sudo apt install python3-dev default-libmysqlclient-dev build-essential pkg-config
```


#### 2.2. Ativar o ambiente recém criado
```bash
# Linux e MacOS
source .venv/bin/activate
```

No windows, é necessária a configuração da política de execução de scripts:
```bash
# Windows Powershell
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
```

```bash
# Windows Powershell
.venv\Scripts\activate.ps1
```

<br>

### 3. Criar arquivo .env com a seguinte estrutura:

```
# Para o SGBD
MYSQL_HOST=localhost
MYSQL_ROOT_PASSWORD=
MYSQL_DATABASE=qcyberDB
MYSQL_USER=
MYSQL_PASSWORD=

# Para as APIs
APP_SECRET_KEY=
```
<br>

### 4. Inicializar o projeto

Certifique-se de estar exatamente na raiz do projeto. Existem duas opções de você executar o fluxo:

```
# Primeira opção: 
crewai run

# Segunda opção:
# Executando diretamente o arquivo main.py

".../.venv/Scripts/python.exe" ".../QML/src/qml/main.py"
```