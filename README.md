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
MYSQL_PORT=3306
MYSQL_ROOT_PASSWORD=root
MYSQL_DATABASE=qcyber_db
MYSQL_USER=
MYSQL_PASSWORD=
FLASK_API_PORT=5000

# Para as APIs
APP_SECRET_KEY=
```
<br>

### 4. Configuração do Ollama
#### 4.1. Sem Docker

```bash
# Linux (Debian based)
curl -fsSL https://ollama.com/install.sh | sh
```

```bash
ollama run qwen3:4b
```

#### 4.2. Com Docker

##### 4.2.1. Instale o [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html);

##### 4.2.2. Subir container;
```bash
docker compose up ollama
```

##### 4.2.3. Rodar o Modelo;
```bash
docker exec -it qcyber_ollama ollama run qwen3:4b
```

<br>

### 5. Inicializar o projeto

Lembre-se de adicionar os pesos em IA/weights/traditional...
Além disso, certifique-se de estar exatamente na raiz do projeto. Existem duas opções de você executar o fluxo:

```bash
# Primeira opção: 
crewai run

# Segunda opção:
# Executando diretamente o arquivo main.py

".../.venv/Scripts/python.exe" ".../QML/src/qml/main.py"
```
