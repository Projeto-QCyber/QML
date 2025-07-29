### Setup
##### Instalar o uv
```
# Linux e MacOS
curl -LsSf https://astral.sh/uv/install.sh | sh
```

```
# On Windows.
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
```

##### Baixando dependencias e subindo APIs
1-
```
uv sync
```
2- Ativar ambiente virual
3-
```
cd notebooks/
```
4-
```
python API_***.py
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