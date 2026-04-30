# QCyber/QML - Guia operacional

Este guia mostra como executar o projeto em dois modos:

1. Pelo terminal, para testar o workflow multiagente diretamente.
2. Como sistema completo, com API Flask, MySQL e Ollama via Docker Compose.

O `README.md` original continua sendo o guia curto de setup. Este arquivo e uma versao mais operacional, pensada para rodar, validar logs, entender o fluxo e depurar latencia.

## Visao geral do fluxo

O ponto central do pipeline atual e o `BatchDetectionService`.

Fluxo executado:

1. Recebe uma janela de amostras.
2. Calcula predicao binaria e probabilidades com o modelo tradicional.
3. Opcionalmente chama a Crew binaria com `CyberPredict().crew().kickoff`.
4. Filtra somente as amostras aceitas como ataque.
5. Envia apenas essas amostras para o fluxo multiclasse.
6. Calcula probabilidades multiclasse.
7. Opcionalmente chama a Crew multiclasse com `CyberPredictMult().crew().kickoff`.
8. Se a confianca for baixa ou houver divergencia, marca a amostra para revisao especializada.
9. Se houver ataque aceito, chama a Crew de resposta a incidente.
10. Em seguida, gera uma sugestao inicial de remediacao para o operador.

O sistema trabalha com hipotese de rejeicao por confianca:

- `QCYBER_BINARY_MIN_CONFIDENCE`: limiar minimo da etapa binaria.
- `QCYBER_MULTICLASS_MIN_CONFIDENCE`: limiar minimo da etapa multiclasse.

Quando a confianca fica abaixo do limiar, a decisao ideal passa a ser revisao por especialista, e nao uma classificacao forcada.

## Pre-requisitos

Na maquina local:

- Python 3.12.
- `uv`.
- Docker e Docker Compose, se for subir o sistema completo.
- Ollama, se for rodar LLM local fora do Docker.
- Pesos/modelos tradicionais em `IA/weights/traditional`.
- Dataset de teste em `data/test/dados_de_teste.csv`.

Instale o `uv`:

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

Sincronize o ambiente:

```bash
uv sync
```

Em Linux, se houver erro com dependencias MySQL:

```bash
sudo apt update
sudo apt install python3-dev default-libmysqlclient-dev build-essential pkg-config
```

## Arquivo `.env`

Crie um arquivo `.env` na raiz do projeto:

```env
MYSQL_HOST=localhost
MYSQL_PORT=50000
MYSQL_ROOT_PASSWORD=root
MYSQL_DATABASE=qcyber_db
MYSQL_USER=
MYSQL_PASSWORD=

FLASK_API_PORT=5000
APP_SECRET_KEY=troque-este-valor

QCYBER_ENV=DEV
QCYBER_CREW_VERBOSE=0
QCYBER_ALLOW_CREW_FALLBACK=1
QCYBER_BINARY_MIN_CONFIDENCE=0.60
QCYBER_MULTICLASS_MIN_CONFIDENCE=0.55

LLM_MODEL_DEV_LLM_DEFAULT=ollama/qwen3:3b-q4_K_M
LLM_MODEL_DEV_LLM_LEADER=ollama/qwen3:8b-q4_K_M
LLM_MODEL_DEV_LLM_CODER=ollama/qwen3-coder:latest
QML_CODER_MODEL=ollama/qwen3-coder:latest

MAX_TOKENS_DEV_LLM_DEFAULT=256
MAX_TOKENS_DEV_LLM_LEADER=512
MAX_TOKENS_DEV_LLM_CODER=1024

QCYBER_SHAP_MODE=fast
```

Notas:

- Em `TEST`, o sistema usa modelos menores por padrao.
- Em `DEV`, o sistema usa configuracoes equilibradas para desenvolvimento.
- Em `PROD`, configure explicitamente os modelos `LLM_MODEL_PROD_*`.
- `QCYBER_CREW_VERBOSE=1` mostra logs completos da CrewAI.
- `QCYBER_CREW_VERBOSE=0` reduz ruido no backend.
- `QCYBER_ALLOW_CREW_FALLBACK=1` mantem o pipeline rodando com o modelo probabilistico quando algum LLM local falhar.
- `QCYBER_SHAP_MODE=fast` reduz custo de explicabilidade.

## Modelos Ollama

Para rodar localmente sem Docker:

```bash
ollama serve
```

Em outro terminal:

```bash
ollama pull qwen3:3b-q4_K_M
ollama pull qwen3:8b-q4_K_M
ollama pull qwen3-coder:latest
```

Se algum nome de modelo nao existir na sua instalacao do Ollama, ajuste o `.env` para um tag disponivel localmente.

Para verificar os modelos instalados:

```bash
ollama list
```

## Execucao pelo terminal

Use este modo para testar o workflow multiagente sem subir a API.

Entre na raiz do projeto:

```bash
cd "/mnt/data/Area de trabalho/Faculdade/BioData/QCyber/QML"
```

Execute o fluxo principal:

```bash
uv run run_crew
```

Alternativa com o comando da CrewAI:

```bash
uv run crewai run
```

O script executado e `qml.main:run`, definido em `pyproject.toml`.

Por padrao, o `main.py` habilita logs verbosos no terminal:

```python
os.environ.setdefault("QCYBER_CREW_VERBOSE", "1")
```

Isso e util para confirmar que os agentes estao rodando de verdade. Para testes mais silenciosos, rode:

```bash
QCYBER_CREW_VERBOSE=0 uv run run_crew
```

O comando carrega amostras de:

```text
data/test/dados_de_teste.csv
```

A selecao atual e deterministica, com `random_state`. Portanto, a cada rodada, as amostras devem ser as mesmas enquanto o arquivo de dados nao mudar.

## O que observar na saida do terminal

A saida final deve ser um JSON contendo campos como:

```json
{
  "binary_predictions": [1, 0, 1],
  "binary_decisions": [],
  "window_has_attack": true,
  "attack_indices": [0, 2],
  "review_indices": [],
  "classified_attacks": [],
  "primary_attack_type_id": 0,
  "primary_attack_type_label": "Backdoor",
  "incident_response_status": "generated",
  "remediation_suggestion_status": "generated",
  "workflow_trace": []
}
```

Os campos mais importantes para validacao sao:

- `binary_decisions`: mostra probabilidades, confianca e status da etapa binaria.
- `attack_indices`: indices das amostras aceitas como ataque.
- `review_indices`: amostras encaminhadas para especialista por baixa confianca.
- `classified_attacks`: classificacoes multiclasse com probabilidades.
- `incident_response_status`: informa se a Crew de resposta gerou plano.
- `remediation_suggestion_status`: informa se a Crew de remediacao gerou sugestao.
- `workflow_trace`: mostra quais etapas foram executadas.

Se `attack_indices` estiver vazio, o fluxo multiclasse, resposta a incidente e remediacao nao devem rodar. Isso e esperado.

Nao e necessario rodar um `response.py` separado. A resposta a incidente ja e chamada pelo `BatchDetectionService` por meio da Crew definida em `src/qml/crew_response.py`.

## Subindo o sistema completo com Docker

O `docker-compose.yml` sobe tres servicos:

- `mysql`: banco de dados.
- `ollama`: servidor local de modelos.
- `flask-api`: API do QCyber.

Suba tudo:

```bash
docker compose up --build
```

Para rodar em segundo plano:

```bash
docker compose up --build -d
```

Na primeira execucao, o container do Ollama baixa automaticamente o modelo `qwen3:8b-q4_K_M`. Se o `.env` estiver usando tambem `qwen3:3b-q4_K_M` e `qwen3-coder:latest`, baixe os modelos adicionais em outro terminal:

```bash
docker exec -it qcyber_ollama ollama pull qwen3:3b-q4_K_M
docker exec -it qcyber_ollama ollama pull qwen3-coder:latest
```

Para acompanhar logs:

```bash
docker compose logs -f flask-api
```

Logs do Ollama:

```bash
docker compose logs -f ollama
```

Logs do MySQL:

```bash
docker compose logs -f mysql
```

Parar o sistema:

```bash
docker compose down
```

Parar e remover volumes nomeados criados pelo Compose:

```bash
docker compose down -v
```

Os dados deste projeto usam diretorios bindados, como `./mysql_volume` e `./ollama`. Portanto, `docker compose down -v` nao deve apagar esses diretorios automaticamente. Para resetar completamente banco ou modelos, remova esses diretorios manualmente apenas quando tiver certeza.

## API Flask

Com `FLASK_API_PORT=5000`, a API fica disponivel em:

```text
http://localhost:5000
```

### Endpoint `/analisar`

Executa o pipeline de deteccao e salva o resultado no banco.

Exemplo com uma amostra:

```bash
curl -X POST "http://localhost:5000/analisar" \
  -H "Content-Type: application/json" \
  -d '{
    "device_id": "sensor-lab-01",
    "features": {
      "PC_1": 0.8270502297613899,
      "PC_2": 0.613100973637563,
      "PC_3": 0.7984897286746298,
      "PC_4": 0.743170600311643,
      "PC_5": 0.7982729912934826,
      "PC_6": 0.5149336760525176,
      "PC_7": 0.8308746906501435,
      "PC_8": 0.3541790597061432
    }
  }'
```

Exemplo com batch:

```bash
curl -X POST "http://localhost:5000/analisar" \
  -H "Content-Type: application/json" \
  -d '{
    "device_id": "sensor-lab-01",
    "samples": [
      {
        "PC_1": 0.8270502297613899,
        "PC_2": 0.613100973637563,
        "PC_3": 0.7984897286746298,
        "PC_4": 0.743170600311643,
        "PC_5": 0.7982729912934826,
        "PC_6": 0.5149336760525176,
        "PC_7": 0.8308746906501435,
        "PC_8": 0.3541790597061432
      },
      {
        "PC_1": 0.0037644404423902,
        "PC_2": 0.4651301042713207,
        "PC_3": 0.2492095790876915,
        "PC_4": 0.1894236823030872,
        "PC_5": 0.0565208158852915,
        "PC_6": 0.7440521634665223,
        "PC_7": 0.0211730390028705,
        "PC_8": 0.2646215222639519
      }
    ]
  }'
```

Resposta esperada:

```json
{
  "mensagem": "Detecção analisada e registrada com sucesso!",
  "deteccao_id": 1,
  "incidente_id": 1,
  "tipo_ataque_detectado": "Backdoor",
  "batch": {
    "predicoes_binarias": [1, 1],
    "indices_suspeitos": [0, 1],
    "ataques_classificados": []
  }
}
```

### Endpoint `/remediation/chat`

Gera uma resposta assistida para o operador com base no tipo de ataque.

```bash
curl -X POST "http://localhost:5000/remediation/chat" \
  -H "Content-Type: application/json" \
  -d '{
    "attack_label": "Backdoor",
    "operator_message": "Sugira acoes defensivas para conter esse incidente em um servidor Linux.",
    "context": {
      "device_id": "sensor-lab-01",
      "confidence": 0.82,
      "automatic_execution_requested": false
    }
  }'
```

A resposta deve vir em JSON com mensagem, comandos sugeridos, riscos, rollback e validacoes. O sistema foi desenhado para sugerir comandos defensivos. Execucao automatica ainda deve exigir aprovacao explicita do operador.

## Banco de dados

No Docker, o banco sobe com o servico `mysql`.

O entrypoint da API executa o bootstrap automaticamente:

```text
qml.api.create_qcyber_db.ensure_bootstrap()
```

Para conectar no MySQL do host:

```bash
docker exec -it qcyber_mysql mysql -uroot -proot qcyber_db
```

Listar tabelas:

```sql
SHOW TABLES;
```

Consultar deteccoes:

```sql
SELECT id, data_deteccao, dispositivo_id, tipo_ataque_id, status_resposta_id
FROM deteccoes
ORDER BY id DESC
LIMIT 10;
```

Consultar incidentes:

```sql
SELECT id, titulo, status_id, dispositivo_id, data_deteccao
FROM incidentes_analisados
ORDER BY id DESC
LIMIT 10;
```

## Arquivos de saida

Algumas Crews podem gravar artefatos em:

```text
src/qml/output/
```

Arquivos comuns:

- `incident_response_plan.md`
- `remediation_chat.json`
- arquivos de predicao preliminar/final, dependendo da Crew executada.

Esses arquivos sao auxiliares. A fonte principal do resultado operacional deve ser o JSON retornado pelo terminal ou pela API.

## Controle de logs

Para desenvolvimento no terminal:

```bash
QCYBER_CREW_VERBOSE=1 uv run run_crew
```

Para backend/API com menos ruido:

```env
QCYBER_CREW_VERBOSE=0
```

Para conferir quais modelos LLM foram selecionados:

```env
QCYBER_DEBUG_LLM_CONFIG=1
```

Com isso, o sistema imprime o ambiente, papel do LLM, modelo, token limit e `base_url`.

## Latencia e desempenho

O fluxo completo pode ser lento em maquina limitada porque envolve:

- Modelo tradicional binario.
- Crew binaria.
- Modelo tradicional multiclasse.
- Crew multiclasse.
- SHAP/explicabilidade.
- Crew de resposta a incidente.
- Crew de remediacao com modelo coder.

Para acelerar testes:

```env
QCYBER_ENV=TEST
QCYBER_CREW_VERBOSE=0
QCYBER_SHAP_MODE=fast
LLM_MODEL_TEST_LLM_DEFAULT=ollama/qwen3:3b-q4_K_M
LLM_MODEL_TEST_LLM_LEADER=ollama/qwen3:3b-q4_K_M
LLM_MODEL_TEST_LLM_CODER=ollama/qwen2.5-coder:7b
MAX_TOKENS_TEST_LLM_DEFAULT=128
MAX_TOKENS_TEST_LLM_LEADER=256
MAX_TOKENS_TEST_LLM_CODER=512
```

Para validar apenas a logica probabilistica sem custo de LLM, use o servico em Python com as flags de Crew desativadas:

```python
from qml.services.detection import BatchDetectionService

result = BatchDetectionService(
    use_binary_crew=False,
    use_multiclass_crew=False,
    use_incident_response_crew=False,
    use_remediation_crew=False,
).predict(samples)

print(result.to_dict())
```

## Troubleshooting

### `crewai run` nao mostra o pipeline esperado

Use o comando direto:

```bash
uv run run_crew
```

Ele chama `qml.main:run`, que executa o `BatchDetectionService` completo.

### A API parece travar

Verifique:

```bash
docker compose logs -f flask-api
docker compose logs -f ollama
```

As causas mais comuns sao:

- Ollama baixando modelo na primeira execucao.
- Modelo grande demais para a maquina.
- Crew multiclasse chamando muitos agentes.
- SHAP em modo completo.
- MySQL ainda inicializando.

### `Invalid response from LLM call - None or empty`

Esse erro indica que a CrewAI chamou o LLM local, mas o Ollama/modelo retornou resposta vazia antes de chamar a tool. Em maquinas limitadas isso pode ocorrer por timeout, modelo pesado, truncamento ou incompatibilidade fraca de tool calling.

Use:

```env
QCYBER_ALLOW_CREW_FALLBACK=1
QCYBER_ENV=TEST
QCYBER_CREW_VERBOSE=0
```

Com o fallback ativo, o pipeline continua usando as predicoes e probabilidades do modelo tradicional quando uma Crew falhar. O `workflow_trace` indicara `crew_failed_model_fallback` ou `crew_partial_model_fallback`.

### Erro de conexao com Ollama

Dentro do Docker, o sistema usa:

```text
http://ollama:11434
```

Fora do Docker, usa:

```text
http://localhost:11434
```

Teste:

```bash
curl http://localhost:11434/api/tags
```

### Erro de modelo nao encontrado

Liste os modelos:

```bash
ollama list
```

Baixe o modelo faltante:

```bash
ollama pull qwen3:8b-q4_K_M
```

Ou ajuste o `.env` para um modelo disponivel.

### Nenhum ataque passa para o multiclasse

Isso acontece quando:

- Todas as predicoes binarias sao `0`.
- As predicoes `1` ficaram abaixo de `QCYBER_BINARY_MIN_CONFIDENCE`.

Nesse caso, o sistema registra `review_indices` ou classifica a janela como normal, dependendo das probabilidades.

### A Crew multiclasse discorda do modelo probabilistico

Quando a Crew retorna uma classe diferente da classe do modelo tradicional, a amostra e marcada como:

```text
needs_specialist_review
```

Essa regra evita que o LLM sobrescreva a decisao probabilistica sem evidencias fortes.

## Comandos rapidos

Instalar dependencias:

```bash
uv sync
```

Rodar workflow no terminal:

```bash
uv run run_crew
```

Rodar workflow no terminal com menos logs:

```bash
QCYBER_CREW_VERBOSE=0 uv run run_crew
```

Subir sistema completo:

```bash
docker compose up --build
```

Subir sistema completo em background:

```bash
docker compose up --build -d
```

Ver logs da API:

```bash
docker compose logs -f flask-api
```

Parar containers:

```bash
docker compose down
```

Testar API:

```bash
curl -X POST "http://localhost:5000/remediation/chat" \
  -H "Content-Type: application/json" \
  -d '{"attack_label":"Backdoor","operator_message":"Sugira uma contencao inicial.","context":{}}'
```
