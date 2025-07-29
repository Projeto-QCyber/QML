#### PROTOTIPO PARA SIMULAR A CAPTURA DO DADOS E ENVIO PARA ANALISE

import requests
import json
import time
import mysql.connector
from mysql.connector import Error

# --- 1. CONFIGURAÇÕES ---

# URL do endpoint da nossa API Flask de predição
API_URL = "http://127.0.0.1:5000/predict"

# Intervalo de tempo entre as análises (em segundos)
INTERVALO_DE_TEMPO = 10

# Configurações do banco de dados MySQL
DB_CONFIG = {
    'host': 'localhost',
    'user': 'root',
    'password': 'root',
    'database': 'monitoramento_qml'
}

# Lista de dados para simular a leitura em intervalos.
dados_para_analise = [
    {"y": [1, -1], "X": [[0, 0, 1, 0], [0, 0, 1, 0]]},
    {"y": [1, -1], "X": [[0, 1, 0, 0], [0, 0, 0, 1]]},
    {"y": [1, -1], "X": [[1, 0, 0, 0], [0, 1, 0, 0]]},
    {"y": [1, -1], "X": [[0, 0, 0, 1], [0, 0, 1, 0]]},
    {"y": [1, -1], "X": [[1, 0, 0, 0], [0, 0, 1, 0]]}  # Adicionando um caso de erro
]


# --- 2. FUNÇÃO PARA SALVAR NO BANCO DE DADOS ---

def salvar_analise_no_banco(dados_analise):
    """
    Conecta-se ao banco de dados MySQL e insere os resultados da análise.
    """
    # SQL para inserir um novo registro na tabela
    query = """
        INSERT INTO qcyber_analises_vqc (acuracia, tp, tn, fp, fn, is_alerta) 
        VALUES (%s, %s, %s, %s, %s, %s)
    """

    # Extrai os dados do dicionário de análise
    acuracia_str = dados_analise.get('acuracia', '0.0%')
    # Converte a string '100.00%' para o float 1.00
    acuracia_float = float(acuracia_str.strip('%')) / 100.0

    tp = dados_analise.get('verdadeiros_positivos (previu 1, era 1)', 0)
    tn = dados_analise.get('verdadeiros_negativos (previu -1, era -1)', 0)
    fp = dados_analise.get('falsos_positivos (previu 1, era -1)', 0)
    fn = dados_analise.get('falsos_negativos (previu -1, era 1)', 0)

    # Lógica de negócio para definir um alerta
    # Exemplo: é um alerta se a acurácia for menor que 100%
    is_alerta = acuracia_float < 1.0

    # Prepara os valores para a inserção no banco
    valores = (acuracia_float, tp, tn, fp, fn, is_alerta)

    connection = None
    try:
        # Tenta conectar ao banco de dados
        connection = mysql.connector.connect(**DB_CONFIG)
        cursor = connection.cursor()

        # Executa a query com os valores
        cursor.execute(query, valores)
        connection.commit()

        print(f"Análise salva no banco de dados com sucesso! (ID: {cursor.lastrowid})")

    except Error as e:
        print(f"Erro ao salvar no banco de dados: {e}")

    finally:
        # Garante que a conexão seja fechada
        if connection and connection.is_connected():
            cursor.close()
            connection.close()


# --- 3. FUNÇÃO PRINCIPAL DO CLIENTE ---

def executar_analise_periodica():
    """
    Função que entra em um loop infinito para enviar dados para a API
    e depois salvar o resultado no banco de dados.
    """
    print(">>> Cliente iniciado. Enviando dados para análise a cada 10 segundos.")
    print(">>> Pressione CTRL+C para encerrar.")

    i = 0
    while True:
        # Pega o próximo item da lista (usando o operador de módulo para repetir a lista)
        payload = dados_para_analise[i % len(dados_para_analise)]

        print("-" * 50)
        print(f"[{time.ctime()}] Enviando dados para a API de Análise:")
        print(json.dumps(payload, indent=2))

        try:
            # Faz a requisição POST para a API, enviando os dados em formato JSON
            response = requests.post(API_URL, json=payload)
            response.raise_for_status()  # Lança um erro para status HTTP 4xx/5xx

            # Extrai o resultado da resposta
            resultado = response.json()
            print("\n<<< Resposta da API recebida:")
            print(json.dumps(resultado, indent=2))

            # [NOVO] Chama a função para salvar os dados no banco
            if 'analise' in resultado:
                salvar_analise_no_banco(resultado['analise'])

        except requests.exceptions.RequestException as e:
            print(f"\n ERRO: Não foi possível conectar à API de Análise. Verifique se o servidor está no ar.")
            print(f"   Detalhe do erro: {e}")

        # Incrementa o contador para pegar o próximo dado
        i += 1

        # Aguarda o intervalo de tempo definido
        print(f"\n>>> Aguardando {INTERVALO_DE_TEMPO} segundos para a próxima análise...")
        time.sleep(INTERVALO_DE_TEMPO)


# --- 4. EXECUÇÃO DO SCRIPT ---

if __name__ == "__main__":
    executar_analise_periodica()
