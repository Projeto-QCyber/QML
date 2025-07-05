#### PROTOTIPO PARA SIMULAR A CAPTURA DO DADOS E ENVIO PARA ANALISE


import requests
import json
import time

# URL do endpoint da nossa API Flask
API_URL = "http://127.0.0.1:5000/predict"

# Lista de dados para simular a leitura em intervalos.
# Na vida real, isso poderia vir de um banco de dados, um sensor, um arquivo, etc.
dados_para_analise = [
    {"y": [1, -1], "X": [[0, 0, 1, 0], [0, 0, 1, 0]]},
    {"y": [1, -1], "X": [[0, 1, 0, 0], [0, 0, 0, 1]]},
    {"y": [1, -1], "X": [[1, 0, 0, 0], [0, 1, 0, 0]]},
    {"y": [1, -1], "X": [[0, 0, 0, 1], [0, 0, 1, 0]]}
]

# Intervalo de tempo entre as análises (em segundos)
INTERVALO_DE_TEMPO = 10


def executar_analise_periodica():
    """
    Função que entra em um loop infinito para enviar dados para a API
    em intervalos de tempo definidos.
    """
    print(">>> Cliente iniciado. Enviando dados para análise a cada 10 segundos.")
    print(">>> Pressione CTRL+C para encerrar.")

    i = 0
    while True:
        # Pega o próximo item da lista (usando o operador de módulo para repetir a lista)
        payload = dados_para_analise[i % len(dados_para_analise)]

        print("-" * 50)
        print(f"[{time.ctime()}] Enviando dados para a API:")
        print(json.dumps(payload, indent=2))

        try:
            # Faz a requisição POST para a API, enviando os dados em formato JSON
            response = requests.post(API_URL, json=payload)
            response.raise_for_status()  # Lança um erro para status HTTP 4xx/5xx

            # Imprime a resposta recebida da API
            resultado = response.json()
            print("\n<<< Resposta da API recebida:")
            print(json.dumps(resultado, indent=2))

        except requests.exceptions.RequestException as e:
            print(f"\nERRO: Não foi possível conectar à API. Verifique se o servidor está no ar.")
            print(f"Detalhe do erro: {e}")

        # Incrementa o contador para pegar o próximo dado
        i += 1

        # Aguarda o intervalo de tempo definido
        print(f"\n>>> Aguardando {INTERVALO_DE_TEMPO} segundos para a próxima análise...")
        time.sleep(INTERVALO_DE_TEMPO)


if __name__ == "__main__":
    executar_analise_periodica()