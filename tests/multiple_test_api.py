import os, json, random, time
import pandas as pd, requests
from dotenv import load_dotenv

load_dotenv()

API_URL = f"http://localhost:{os.getenv('FLASK_API_PORT', 5000)}/analisar"
MINUTES = 30

# 1. Carregamos o CSV apenas uma vez (fora do loop) para economizar recursos
print("Carregando base de dados...")
df = pd.read_csv("data/test/dados_de_teste.csv")

print(f"Iniciando envios para {API_URL} a cada {MINUTES} minutos. Pressione Ctrl+C para parar.")

# 2. Criamos um loop infinito
while True:
    try:
        print(f"\n[{time.strftime('%Y-%m-%d %H:%M:%S')}] Gerando amostra e enviando requisição...")
        
        # A lógica de amostragem fica dentro do loop para gerar dados novos a cada envio
        att = df[df.Attack_type != 0]
        first3 = att.sample(3, replace=len(att)<3)
        rest = df.drop(first3.index)
        last2 = rest.sample(2, replace=len(rest)<2)
        samples = pd.concat([first3, last2], ignore_index=True)
        assert (samples.Attack_type.iloc[:3] != 0).all()

        print("labels:", samples.Attack_type.tolist(),
              "flags:", ["safe" if x==0 else "attack" for x in samples.Attack_type])

        payload = {
            "device_id": f"test-device-{random.randint(100,999)}",
            "samples": samples.drop(columns=['Attack_type', 'Attack_label']).to_dict("records"),
        }

        # Envia a requisição
        r = requests.post(API_URL, json=payload)
        r.raise_for_status()
        print("Resposta da API:")
        print(json.dumps(r.json(), indent=2))

    except requests.exceptions.RequestException as e:
        # Se a API estiver offline ou der erro 500, o script avisa mas não quebra
        print(f"Erro de conexão com a API: {e}")
    except Exception as e:
        # Captura outros erros inesperados (como falha na amostragem)
        print(f"Ocorreu um erro inesperado: {e}")

    # 3. Pausa a execução por 10 minutos (600 segundos) antes de repetir
    print(f"Aguardando {MINUTES} minutos para o próximo envio...")
    time.sleep(MINUTES*60)
