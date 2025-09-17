# main.py (agora como uma API Flask)
import json
import re
import pandas as pd
from pathlib import Path
import random
from dotenv import load_dotenv
from flask import Flask, request, jsonify
from datetime import datetime
from qml.api.database_api import get_db_connection, get_lookup_ids


# Importe suas classes da crewai
from qml.crew import CyberPredict
from qml.crew_multiclass import CyberPredictMult



# --- CONFIGURAÇÃO DO FLASK E BANCO DE DADOS ---
ROOT_DIR = Path(__file__).resolve().parents[3]
app = Flask(__name__)
load_dotenv(ROOT_DIR/'.env')


def extract_json_from_string(text):
    """
    Encontra e decodifica o primeiro JSON válido encontrado dentro de uma string.
    """
    # Expressão regular para encontrar um objeto JSON na string
    match = re.search(r'\{.*\}', text, re.DOTALL)
    if match:
        try:
            return json.loads(match.group(0))
        except json.JSONDecodeError:
            return None
    return None

# --- O ENDPOINT DA API DE ANÁLISE ---
@app.route('/analisar', methods=['POST'])
def analisar_pacote():
    """
    Recebe dados de um "pacote", executa a análise com a crew e salva o resultado no banco.
    """
    print(f"\n[{datetime.now()}] Nova requisição recebida em /analisar...")
    
    # 1. Valida os dados de entrada
    data = request.get_json()
    if not data or 'device_id' not in data or 'features' not in data:
        return jsonify({"error": "Payload inválido. 'device_id' e 'features' são obrigatórios."}), 400

    device_id = data['device_id']
    features = data['features']
    
    try:
        # 2. Prepara os dados para a crewai
        x_test = pd.DataFrame([features])
        input_records = x_test.to_dict(orient='records')
        inputs_for_crew = {'samples': input_records}

        # 3. Executa a análise da crew (binária e multiclasse)
        # NOTA: A sua crew binária parece não ser mais necessária se o CSV só tem ataques,
        # mas mantive a lógica caso você use outros dados no futuro.
        result_bin = CyberPredict().crew().kickoff(inputs=inputs_for_crew)
        bin_output = extract_json_from_string(result_bin.raw)
        
        if not bin_output or bin_output.get("predictions", [0])[0] == 0:
            print("➡️ Evento não classificado como ataque. Nenhuma ação tomada.")
            return jsonify({"status": "ignorado", "reason": "Não é um ataque"}), 200

        result_mult = CyberPredictMult().crew().kickoff(inputs=inputs_for_crew)
        mult_output = extract_json_from_string(result_mult.raw)

        if not mult_output or "predictions" not in mult_output:
            print("⚠️ A análise multiclasse falhou ou não retornou predições.")
            return jsonify({"status": "falha", "reason": "Análise multiclasse não retornou predições"}), 500
            
        tipo_ataque_id = mult_output["predictions"][0]
        print(f"✅ Análise concluída. Ataque tipo: {tipo_ataque_id} para o dispositivo: {device_id}")

        # 4. Salva a detecção no banco de dados
        conn = get_db_connection()
        with conn.cursor() as cursor:
            status_resp_ids = get_lookup_ids(cursor, "enum_status_resposta")
            status_resposta = random.choice(['Bloqueado', 'Monitorado', 'Permitido'])

            sql = """
            INSERT INTO deteccoes (data_deteccao, dispositivo_id, tipo_ataque_id, status_resposta_id, incidente_id)
            VALUES (%s, %s, %s, %s, NULL)
            """
            cursor.execute(sql, (datetime.now(), device_id, tipo_ataque_id, status_resp_ids[status_resposta]))
            new_detection_id = cursor.lastrowid
            conn.commit()
        
        conn.close()
        
        print(f"💾 Detecção #{new_detection_id} salva no banco de dados.")
        return jsonify({"status": "sucesso", "detection_id": new_detection_id, "tipo_ataque": tipo_ataque_id}), 201

    except Exception as e:
        print(f"❌ ERRO GERAL DURANTE A ANÁLISE: {e}")
        return jsonify({"error": "Ocorreu um erro interno no servidor de análise."}), 500


@app.route('/teste', methods=['GET'])
def teste():
    print(f"\n[{datetime.now()}] Requisição de teste recebida em /teste...")
    return jsonify({"status": "API está funcionando!"}), 200


# --- INICIA O SERVIDOR FLASK ---
if __name__ == "__main__":
    # O threaded=True ajuda a lidar com múltiplas conexões de forma mais estável
    # Em produção, você usaria um servidor WSGI como Gunicorn ou Waitress
    app.run(host='0.0.0.0', port=5000, debug=True, threaded=True)
