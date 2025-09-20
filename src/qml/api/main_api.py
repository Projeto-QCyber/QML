# main.py (agora como uma API Flask)
import json
import re
import os
import random
import pandas as pd
import pymysql
from dotenv import load_dotenv
from flask import Flask, request, jsonify
from datetime import datetime

# Importe suas classes da crewai
from qml.crew import CyberPredict
from qml.crew_multiclass import CyberPredictMult
from qml.response import IncidentResponseCrew
from qml.tools.shap_explain import ExplainTop2SHAP

# --- CONFIGURAÇÃO DO FLASK E BANCO DE DADOS ---
app = Flask(__name__)
load_dotenv()

def resolve_tipo_ataque_id(cursor, tipo_ataque_label):
    """
    Resolve o ID do tipo de ataque no banco de dados baseado no label.
    Retorna None se o tipo não for encontrado.
    """
    if not tipo_ataque_label or tipo_ataque_label.lower() == "normal":
        return None
    try:
        # Busca o ID baseado no nome do tipo de ataque
        cursor.execute("SELECT id FROM tipo_ataque WHERE nome = %s", (tipo_ataque_label,))
        result = cursor.fetchone()
        return result['id'] if result else None
    except Exception as e:
        print(f"❌ Erro ao resolver ID do tipo de ataque '{tipo_ataque_label}': {e}")
        return None


def extract_first_int_list_from_text(text):
    """
    Procura pela primeira lista de inteiros no formato [0, 1, 0, ...] dentro do texto
    e retorna como lista[int]. Retorna [] se não encontrar.
    """
    if not text:
        return []
    m = re.search(r"\[(?:\s*[-+]?\d+\s*,?\s*)+\]", text)
    if not m:
        return []
    content = m.group(0)
    try:
        arr = json.loads(content)
        if isinstance(arr, list):
            return [int(x) for x in arr]
    except Exception:
        # fallback to manual split
        nums = re.findall(r"[-+]?\d+", content)
        try:
            return [int(x) for x in nums]
        except Exception:
            return []
    return []



def get_db_connection():
    """Cria e retorna uma nova conexão com o banco para cada requisição."""
    return pymysql.connect(
        host=os.getenv('MYSQL_HOST', '192.168.1.87'),
        user=os.getenv('MYSQL_USER', 'root'),
        password=os.getenv('MYSQL_PASSWORD', 'root'),
        database=os.getenv('MYSQL_DB', 'qcyberDB'),
        cursorclass=pymysql.cursors.DictCursor
    )

def get_lookup_ids(cursor, table_name):
    """Busca IDs de uma tabela de lookup."""
    cursor.execute(f"SELECT id, nome FROM {table_name}")
    return {row['nome']: row['id'] for row in cursor.fetchall()}

def extract_json_from_string(text):
    """
    Encontra e decodifica um JSON válido dentro de uma string.
    Tenta, nesta ordem:
    - Blocos cercados por ``` ``` (eventualmente com marcador json)
    - Todos os objetos { ... } não gulosos e retorna o primeiro que contenha a chave
      "predictions" ou "votes".
    - Caso não encontre, retorna o último objeto JSON válido encontrado.
    """
    if not text:
        return None

    # 1) Tenta capturar bloco de código markdown ```json ... ```
    code_block = re.search(r"```(?:json)?\s*(.*?)```", text, re.DOTALL | re.IGNORECASE)
    if code_block:
        candidate = code_block.group(1).strip()
        try:
            obj = json.loads(candidate)
            if isinstance(obj, dict):
                return obj
        except json.JSONDecodeError:
            pass

    # 2) Tenta capturar múltiplos objetos { ... } de forma não gulosa e achar um com 'predictions'/'votes'
    candidates = re.findall(r"\{.*?\}", text, re.DOTALL)
    for cand in candidates:
        try:
            obj = json.loads(cand)
            if isinstance(obj, dict) and ("predictions" in obj or "votes" in obj):
                return obj
        except json.JSONDecodeError:
            continue

    # 3) Fallback: retorna o último objeto JSON válido encontrado
    for cand in reversed(candidates):
        try:
            obj = json.loads(cand)
            if isinstance(obj, dict):
                return obj
        except json.JSONDecodeError:
            continue

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
    if not data or 'device_id' not in data:
        return jsonify({"error": "Payload inválido. 'device_id' é obrigatório."}), 400

    device_id = data['device_id']
    features = data.get('features')
    samples_payload = data.get('samples')  # lista de dicionários opcional
    
    try:
        # 2. Prepara os dados para a crewai
        if samples_payload and isinstance(samples_payload, list):
            # Recebemos múltiplas amostras já no payload
            x_test = pd.DataFrame(samples_payload)
            inputs_for_bin_crew = {"samples": samples_payload}
            inputs_for_mult_crew = {"samples": samples_payload}
        else:
            if not isinstance(features, dict):
                return jsonify({"error": "Payload inválido. Envie 'samples' (lista) ou 'features' (objeto)."}), 400
            x_test = pd.DataFrame([features])
            input_records_single = x_test.to_dict(orient='records')  # lista com 1 dict
            # A crew binária exige exatamente 5 amostras, então duplicamos a mesma amostra.
            inputs_for_bin_crew = {'samples': input_records_single * 5}
            # A crew multiclasse usa 1 amostra.
            inputs_for_mult_crew = {'samples': input_records_single}

        # 3. Executa a análise da crew (binária e multiclasse)
        # NOTA: A sua crew binária parece não ser mais necessária se o CSV só tem ataques,
        # mas mantive a lógica caso você use outros dados no futuro.
        result_bin = CyberPredict().crew().kickoff(inputs=inputs_for_bin_crew)
        bin_output = extract_json_from_string(result_bin.raw)
        
        preds = []
        if isinstance(bin_output, dict):
            preds = bin_output.get("predictions") or bin_output.get("votes") or []
        if not preds:
            # Fallback: tenta extrair a primeira lista de inteiros do texto bruto
            preds = extract_first_int_list_from_text(getattr(result_bin, "raw", str(result_bin)))
        print(f"[BINÁRIO] Predições extraídas: {preds}")
        # Se múltiplas amostras foram enviadas, considere ataque se qualquer for 1
        is_attack = any(int(p) == 1 for p in preds) if preds else False
        if not is_attack:
            print("➡️ Evento não classificado como ataque. Nenhuma ação tomada.")
            return jsonify({"status": "ignorado", "reason": "Não é um ataque"}), 200

        # 3.1. Compute SHAP Top-2 lines for the samples to guide the leader's report
        shap_lines = ""
        try:
            explainer = ExplainTop2SHAP(classification="multiclass")
            shap_lines = explainer._run(inputs_for_mult_crew['samples'])
            print("[SHAP] Top-2 lines computed for leader context.")
        except Exception as e:
            print(f"⚠️ Falha ao computar SHAP Top-2: {e}")

        # 3.2. Executa a crew multiclasse com SHAP como contexto adicional para o líder
        mult_inputs = {**inputs_for_mult_crew, "shap_explain": shap_lines}
        result_mult = CyberPredictMult().crew().kickoff(inputs=mult_inputs)
        mult_output = extract_json_from_string(result_mult.raw)

        if not mult_output or "predictions" not in mult_output:
            print("⚠️ A análise multiclasse falhou ou não retornou predições.")
            return jsonify({"status": "falha", "reason": "Análise multiclasse não retornou predições"}), 500
        
        # Pode haver múltiplas predições; selecionamos a primeira para persistência
        tipo_ataque_label = mult_output["predictions"][0]
        # Explicação do líder (multiclasse) com fallback para binário
        explanation_text = None
        if isinstance(mult_output, dict):
            explanation_text = mult_output.get("report")
        if not explanation_text and isinstance(bin_output, dict):
            explanation_text = bin_output.get("report")
        if not explanation_text:
            explanation_text = "Sem explicação fornecida pelo líder."
        # Anexa um apêndice com as linhas SHAP (resumo técnico) para auditoria
        if shap_lines:
            explanation_text = (explanation_text or "") + "\n\nSHAP Top-2 (por amostra):\n" + shap_lines

        # Plano de resposta ao incidente (markdown)
        try:
            ir_result = IncidentResponseCrew().crew().kickoff(inputs={"attack_type": tipo_ataque_label})
            incident_plan = getattr(ir_result, "raw", str(ir_result))
        except Exception:
            incident_plan = ""

        print(f"✅ Análise concluída. Ataque tipo: {tipo_ataque_label} para o dispositivo: {device_id}")

        # 4. Salva a detecção no banco de dados
        conn = get_db_connection()
        with conn.cursor() as cursor:
            status_resp_ids = get_lookup_ids(cursor, "enum_status_resposta")
            # Tenta selecionar um status conhecido; caso contrário, usa o primeiro disponível.
            preferred = ['Monitorado', 'Bloqueado', 'Permitido']
            status_resposta_id = next((status_resp_ids[n] for n in preferred if n in status_resp_ids), None)
            if status_resposta_id is None and status_resp_ids:
                status_resposta_id = next(iter(status_resp_ids.values()), None)

            # Resolve FK do tipo de ataque, se existir mapeamento no banco.
            tipo_ataque_fk = resolve_tipo_ataque_id(cursor, tipo_ataque_label)

            sql = """
            INSERT INTO deteccoes (data_deteccao, dispositivo_id, tipo_ataque_id, status_resposta_id, incidente_id)
            VALUES (%s, %s, %s, %s, NULL)
            """
            cursor.execute(sql, (datetime.now(), device_id, tipo_ataque_fk, status_resposta_id))
            new_detection_id = cursor.lastrowid
            conn.commit()
        
        conn.close()
        
        print(f"💾 Detecção #{new_detection_id} salva no banco de dados.")
        return jsonify({
            "status": "sucesso",
            "detection_id": new_detection_id,
            "binary_votes": preds,
            "tipo_ataque": tipo_ataque_label,
            "tipo_ataque_id": tipo_ataque_fk,
            "explanation": explanation_text,
            "actions": incident_plan,
            "shap_top2": shap_lines
        }), 201

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



