# main.py (agora como uma API Flask)
import json
import re
import os
import random
import traceback
from pprint import pformat
import pandas as pd
import warnings
import pymysql
from dotenv import load_dotenv
from flask import Flask, request, jsonify
from datetime import datetime

# Importe suas classes da crewai
from qml.crew import CyberPredict
from qml.crew_multiclass import CyberPredictMult
from qml.response import IncidentResponseCrew

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
    req_id = f"REQ-{datetime.now().strftime('%Y%m%d-%H%M%S')}-{random.randint(1000,9999)}"
    start_ts = datetime.now()
    print(f"\n[{start_ts}] [{req_id}] Nova requisição recebida em /analisar de {request.remote_addr}", flush=True)
    
    # 1. Valida os dados de entrada
    data = request.get_json(silent=True)
    try:
        keys_info = list(data.keys()) if isinstance(data, dict) else str(type(data))
        print(f"[{req_id}] Payload bruto recebido (chaves ou tipo): {keys_info}", flush=True)
        if isinstance(data, dict):
            # Evita logar dados muito grandes: mostra no máx 5 campos e 200 chars por valor
            preview = {k: (str(v)[:200] + ('…' if len(str(v)) > 200 else '')) for i, (k, v) in enumerate(data.items()) if i < 5}
            print(f"[{req_id}] Payload preview: {pformat(preview)}", flush=True)
    except Exception as e:
        print(f"[{req_id}] Falha ao inspecionar payload: {e}", flush=True)
    
    if not data or 'device_id' not in data:
        print(f"[{req_id}] Payload inválido: ausente ou sem 'device_id'", flush=True)
        return jsonify({"error": "Payload inválido. 'device_id' é obrigatório."}), 400

    device_id = data['device_id']
    features = data.get('features')
    samples_payload = data.get('samples')  # lista de dicionários opcional
    print(f"[{req_id}] device_id={device_id} | features_tipo={type(features)} | samples_tipo={type(samples_payload)}", flush=True)
    
    try:
        # 2. Prepara os dados para a crewai
        if samples_payload and isinstance(samples_payload, list):
            # Recebemos múltiplas amostras já no payload
            x_test = pd.DataFrame(samples_payload)
            inputs_for_bin_crew = {"samples": samples_payload}
            inputs_for_mult_crew = {"samples": samples_payload}
            try:
                first_keys = list(samples_payload[0].keys()) if samples_payload else []
                print(f"[{req_id}] Modo multi-amostras: {len(samples_payload)} amostras | primeiras chaves: {first_keys[:10]}", flush=True)
            except Exception as e:
                print(f"[{req_id}] Falha ao inspecionar amostras: {e}", flush=True)
        else:
            if not isinstance(features, dict):
                return jsonify({"error": "Payload inválido. Envie 'samples' (lista) ou 'features' (objeto)."}), 400
            x_test = pd.DataFrame([features])
            input_records_single = x_test.to_dict(orient='records')  # lista com 1 dict
            # A crew binária exige exatamente 5 amostras, então duplicamos a mesma amostra.
            inputs_for_bin_crew = {'samples': input_records_single * 5}
            # A crew multiclasse usa 1 amostra.
            inputs_for_mult_crew = {'samples': input_records_single}
            try:
                print(f"[{req_id}] Modo single-feature: {len(features.keys())} chaves | duplicando para 5 amostras para binário", flush=True)
            except Exception:
                print(f"[{req_id}] Modo single-feature: não foi possível contar chaves de 'features'", flush=True)

        try:
            print(f"[{req_id}] x_test shape={x_test.shape} | bin_samples={len(inputs_for_bin_crew.get('samples', []))} | mult_samples={len(inputs_for_mult_crew.get('samples', []))}", flush=True)
        except Exception as e:
            print(f"[{req_id}] Falha ao inspecionar x_test ou inputs: {e}", flush=True)

        # 3. Executa a análise da crew (binária e multiclasse)
        # NOTA: A sua crew binária parece não ser mais necessária se o CSV só tem ataques,
        # mas mantive a lógica caso você use outros dados no futuro.
        print(f"[{req_id}] [BIN] Iniciando crew binária com {len(inputs_for_bin_crew.get('samples', []))} amostras...", flush=True)
        result_bin = CyberPredict().crew().kickoff(inputs=inputs_for_bin_crew)
        print(f"[{req_id}] [BIN] Crew binária finalizada. Tipo de retorno: {type(result_bin)}", flush=True)
        try:
            raw_text_bin = getattr(result_bin, "raw", str(result_bin))
            print(f"[{req_id}] [BIN] Saída bruta (até 1000 chars): {raw_text_bin[:1000]}", flush=True)
        except Exception as e:
            print(f"[{req_id}] [BIN] Falha ao obter saída bruta: {e}", flush=True)
        bin_output = extract_json_from_string(getattr(result_bin, "raw", str(result_bin)))
        print(f"[{req_id}] [BIN] JSON extraído: {list(bin_output.keys()) if isinstance(bin_output, dict) else type(bin_output)}", flush=True)
        
        preds = []
        if isinstance(bin_output, dict):
            preds = bin_output.get("predictions") or bin_output.get("votes") or []
        if not preds:
            # Fallback: tenta extrair a primeira lista de inteiros do texto bruto
            preds = extract_first_int_list_from_text(getattr(result_bin, "raw", str(result_bin)))
        print(f"[{req_id}] [BIN] Predições extraídas: {preds}", flush=True)
        # Se múltiplas amostras foram enviadas, considere ataque se qualquer for 1
        is_attack = any(int(p) == 1 for p in preds) if preds else False
        if not is_attack:
            print(f"[{req_id}] ➡️ Evento não classificado como ataque. Nenhuma ação tomada.", flush=True)
            return jsonify({"status": "ignorado", "reason": "Não é um ataque"}), 200

        # 3.1. Executa a crew multiclasse sem SHAP (colocado de lado por ora)
        mult_inputs = inputs_for_mult_crew
        print(f"[{req_id}] [MULT] Iniciando crew multiclasse; campos no input: {list(mult_inputs.keys())}", flush=True)
        result_mult = CyberPredictMult().crew().kickoff(inputs=mult_inputs)
        print(f"[{req_id}] [MULT] Crew multiclasse finalizada. Tipo de retorno: {type(result_mult)}", flush=True)
        try:
            raw_text_mult = getattr(result_mult, "raw", str(result_mult))
            print(f"[{req_id}] [MULT] Saída bruta (até 1000 chars): {raw_text_mult[:1000]}", flush=True)
        except Exception as e:
            print(f"[{req_id}] [MULT] Falha ao obter saída bruta: {e}", flush=True)
        mult_output = extract_json_from_string(getattr(result_mult, "raw", str(result_mult)))
        print(f"[{req_id}] [MULT] JSON extraído: {list(mult_output.keys()) if isinstance(mult_output, dict) else type(mult_output)}", flush=True)

        if not mult_output or "predictions" not in mult_output:
            try:
                keys = list(mult_output.keys()) if isinstance(mult_output, dict) else str(type(mult_output))
            except Exception:
                keys = 'desconhecido'
            print(f"[{req_id}] ⚠️ A análise multiclasse falhou ou não retornou predições. keys/tipo={keys}", flush=True)
            return jsonify({"status": "falha", "reason": "Análise multiclasse não retornou predições"}), 500
        
        # Pode haver múltiplas predições; selecionamos a primeira para persistência
        tipo_ataque_label = mult_output["predictions"][0]
        print(f"[{req_id}] [MULT] Label de ataque selecionado: {tipo_ataque_label}", flush=True)
        # Explicação do líder (multiclasse) com fallback para binário
        explanation_text = None
        if isinstance(mult_output, dict):
            explanation_text = mult_output.get("report")
        if not explanation_text and isinstance(bin_output, dict):
            explanation_text = bin_output.get("report")
        if not explanation_text:
            explanation_text = "Sem explicação fornecida pelo líder."
        print(f"[{req_id}] [MULT] Tamanho do relatório do líder: {len(explanation_text or '')}", flush=True)

        # Plano de resposta ao incidente (markdown)
        try:
            print(f"[{req_id}] [IR] Iniciando geração do plano de resposta para '{tipo_ataque_label}'", flush=True)
            ir_result = IncidentResponseCrew().crew().kickoff(inputs={"attack_type": tipo_ataque_label})
            incident_plan = getattr(ir_result, "raw", str(ir_result))
            print(f"[{req_id}] [IR] Plano de resposta gerado (até 800 chars):\n{incident_plan[:800]}", flush=True)
        except Exception:
            print(f"[{req_id}] ⚠️ Falha ao gerar plano de resposta. Prosseguindo sem ações.\n{traceback.format_exc()}", flush=True)
            incident_plan = ""

        print(f"[{req_id}] ✅ Análise concluída. Ataque tipo: {tipo_ataque_label} para o dispositivo: {device_id}", flush=True)

        # 4. Salva a detecção no banco de dados
        try:
            db_host = os.getenv('MYSQL_HOST', '192.168.1.87')
            db_user = os.getenv('MYSQL_USER', 'root')
            db_name = os.getenv('MYSQL_DB', 'qcyberDB')
            print(f"[{req_id}] [DB] Conectando ao MySQL host={db_host} user={db_user} db={db_name}", flush=True)
        except Exception:
            pass
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
            try:
                print(f"[{req_id}] [DB] Status resposta mapeados: {len(status_resp_ids)} | escolhido_id={status_resposta_id}", flush=True)
                print(f"[{req_id}] [DB] tipo_ataque_label='{tipo_ataque_label}' -> fk_id={tipo_ataque_fk}", flush=True)
            except Exception:
                pass

            sql = """
            INSERT INTO deteccoes (data_deteccao, dispositivo_id, tipo_ataque_id, status_resposta_id, incidente_id)
            VALUES (%s, %s, %s, %s, NULL)
            """
            cursor.execute(sql, (datetime.now(), device_id, tipo_ataque_fk, status_resposta_id))
            new_detection_id = cursor.lastrowid
            conn.commit()
        
        conn.close()
        
        print(f"[{req_id}] 💾 Detecção #{new_detection_id} salva no banco de dados.", flush=True)
        try:
            duration = (datetime.now() - start_ts).total_seconds()
            print(f"[{req_id}] ⏱️ Duração total da requisição: {duration:.2f}s", flush=True)
        except Exception:
            pass
        return jsonify({
            "status": "sucesso",
            "detection_id": new_detection_id,
            "binary_votes": preds,
            "tipo_ataque": tipo_ataque_label,
            "tipo_ataque_id": tipo_ataque_fk,
            "explanation": explanation_text,
            "actions": incident_plan
        }), 201

    except Exception as e:
        print(f"[{req_id}] ❌ ERRO GERAL DURANTE A ANÁLISE: {e}\n{traceback.format_exc()}", flush=True)
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



