import json
import re
import os
import random
import traceback
from pprint import pformat
import pandas as pd
import pymysql
from dotenv import load_dotenv
from flask import Flask, request, jsonify
from datetime import datetime, timezone
from qml.api.create_qcyber_db import ensure_bootstrap
from qml.utils.generic import get_env_var

# Importe suas classes da crewai
from qml.crew import CyberPredict
from qml.crew_multiclass import CyberPredictMult
from qml.response import IncidentResponseCrew
from qml.tools.shap_explain import ExplainTop2SHAP
os.environ['CREWAI_DISABLE_TELEMETRY'] = 'true'
os.environ['OTEL_SDK_DISABLED'] = 'true'


app = Flask(__name__)
load_dotenv()

def resolve_tipo_ataque_id(cursor, tipo_ataque_label):
    """
    Resolve o tipo de ataque no banco com base no label informado e retorna:
    (id_inteiro, nome_canonico_do_enum)

    - Normaliza (case/espaços/underscore) o label recebido.
    - Se não encontrar, faz fallback para (99, 'Normal').
    """
    try:
        cursor.execute("SELECT id, nome FROM enum_tipo_ataque")
        rows = cursor.fetchall()

        def _norm(s):
            s = re.sub(r'[^A-Za-z0-9]+', '_', str(s)).strip('_').lower()
            return re.sub(r'_+', '_', s)

        # Mapas de apoio
        id_map = {int(r['id']): (int(r['id']), r['nome']) for r in rows}
        norm_map = {_norm(r['nome']): (int(r['id']), r['nome']) for r in rows}

        # 1) Se veio número (ou string numérica), tenta por ID
        if isinstance(tipo_ataque_label, (int, float)) or (isinstance(tipo_ataque_label, str) and tipo_ataque_label.isdigit()):
            try:
                val = int(tipo_ataque_label)
                if val in id_map:
                    return id_map[val]
            except Exception:
                pass

        # 2) Caso contrário, tenta por nome normalizado
        key = _norm(tipo_ataque_label) if tipo_ataque_label else 'normal'
        if key in norm_map:
            return norm_map[key]

        # 3) Fallback seguro
        if 'normal' in norm_map:
            return norm_map['normal']
        return (99, 'Normal')
    except Exception as e:
        print(f"❌ Erro ao resolver tipo_ataque '{tipo_ataque_label}': {e}")
        return (99, 'Normal')


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
    """Cria e retorna uma nova conexão com o banco para cada requisição.
    Usa variáveis de ambiente com fallbacks robustos e ignora valores vazios.
    """
    # Evita valores vazios vindos do .env (e.g., MYSQL_USER="")
    db_host = get_env_var('MYSQL_HOST', 'mysql')
    db_user = os.getenv('MYSQL_USER') or os.getenv('MYSQL_USERNAME') or 'root'
    db_pass = os.getenv('MYSQL_PASSWORD') or os.getenv('MYSQL_ROOT_PASSWORD') or ''
    db_name = os.getenv('MYSQL_DATABASE') or os.getenv('MYSQL_DATABASE') or 'qcyber_db'

    return pymysql.connect(
        host=db_host,
        user=db_user,
        password=db_pass,
        database=db_name,
        cursorclass=pymysql.cursors.DictCursor
    )

def get_lookup_ids(cursor, table_name):
    """Busca IDs de uma tabela de lookup."""
    cursor.execute(f"SELECT id, nome FROM {table_name}")
    return {row['nome']: row['id'] for row in cursor.fetchall()}

def ensure_dispositivo(cursor, device_id):
    """
    Garante e retorna o ID do dispositivo.
    - Se device_id for int: verifica por ID; se não existir, cria um placeholder com host 'device-<id>'.
    - Caso contrário, trata device_id como 'host' (string); cria se não existir.
    Usa status_id=1 ('Ativo') quando criar.
    """
    status_map = get_lookup_ids(cursor, "enum_status_dispositivo")
    status_ativo_id = status_map.get('Ativo') or next(iter(status_map.values()), None)

    if isinstance(device_id, int):
        cursor.execute("SELECT id FROM dispositivos WHERE id=%s", (device_id,))
        row = cursor.fetchone()
        if row:
            return row['id']
        host = f"device-{device_id}"
        nome = f"Device {device_id}"
        cursor.execute(
            "INSERT INTO dispositivos (nome, host, status_id) VALUES (%s, %s, %s)",
            (nome, host, status_ativo_id)
        )
        return cursor.lastrowid
    else:
        host = str(device_id)
        cursor.execute("SELECT id FROM dispositivos WHERE host=%s", (host,))
        row = cursor.fetchone()
        if row:
            return row['id']
        nome = host
        cursor.execute(
            "INSERT INTO dispositivos (nome, host, status_id) VALUES (%s, %s, %s)",
            (nome, host, status_ativo_id)
        )
        return cursor.lastrowid

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
    req_id = f"REQ-{datetime.now(timezone.utc).strftime('%Y%m%d-%H%M%S')}-{random.randint(1000,9999)}"
    start_ts = datetime.now(timezone.utc)
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
    # Inicializações defensivas para evitar variáveis não definidas em fluxos excepcionais
    tipo_ataque_label = None
    tipo_ataque_fk = None
    tipo_ataque_nome = None
    explanation_text = None
    incident_plan = ""
    print(f"[{req_id}] device_id={device_id} | features_tipo={type(features)} | samples_tipo={type(samples_payload)}", flush=True)
    
    # Garante estrutura mínima do banco de dados
    try:
        if not ensure_bootstrap():
            print(f"[{req_id}] ❌ Falha ao garantir estrutura do banco.", flush=True)
            return jsonify({"error": "Falha ao preparar banco de dados."}), 500
    except Exception as e:
        print(f"[{req_id}] ⚠️ Erro ao tentar bootstrap do banco: {e}", flush=True)

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
        # Falha explícita se não houve predições do binário (evita mascarar como 'Normal')
        if preds is None or len(preds) == 0:
            print(f"[{req_id}] [BIN] ❌ Nenhuma predição retornada pela crew binária.", flush=True)
            return jsonify({"error": "Nenhuma predição retornada pela crew binária"}), 502
        # Se múltiplas amostras foram enviadas, considere ataque se qualquer for 1
        is_attack = any(int(p) == 1 for p in preds)
        skip_mult = not is_attack
        if skip_mult:
            print(f"[{req_id}] ➡️ Evento não classificado como ataque pela crew binária. Prosseguindo com fallback para 'Normal'.", flush=True)

        # 3.1. Seleciona a PRIMEIRA amostra classificada como ATAQUE para a multiclasse
        attack_index = None
        try:
            for i, p in enumerate(preds):
                if int(p) == 1:
                    attack_index = i
                    break
        except Exception as e:
            print(f"[{req_id}] [MULT] Falha ao determinar índice do ataque a partir de preds={preds}: {e}", flush=True)

        if samples_payload and isinstance(samples_payload, list) and attack_index is not None and 0 <= attack_index < len(samples_payload):
            chosen_sample = samples_payload[attack_index]
            mult_inputs = {"samples": [chosen_sample]}
            try:
                preview_keys = list(chosen_sample.keys())[:10]
                print(f"[{req_id}] [MULT] Usando amostra de índice {attack_index} (primeira com voto=1) | primeiras chaves: {preview_keys}", flush=True)
            except Exception:
                print(f"[{req_id}] [MULT] Usando amostra de índice {attack_index} (primeira com voto=1)", flush=True)
        elif isinstance(features, dict):
            # Caso single-feature, já temos 1 amostra
            mult_inputs = {'samples': [features]}
            print(f"[{req_id}] [MULT] Modo single-feature -> enviando a única amostra para multiclasse", flush=True)
        else:
            # Fallback: não foi possível mapear índice -> envia a primeira amostra disponível
            fallback_sample = samples_payload[0] if isinstance(samples_payload, list) and samples_payload else None
            mult_inputs = {'samples': [fallback_sample]} if fallback_sample else inputs_for_mult_crew
            print(f"[{req_id}] [MULT] Fallback -> enviando a primeira amostra para multiclasse", flush=True)

        # Se binário indicou não-ataque, assume 'Normal' e não roda multiclasse
        if skip_mult:
            tipo_ataque_label = "Normal"
            explanation_text = "Sem explicação fornecida pelo líder."
            incident_plan = ""
        else:
            print(f"[{req_id}] [MULT] Iniciando crew multiclasse; qtd_amostras={len(mult_inputs.get('samples', []))}", flush=True)

            shap_explanation = ""
            try:
                # Use the centralized SHAP tool (handles model loading/caching and SHAP shapes)
                sample_for_shap = mult_inputs['samples'][0]
                print(f"[{req_id}] [SHAP] Preparing ExplainTop2SHAP for multiclass...", flush=True)
                explainer_tool = ExplainTop2SHAP(classification="multiclass")

                # Ensure feature ordering matches the model training schema
                if hasattr(explainer_tool.model, "feature_names_in_"):
                    ordered_sample = {k: float(sample_for_shap.get(k, 0.0)) for k in explainer_tool.model.feature_names_in_}
                else:
                    ordered_sample = sample_for_shap

                shap_explanation = explainer_tool._run([ordered_sample])
                print(f"[{req_id}] [SHAP] Explanation generated: {shap_explanation}", flush=True)
                mult_inputs['shap_explanation'] = shap_explanation
                print(f"[{req_id}] [SHAP] Explanation added to crew context.", flush=True)
            except Exception as e:
                mult_inputs['shap_explanation'] = ""
                print(f"[{req_id}] ⚠️ [SHAP] Failed to generate explanation: {e}", flush=True)

            result_mult = CyberPredictMult().crew().kickoff(inputs=mult_inputs)
            print(f"[{req_id}] [MULT] Crew multiclasse finalizada. Tipo de retorno: {type(result_mult)}", flush=True)
            try:
                raw_text_mult = getattr(result_mult, "raw", str(result_mult))
                print(f"[{req_id}] [MULT] Saída bruta (até 1000 chars): {raw_text_mult[:1000]}", flush=True)
            except Exception as e:
                print(f"[{req_id}] [MULT] Falha ao obter saída bruta: {e}", flush=True)

            # ---------- Tolerant parsing for leader output ----------
            def _strip_think_tags(t):
                try:
                    return re.sub(r"<think>.*?</think>", "", t or "", flags=re.DOTALL | re.IGNORECASE)
                except Exception:
                    return t or ""

            def _tolerant_parse_predictions_and_report(text):
                route = []
                s = _strip_think_tags(text or "")
                obj = extract_json_from_string(s)
                preds = None
                report = None
                if isinstance(obj, dict):
                    route.append("json")
                    raw_preds = obj.get("predictions") or obj.get("votes")
                    if isinstance(raw_preds, list):
                        try:
                            preds = [int(x) for x in raw_preds]
                        except Exception:
                            preds = extract_first_int_list_from_text(str(raw_preds))
                    elif isinstance(raw_preds, str):
                        preds = extract_first_int_list_from_text(raw_preds)
                    report = obj.get("report") or obj.get("explanation") or None

                # Fallback 1: first [..] list in text
                if not preds:
                    arr = extract_first_int_list_from_text(s)
                    if arr:
                        route.append("list")
                        preds = arr

                # Fallback 2: LaTeX \boxed{N}
                if not preds:
                    m = re.search(r"\\boxed\{\s*(-?\d+)\s*\}", s)
                    if m:
                        route.append("boxed")
                        try:
                            preds = [int(m.group(1))]
                        except Exception:
                            preds = None

                # Fallback 3: "class N" pattern
                if not preds:
                    m2 = re.search(r"\bclass\s*[:=]?\s*(\d{1,3})\b", s, re.IGNORECASE)
                    if m2:
                        route.append("classN")
                        try:
                            preds = [int(m2.group(1))]
                        except Exception:
                            preds = None

                # Final fallback: safe default
                if not preds:
                    route.append("default99")
                    preds = [99]

                # Build report
                if not isinstance(report, str) or not report.strip():
                    # Derive minimal, clean report from the sanitized text
                    report_candidate = re.sub(r"```.*?```", "", s, flags=re.DOTALL)
                    report_candidate = re.sub(r"\[.*?\]", "", report_candidate, flags=re.DOTALL)
                    report_candidate = re.sub(r"\\boxed\{.*?\}", "", report_candidate)
                    report_candidate = re.sub(r"\s+", " ", report_candidate).strip()
                    report = report_candidate if report_candidate else "Sem explicação fornecida pelo líder."

                try:
                    print(f"[{req_id}] [MULT] Parser route: {'>'.join(route)} | preds={preds} | report_len={len(report)}", flush=True)
                except Exception:
                    pass

                return {"predictions": preds, "report": report}

            raw_text_mult = getattr(result_mult, "raw", str(result_mult))
            mult_output = _tolerant_parse_predictions_and_report(raw_text_mult)

            preds = mult_output.get("predictions", [])
            try:
                tipo_ataque_id_cast = int(preds[0])
            except Exception:
                print(f"[{req_id}] ❌ Multiclasse: 'predictions[0]' não é inteiro ou ausente: {preds}", flush=True)
                tipo_ataque_id_cast = 99

            print(f"[{req_id}] [MULT] Predição final do líder (tolerant): {tipo_ataque_id_cast}", flush=True)

            conn = get_db_connection()
            with conn.cursor() as cursor:
                tipo_ataque_fk, tipo_ataque_nome = resolve_tipo_ataque_id(cursor, tipo_ataque_id_cast)
                tipo_ataque_label = tipo_ataque_nome
                print(f"[{req_id}] [MULT] Label de ataque selecionado: ({tipo_ataque_fk}, '{tipo_ataque_nome}')", flush=True)
            conn.close()

            # Explicação do líder (tolerant, sempre string)
            explanation_text = mult_output.get("report") or "Sem explicação fornecida pelo líder."
            if not isinstance(explanation_text, str):
                explanation_text = str(explanation_text)
            print(f"[{req_id}] [MULT] Tamanho do relatório do líder: {len(explanation_text)}", flush=True)

            # Plano de resposta ao incidente (markdown) somente se não for 'Normal'
            incident_plan = ""
            if str(tipo_ataque_label).lower() != "normal":
                try:
                    print(f"[{req_id}] [IR] Iniciando geração do plano de resposta para '{tipo_ataque_label}'", flush=True)
                    # tasks_response.yaml espera a chave 'attack_label' e valor string
                    ir_result = IncidentResponseCrew().crew().kickoff(inputs={"attack_label": tipo_ataque_label})
                    incident_plan = getattr(ir_result, "raw", str(ir_result))
                    print(f"[{req_id}] [IR] Plano de resposta gerado (até 800 chars):\n{incident_plan[:800]}", flush=True)
                except Exception:
                    print(f"[{req_id}] ⚠️ Falha ao gerar plano de resposta. Prosseguindo sem ações.\n{traceback.format_exc()}", flush=True)
                    incident_plan = ""

        print(f"[{req_id}] ✅ Análise concluída. Ataque tipo: {tipo_ataque_label} para o dispositivo: {device_id}", flush=True)

        # 4. Salva a detecção no banco de dados
        try:
            # Mostra os valores efetivos (com fallbacks) sem expor senha
            _db_host = os.getenv('MYSQL_HOST') or 'mysql'
            _db_user = os.getenv('MYSQL_USER') or os.getenv('MYSQL_USERNAME') or 'root'
            _db_name = os.getenv('MYSQL_DB') or os.getenv('MYSQL_DATABASE') or 'qcyber_db'
            print(f"[{req_id}] [DB] Conectando ao MySQL host={_db_host} user={_db_user} db={_db_name}", flush=True)
        except Exception:
            pass

        conn = get_db_connection()
        with conn.cursor() as cursor:
            status_resp_ids = get_lookup_ids(cursor, "enum_status_resposta")
            preferred = ['Pendente', 'Ação Automática Executada', 'Análise Manual Necessária', 'Ignorado']
            status_resposta_id = next((status_resp_ids[n] for n in preferred if n in status_resp_ids), None)
            if status_resposta_id is None and status_resp_ids:
                status_resposta_id = next(iter(status_resp_ids.values()), None)
            status_pendente_id = status_resp_ids.get('Pendente', status_resposta_id)
            status_manual_id = status_resp_ids.get('Análise Manual Necessária', status_resposta_id)

            # Resolve FK do tipo de ataque normalizado (fallback=(99,'Normal'))
            # Se já conhecido (via multiclasse), reutiliza; caso contrário, resolve agora (ex.: 'Normal')
            if tipo_ataque_fk is None or tipo_ataque_nome is None:
                tipo_ataque_fk, tipo_ataque_nome = resolve_tipo_ataque_id(cursor, tipo_ataque_label)

            # Resolve/garante dispositivo
            dispositivo_id = ensure_dispositivo(cursor, device_id)
            if not dispositivo_id:
                raise Exception("Não foi possível resolver/garantir dispositivo_id válido.")
            try:
                print(f"[{req_id}] [DB] Status resposta mapeados: {len(status_resp_ids)} | pendente={status_pendente_id} manual={status_manual_id}", flush=True)
                print(f"[{req_id}] [DB] tipo_ataque_label='{tipo_ataque_label}' -> fk_id={tipo_ataque_fk} | dispositivo_id={dispositivo_id}", flush=True)
            except Exception:
                pass

            # Nome canônico do tipo de ataque já resolvido
            if not tipo_ataque_nome:
                tipo_ataque_nome = str(tipo_ataque_label) if tipo_ataque_label else 'Normal'
            # Torna o relatório útil: combine relatório do líder e plano (quando houver)
            resumo_detect = f"Detecção: '{tipo_ataque_nome}'"
            partes_relatorio = []
            if explanation_text and explanation_text.strip() and explanation_text.strip().lower() != 'sem explicação fornecida pelo líder.':
                partes_relatorio.append(f"Relatório de Classificação (Crew):\n{explanation_text.strip()}")
            if incident_plan and str(incident_plan).strip():
                partes_relatorio.append(f"Plano de Resposta ao Incidente (Crew):\n{str(incident_plan).strip()}")

            # Constrói texto do relatório para persistência
            try:
                relatorio_api_text = resumo_detect
                if partes_relatorio:
                    relatorio_api_text = resumo_detect + "\n\n" + "\n\n".join(partes_relatorio)
            except Exception:
                relatorio_api_text = resumo_detect

            # Insere detecção com predicao=enum_id (não 0/1)
            sql_det = """
            INSERT INTO deteccoes (data_deteccao, dispositivo_id, predicao, tipo_ataque_id, relatorio_api, status_resposta_id, incidente_id)
            VALUES (%s, %s, %s, %s, %s, %s, NULL)
            """
            cursor.execute(sql_det, (datetime.now(timezone.utc), dispositivo_id, tipo_ataque_fk, tipo_ataque_fk, relatorio_api_text, status_pendente_id))
            new_detection_id = cursor.lastrowid

            new_incidente_id = None
            if tipo_ataque_fk != 99:
                status_inc_map = get_lookup_ids(cursor, "enum_status_incidente")
                status_inc_aberto = status_inc_map.get('Aberto') or next(iter(status_inc_map.values()), None)
                risco_map = get_lookup_ids(cursor, "enum_nivel_risco")
                nivel_risco_id = risco_map.get('Médio') or next(iter(risco_map.values()), None)

                titulo = f"Detecção de {tipo_ataque_nome}"
                resumo_tecnico = f"Detecção do tipo '{tipo_ataque_nome}' registrada pelo analisador."
                try:
                    acoes_json = json.dumps([incident_plan or ""])
                except Exception:
                    acoes_json = json.dumps({"plano": ""})

                sql_inc = """
                INSERT INTO incidentes_analisados (titulo, status_id, dispositivo_id, nivel_risco_id, data_deteccao, resumo_tecnico, explicacao_llm, acoes_recomendadas)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
                """
                cursor.execute(sql_inc, (titulo, status_inc_aberto, dispositivo_id, nivel_risco_id, datetime.now(timezone.utc), resumo_tecnico, explanation_text, acoes_json))
                new_incidente_id = cursor.lastrowid

                # Atualiza detecção com incidente e status 'Análise Manual Necessária'
                cursor.execute("UPDATE deteccoes SET incidente_id=%s, status_resposta_id=%s WHERE id=%s", (new_incidente_id, status_manual_id, new_detection_id))

            conn.commit()
        conn.close()

        # Single-line snapshot for unit tests
        try:
            snapshot = {
                "req_id": req_id,
                "device_id": device_id,
                "bin": {
                    "raw": locals().get("raw_text_bin", ""),
                    "json": locals().get("bin_output", None),
                    "preds": locals().get("preds", []),
                    "is_attack": locals().get("is_attack", False),
                    "skip_mult": locals().get("skip_mult", False),
                },
                "mult": {
                    "raw": locals().get("raw_text_mult", ""),
                    "json": locals().get("mult_output", None),
                    "pred": locals().get("tipo_ataque_id_cast", None),
                    "label": locals().get("tipo_ataque_label", None),
                    "report": locals().get("explanation_text", None),
                },
                "incident": {
                    "plan": locals().get("incident_plan", ""),
                },
                "db": {
                    "tipo_ataque_fk": locals().get("tipo_ataque_fk", None),
                    "tipo_ataque_nome": locals().get("tipo_ataque_nome", None),
                    "dispositivo_id": locals().get("dispositivo_id", None),
                },
                "response": {
                    "deteccao_id": locals().get("new_detection_id", None),
                    "incidente_id": locals().get("new_incidente_id", None),
                }
            }
            print("[UNIT] " + json.dumps(snapshot, ensure_ascii=False), flush=True)
        except Exception as _unit_err:
            # Do not break API if snapshot fails; just log a short note
            print(f"[{req_id}] [UNIT] Snapshot failed: {_unit_err}", flush=True)

        print(f"[{req_id}] 💾 Detecção #{new_detection_id} salva no banco de dados.", flush=True)
        try:
            duration = (datetime.now(timezone.utc) - start_ts).total_seconds()
            print(f"[{req_id}] ⏱️ Duração total da requisição: {duration:.2f}s", flush=True)
        except Exception:
            pass

        # Resposta mínima conforme contrato
        response_payload = {
            "mensagem": "Detecção analisada e registrada com sucesso!",
            "deteccao_id": new_detection_id,
            "incidente_id": new_incidente_id,
            "tipo_ataque_detectado": tipo_ataque_nome,
        }
        return jsonify(response_payload), 201

    except Exception as e:
        print(f"[{req_id}] ❌ ERRO GERAL DURANTE A ANÁLISE: {e}\n{traceback.format_exc()}", flush=True)
        return jsonify({"error": "Ocorreu um erro interno no servidor de análise."}), 500



if __name__ == "__main__":
    # O threaded=True ajuda a lidar com múltiplas conexões de forma mais estável
    # Em produção, você usaria um servidor WSGI como Gunicorn ou Waitress
    app.run(host='0.0.0.0', port=int(get_env_var('FLASK_API_PORT', 5000)), debug=True, threaded=True)
