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
from qml.utils.generic import get_env_var
from qml.utils.runtime import default_crew_enabled, env_flag

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
from qml.services.detection import BatchDetectionService, build_detection_report
from qml.services.remediation import run_remediation_chat
from qml.tools.shap_explain import ExplainTop2SHAP
os.environ['CREWAI_DISABLE_TELEMETRY'] = 'true'
os.environ['OTEL_SDK_DISABLED'] = 'true'


app = Flask(__name__)
load_dotenv()


@app.route('/remediation/chat', methods=['POST'])
def remediation_chat():
    data = request.get_json(silent=True) or {}
    attack_label = str(data.get("attack_label") or "").strip()
    operator_message = str(data.get("operator_message") or "").strip()
    context = data.get("context")

    if not attack_label:
        return jsonify({"error": "'attack_label' é obrigatório."}), 400
    if not operator_message:
        return jsonify({"error": "'operator_message' é obrigatório."}), 400

    try:
        response = run_remediation_chat(
            attack_label=attack_label,
            operator_message=operator_message,
            context=context,
        )
        return jsonify(response), 200
    except Exception as exc:
        print(f"[REMEDIATION] Erro ao gerar chat de remediação: {exc}\n{traceback.format_exc()}", flush=True)
        return jsonify({"error": "Falha ao gerar recomendações de remediação."}), 500

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
        if 'others' not in norm_map and 14 in id_map:
            norm_map['others'] = id_map[14]
        if 'outras' not in norm_map and 14 in id_map:
            norm_map['outras'] = id_map[14]

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



def _required_env(name: str) -> str:
    value = (os.getenv(name) or "").strip()
    if not value:
        raise RuntimeError(f"{name} must be set for the qcyber_api MySQL connection")
    return value


def get_db_connection():
    """Cria uma conexão com o banco mantido pelo qcyber_api."""
    db_host = _required_env("MYSQL_HOST")
    db_port = int(_required_env("MYSQL_PORT"))
    db_user = _required_env("MYSQL_USER")
    db_pass = _required_env("MYSQL_PASSWORD")
    db_name = _required_env("MYSQL_DATABASE")

    return pymysql.connect(
        host=db_host,
        port=db_port,
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
    
    try:
        # 2. Prepara os dados para a crewai
        if samples_payload and isinstance(samples_payload, list):
            # Recebemos múltiplas amostras já no payload
            x_test = pd.DataFrame(samples_payload)
            inputs_for_detection = {"samples": samples_payload}
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
            inputs_for_detection = {'samples': input_records_single}
            try:
                print(f"[{req_id}] Modo single-feature: {len(features.keys())} chaves", flush=True)
            except Exception:
                print(f"[{req_id}] Modo single-feature: não foi possível contar chaves de 'features'", flush=True)

        try:
            print(f"[{req_id}] x_test shape={x_test.shape} | detection_samples={len(inputs_for_detection.get('samples', []))}", flush=True)
        except Exception as e:
            print(f"[{req_id}] Falha ao inspecionar x_test ou inputs: {e}", flush=True)

        # 3. Executa predição determinística por batch.
        # O modelo binário avalia a janela inteira; se houver amostras suspeitas,
        # o multiclasse roda somente nelas e consolida um tipo primário.
        samples_for_detection = inputs_for_detection.get("samples", [])
        crew_default = default_crew_enabled()
        use_binary_crew = env_flag("QCYBER_USE_BINARY_CREW", default=crew_default)
        use_multiclass_crew = env_flag("QCYBER_USE_MULTICLASS_CREW", default=crew_default)
        use_incident_response_crew = env_flag(
            "QCYBER_USE_INCIDENT_RESPONSE_CREW",
            default=crew_default,
        )
        use_remediation_crew = env_flag("QCYBER_USE_REMEDIATION_CREW", default=crew_default)
        print(
            f"[{req_id}] [BIN] Iniciando fluxo com {len(samples_for_detection)} amostras | "
            f"binary_crew={use_binary_crew} multiclass_crew={use_multiclass_crew} "
            f"incident_response_crew={use_incident_response_crew} remediation_crew={use_remediation_crew}",
            flush=True,
        )
        detection_result = BatchDetectionService(
            use_binary_crew=use_binary_crew,
            use_multiclass_crew=use_multiclass_crew,
            use_incident_response_crew=use_incident_response_crew,
            use_remediation_crew=use_remediation_crew,
        ).predict(samples_for_detection)
        bin_preds = detection_result.binary_predictions
        raw_text_bin = json.dumps(detection_result.to_dict(), ensure_ascii=False)
        bin_output = {
            "predictions": bin_preds,
            "attack_indices": detection_result.attack_indices,
            "window_has_attack": detection_result.window_has_attack,
        }

        print(f"[{req_id}] [BIN] Predições: {bin_preds} | suspeitas={detection_result.attack_indices}", flush=True)
        if not bin_preds:
            print(f"[{req_id}] [BIN] ❌ Nenhuma predição retornada pelo modelo binário.", flush=True)
            return jsonify({"error": "Nenhuma predição retornada pelo modelo binário"}), 502

        is_attack = detection_result.window_has_attack
        skip_mult = not is_attack
        if skip_mult:
            print(f"[{req_id}] ➡️ Janela sem ataque pelo modelo binário. Registrando como 'Normal'.", flush=True)

        # Se binário indicou não-ataque, assume 'Normal' e não roda multiclasse/IR
        if skip_mult:
            tipo_ataque_label = "Normal"
            explanation_text = "Sem explicação fornecida pelo líder."
            incident_plan = ""
        else:
            suspicious_samples = [samples_for_detection[idx] for idx in detection_result.attack_indices]
            print(
                f"[{req_id}] [MULT] Workflow multiclasse concluído para "
                f"{len(suspicious_samples)} amostra(s) suspeita(s): "
                f"{[item.attack_type_label for item in detection_result.classified_attacks]}",
                flush=True,
            )

            shap_explanation = ""
            try:
                # Use the centralized SHAP tool (handles model loading/caching and SHAP shapes)
                print(f"[{req_id}] [SHAP] Preparing ExplainTop2SHAP for multiclass...", flush=True)
                explainer_tool = ExplainTop2SHAP(classification="multiclass")

                # Ensure feature ordering matches the model training schema
                if hasattr(explainer_tool.model, "feature_names_in_"):
                    ordered_samples = [
                        {k: float(sample.get(k, 0.0)) for k in explainer_tool.model.feature_names_in_}
                        for sample in suspicious_samples
                    ]
                else:
                    ordered_samples = suspicious_samples

                shap_explanation = explainer_tool._run(ordered_samples)
                print(f"[{req_id}] [SHAP] Explanation generated: {shap_explanation}", flush=True)
            except Exception as e:
                print(f"[{req_id}] ⚠️ [SHAP] Failed to generate explanation: {e}", flush=True)

            tipo_ataque_id_cast = detection_result.primary_attack_type_id
            raw_text_mult = json.dumps(detection_result.to_dict(), ensure_ascii=False)
            mult_output = {
                "predictions": [tipo_ataque_id_cast],
                "classified_attacks": detection_result.to_dict()["classified_attacks"],
                "report": build_detection_report(detection_result, shap_explanation),
            }

            print(f"[{req_id}] [MULT] Tipo primário consolidado: {tipo_ataque_id_cast}", flush=True)

            conn = get_db_connection()
            with conn.cursor() as cursor:
                tipo_ataque_fk, tipo_ataque_nome = resolve_tipo_ataque_id(cursor, tipo_ataque_id_cast)
                tipo_ataque_label = tipo_ataque_nome
                print(f"[{req_id}] [MULT] Label de ataque selecionado: ({tipo_ataque_fk}, '{tipo_ataque_nome}')", flush=True)
            conn.close()

            # Explicação consolidada do pipeline determinístico.
            explanation_text = mult_output.get("report") or "Sem explicação fornecida pelo líder."
            if not isinstance(explanation_text, str):
                explanation_text = str(explanation_text)
            print(f"[{req_id}] [MULT] Tamanho do relatório do líder: {len(explanation_text)}", flush=True)

            incident_plan = detection_result.incident_response_plan or ""
            if incident_plan:
                print(f"[{req_id}] [IR] Plano de resposta gerado (até 800 chars):\n{incident_plan[:800]}", flush=True)
            if detection_result.attack_response_plans:
                print(
                    f"[{req_id}] [IR] Planos por tipo: "
                    f"{[(item.get('attack_type_label'), item.get('batch_indices')) for item in detection_result.attack_response_plans]}",
                    flush=True,
                )

        print(f"[{req_id}] ✅ Análise concluída. Ataque tipo: {tipo_ataque_label} para o dispositivo: {device_id}", flush=True)

        # 4. Salva a detecção no banco de dados
        try:
            # Mostra os valores efetivos sem expor senha.
            _db_host = _required_env("MYSQL_HOST")
            _db_port = int(_required_env("MYSQL_PORT"))
            _db_user = _required_env("MYSQL_USER")
            _db_name = _required_env("MYSQL_DATABASE")
            print(f"[{req_id}] [DB] Conectando ao MySQL host={_db_host} port={_db_port} user={_db_user} db={_db_name}", flush=True)
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
                partes_relatorio.append(f"Relatório de Classificação:\n{explanation_text.strip()}")
            if incident_plan and str(incident_plan).strip():
                partes_relatorio.append(f"Plano de Resposta ao Incidente (Crew):\n{str(incident_plan).strip()}")
            if "detection_result" in locals() and detection_result.attack_remediation_suggestions:
                remediation_text = json.dumps(
                    detection_result.attack_remediation_suggestions,
                    ensure_ascii=False,
                    indent=2,
                )
                partes_relatorio.append(f"Sugestões de Remediação por Tipo:\n{remediation_text}")

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
                    acoes_payload = {
                        "incident_response_plans": detection_result.attack_response_plans,
                        "remediation_suggestions": detection_result.attack_remediation_suggestions,
                        "legacy_combined_incident_plan": incident_plan or "",
                    }
                    acoes_json = json.dumps(acoes_payload, ensure_ascii=False)
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
                    "preds": locals().get("bin_preds", []),
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
                    "plans_by_attack_type": locals().get("detection_result").attack_response_plans if "detection_result" in locals() else [],
                    "remediation_by_attack_type": locals().get("detection_result").attack_remediation_suggestions if "detection_result" in locals() else [],
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
            "batch": {
                "predicoes_binarias": locals().get("bin_preds", []),
                "indices_suspeitos": locals().get("detection_result").attack_indices if "detection_result" in locals() else [],
                "ataques_classificados": locals().get("detection_result").to_dict().get("classified_attacks", []) if "detection_result" in locals() else [],
                "planos_resposta_por_tipo": locals().get("detection_result").attack_response_plans if "detection_result" in locals() else [],
                "remediacoes_por_tipo": locals().get("detection_result").attack_remediation_suggestions if "detection_result" in locals() else [],
            },
        }
        return jsonify(response_payload), 201

    except Exception as e:
        print(f"[{req_id}] ❌ ERRO GERAL DURANTE A ANÁLISE: {e}\n{traceback.format_exc()}", flush=True)
        return jsonify({"error": "Ocorreu um erro interno no servidor de análise."}), 500



if __name__ == "__main__":
    # O threaded=True ajuda a lidar com múltiplas conexões de forma mais estável
    # Em produção, você usaria um servidor WSGI como Gunicorn ou Waitress
    app.run(host='0.0.0.0', port=int(get_env_var('FLASK_API_PORT', 5000)), debug=True, threaded=True)
