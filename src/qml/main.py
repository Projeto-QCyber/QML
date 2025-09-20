import json
import numpy as np
import pandas as pd
from pathlib import Path
from collections.abc import Mapping
import re

ROOT_DIR = Path(__file__).resolve().parents[2]

# Import Crews
from qml.crew import CyberPredict
from qml.crew_multiclass import CyberPredictMult
from qml.response import IncidentResponseCrew # O novo crew para resposta a incidentes

# Import Utilities
from qml.utils.decode import decode_type_attack # Supondo que seja um helper para prints

def extract_json(text):
    """
    Função para extrair um bloco JSON de uma string de texto, mesmo que haja texto antes ou depois.
    """
    if not isinstance(text, str):
        return None
    
    # Procura por um bloco JSON que pode estar dentro de marcadores de código
    match = re.search(r'```json\s*(\{.*?\})\s*```', text, re.DOTALL)
    if match:
        return match.group(1)

    # Se não encontrar com marcadores, procura pelo primeiro '{' e o último '}'
    json_start = text.find('{')
    json_end = text.rfind('}')
    
    if json_start != -1 and json_end != -1 and json_end > json_start:
        return text[json_start:json_end+1]
        
    return None

def run():
    """
    Executa os crews para detecção, classificação e resposta a incidentes.
    """
    data_path = ROOT_DIR / "data/dados_de_teste.csv"

    # Para demonstração, vamos amostrar 5 registros dos dados de teste
    sampled_data = pd.read_csv(data_path).sample(5, random_state=42)

    # Armazena os rótulos verdadeiros para comparação posterior
    gt_bin_labels = sampled_data["Attack_label"].tolist()
    gt_mult_labels = sampled_data["Attack_type"].tolist()

    # Prepara os dados para predição, removendo os rótulos
    x_test = sampled_data.drop(["Attack_label", "Attack_type"], axis=1)
    x_test = x_test.reset_index(drop=True)
    input_records = x_test.to_dict(orient='records')

    inputs_for_crew = {
        'samples': input_records
    }

    try:
        # ETAPA 1: CLASSIFICAÇÃO BINÁRIA (Ataque vs. Normal)
        print("--- INICIANDO ETAPA 1: CLASSIFICAÇÃO BINÁRIA ---")
        result_bin = CyberPredict().crew().kickoff(inputs=inputs_for_crew)
        
        bin_output = {} # Inicializa como um dicionário vazio
        
        json_str = extract_json(result_bin.raw)
        if json_str:
            try:
                bin_output = json.loads(json_str)
            except json.JSONDecodeError:
                print(f"Aviso: JSON inválido: {json_str}")
        elif hasattr(result_bin, "model_dump"):
            bin_output = result_bin.model_dump()
        else:
            # NOVO: Fallback para caso seja apenas lista no corpo cru
            if result_bin.raw.strip().startswith("["):
                try:
                    bin_output = {"predictions": json.loads(result_bin.raw)}
                except Exception as e:
                    print(f"Aviso: Falha ao converter lista de predições: {e}")
                    bin_output = {}
            else:
                print(f"Aviso: Resposta inesperada: {result_bin.raw}")


        print("-----------------BINÁRIO-----------------")
        print("\n\n--- COMPARAÇÃO (PREDICT vs. GROUND TRUTH) ---")
        print(f"Predições:        {bin_output.get('predictions', 'N/A')}")
        print(f"Valores Reais:    {gt_bin_labels}")
        print("-------------------------------------------------")

        # ETAPA 2: CLASSIFICAÇÃO MULTICLASSE (Tipo de Ataque)
        print("\n\n--- INICIANDO ETAPA 2: CLASSIFICAÇÃO MULTICLASSE ---")
        predictions = bin_output.get("predictions", [])
        if not predictions:
            print("Nenhuma predição válida na Etapa 1. Finalizando o processo.")
            return

        bin_output_ndarray = np.array(predictions)
        attack_indices = np.where(bin_output_ndarray == 1)[0]
        
        if len(attack_indices) == 0:
            print("Nenhum ataque detectado na Etapa 1. Finalizando o processo.")
            return

        x_test_mult = x_test.loc[attack_indices]
        gt_mult_labels_filtered = [gt_mult_labels[i] for i in attack_indices]
        input_records_mult = x_test_mult.to_dict(orient='records')
        inputs_for_mult_crew = {"samples": input_records_mult}
        
        result_mult = CyberPredictMult().crew().kickoff(inputs=inputs_for_mult_crew)
        
        mult_output = {}
        json_str_mult = extract_json(result_mult.raw)
        if json_str_mult:
            try:
                mult_output = json.loads(json_str_mult)
            except json.JSONDecodeError:
                 print(f"Aviso: Falha ao decodificar o JSON extraído da resposta do crew multiclasse. Conteúdo extraído: {json_str_mult}")
        else:
             print(f"Aviso: Resposta inesperada ou sem JSON válido do crew multiclasse. Conteúdo: {result_mult.raw}")


        attack_types_detected = mult_output.get('predictions', [])

        print("-----------------MULTICLASSE-----------------")
        print("\n\n--- COMPARAÇÃO (PREDICT vs. GROUND TRUTH) ---")
        print(f"Tipos de Ataque Previstos: {attack_types_detected}")
        print(f"Valores Reais:             {gt_mult_labels_filtered}")
        print("-------------------------------------------------")

        # ETAPA 3: GERAR PLANO DE RESPOSTA
        print("\n\n--- INICIANDO ETAPA 3: GERANDO PLANOS DE RESPOSTA ---")
        
        unique_attacks = set(attack_types_detected)
        unique_attacks = {attack for attack in unique_attacks if attack.lower() != 'normal'}

        if not unique_attacks:
            print("Nenhum tipo de ataque específico foi identificado. Nenhum plano de resposta será gerado.")
        else:
            for attack_type in unique_attacks:
                print(f"\n--- PLANO DE RESPOSTA PARA: {attack_type.upper()} ---")
                
                response_inputs = {'attack_type': attack_type}
                response_crew_instance = IncidentResponseCrew().crew()
                response_plan_raw = response_crew_instance.kickoff(inputs=response_inputs)
                
                print(response_plan_raw)
                print("-------------------------------------------------")


    except Exception as e:
        print(f"Ocorreu um erro durante a execução: {e}")

if __name__ == "__main__":
    from datetime import datetime

    # Get the current date and time
    current_datetime_i = datetime.now()

    # Print the result
    print("*" * 60)
    print(current_datetime_i)

    run()
