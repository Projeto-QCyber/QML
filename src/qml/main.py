import json
import numpy as np
import pandas as pd
from pathlib import Path
from collections.abc import Mapping
from qml.crew import CyberPredict
from qml.crew_multiclass import CyberPredictMult
from qml.utils.decode import decode_type_attack


ROOT_DIR = Path(__file__).resolve().parents[2]

def run():
    """
    Executa a crew e avalia seu desempenho.
    """
    data_path = ROOT_DIR / "data/dados_de_teste.csv"

    sampled_data = pd.read_csv(data_path).sample(1, random_state=42)

    gt_bin_labels = sampled_data["Attack_label"].tolist()
    gt_mult_labels = sampled_data["Attack_type"].tolist()

    x_test = sampled_data.drop(["Attack_label", "Attack_type"], axis=1)
    x_test = x_test.reset_index(drop=True)
    input_records = x_test.to_dict(orient='records')

    inputs_for_crew = {
        'samples': input_records
    }

    try:
        result_bin = CyberPredict().crew().kickoff(inputs=inputs_for_crew)
        
        # Forçar conversão para dicionário
        if isinstance(result_bin.raw, str):
            print(f"Entendendo o result_bin.raw: {result_bin.raw}")
            bin_output = json.loads(result_bin.raw)
        elif hasattr(result_bin, "dict"):
            bin_output = result_bin.model_dump()
        else:
            bin_output = result_bin.raw

        # Agora, pegamos a lista de predições de dentro do dicionário
        print("-----------------BINARY-----------------")
        print("\n\n--- COMPARAÇÃO (PREDICT vs. GROUND TRUTH) ---")
        print(f"Vizualizando predições:     {bin_output['predictions']}")
        print(f"Valores Reais (Ground Truth):     {gt_bin_labels}")
        print(f"Resultado Bruto da Crew (Predict): {bin_output}")
        print("-------------------------------------------------")

        # Preparando dados para passar como input para o modelo multiclasse
        bin_output_ndarray = np.array(bin_output["predictions"])
        attack_index = np.where(bin_output_ndarray == 1)[0]
        x_test_mult = x_test.loc[attack_index]
        
        gt_mult_labels_filtered = [gt_mult_labels[i] for i in attack_index]
        input_records_mult = x_test_mult.to_dict(orient='records')

        inputs_for_mult_crew = {"samples": input_records_mult}

        # Extra guard rails for the "'function' has no attribute 'get'"
        assert isinstance(inputs_for_mult_crew, Mapping), "inputs_for_mult_crew must be a dict"
        assert "samples" in inputs_for_mult_crew, "missing 'samples' key for multiclass inputs"

        result_mult = CyberPredictMult().crew().kickoff(inputs=inputs_for_mult_crew)
        mult_output = result_mult.raw


        print("-----------------MULTICLASS-----------------")
        print("\n\n--- COMPARAÇÃO (PREDICT vs. GROUND TRUTH) ---")
        print(f"Type of output:     {type(mult_output)}")
        print(f"Valores Reais (Ground Truth):     {gt_mult_labels_filtered}")
        print(f"Resultado Bruto da Crew (Predict): {mult_output}")
        decode_type_attack(mult_output["predictions"])
        print("-------------------------------------------------")
        

    except Exception as e:
        print(f"Erro ao processar o resultado final: {e}")

if __name__ == "__main__":
    from datetime import datetime

    # Get the current date and time
    current_datetime_i = datetime.now()

    # Print the result
    print("*" * 60)
    print(current_datetime_i)

    run()

    current_datetime_f = datetime.now()
    print("*" * 60)
    print(current_datetime_f)
    print(current_datetime_f - current_datetime_i)