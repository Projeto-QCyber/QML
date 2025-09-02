import json
import pandas as pd

from qml.crew import CyberPredict
from qml.crew_multiclass import CyberPredictMult

def run():
    """
    Executa a crew e avalia seu desempenho.
    """
    data_path_bin = "data/dados_de_teste_bin.csv"
    data_path_mult = "data/dados_de_teste_mult.csv"

    sampled_data_bin = pd.read_csv(data_path_bin).sample(20, random_state=42)
    sampled_data_mult = pd.read_csv(data_path_mult).sample(20, random_state=42)

    gt_bin_labels = sampled_data_bin["Attack_label"].tolist()
    gt_mult_labels = sampled_data_mult["Attack_type"].tolist()

    sample_data_test = sampled_data_bin.drop(["Attack_label"], axis=1)
    sample_data_test = sample_data_test.reset_index(drop=True)
    input_records = sample_data_test.to_dict(orient='records')
    inputs_for_crew = {
        'samples': input_records # Trocamos 'argument' por 'samples'
    }

    try:
        result_bin = CyberPredict().crew().kickoff(inputs=inputs_for_crew)
        
        # Forçar conversão para dicionário
        if isinstance(result_bin.raw, str):
            bin_output = json.loads(result_bin.raw)
        elif hasattr(result_bin, "dict"):
            bin_output = result_bin.model_dump()
        else:
            bin_output = result_bin.raw

        # Agora, pegamos a lista de predições de dentro do dicionário
        print("-----------------BINARY-----------------")
        print("\n\n--- COMPARAÇÃO (PREDICT vs. GROUND TRUTH) ---")
        print(f"Type of output:     {type(bin_output)}")
        print(f"Valores Reais (Ground Truth):     {gt_bin_labels}")
        print(f"Resultado Bruto da Crew (Predict): {bin_output}")
        print("-------------------------------------------------")

        inputs_for_mult_crew = {
            'samples': bin_output.predictions
        }

        result_mult = CyberPredictMult().crew().kickoff(inputs=inputs_for_mult_crew)
        mult_output = result_mult.raw

        print("-----------------MULTICLASS-----------------")
        print("\n\n--- COMPARAÇÃO (PREDICT vs. GROUND TRUTH) ---")
        print(f"Type of output:     {type(mult_output)}")
        print(f"Valores Reais (Ground Truth):     {gt_mult_labels}")
        print(f"Resultado Bruto da Crew (Predict): {mult_output}")
        print("-------------------------------------------------")

    except Exception as e:
        print(f"Erro ao processar o resultado final: {e}")

if __name__ == "__main__":
    run()