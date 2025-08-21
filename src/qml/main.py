import os
import pandas as pd
from sklearn.metrics import accuracy_score, classification_report
import ast

from qml.crew import CyberPredict

# Create output directory if it doesn't exist
os.makedirs('output', exist_ok=True)

def run():
    """
    Executa a crew e avalia seu desempenho.
    """
    data_path = "data/dados_de_teste.csv"

    # --- 1. PREPARAÇÃO E SEPARAÇÃO DAS 10 AMOSTRAS ---
    sampled_data = pd.read_csv(data_path).sample(20, random_state=42)
    
    # Guarda as respostas corretas (ground truth)
    ground_truth_labels = sampled_data["Attack_label"].tolist()


    
    # Prepara os dados para a crew (sem as respostas)
    sample_data_test = sampled_data.drop(["Attack_label"], axis=1)
    sample_data_test = sample_data_test.reset_index(drop=True)
    input_records = sample_data_test.to_dict(orient='records')
    inputs_for_crew = {
        'samples': input_records # Trocamos 'argument' por 'samples'
    }

    # --- 2. EXECUÇÃO DA CREW ---
    result = CyberPredict().crew().kickoff(inputs=inputs_for_crew)

    # --- 3. COMPARAÇÃO E AVALIAÇÃO ---
    try:
        # O resultado final do agente está no atributo .raw
        # Acessamos o dicionário que está dentro de .raw
        final_output_dict = result.raw

        # Agora, pegamos a lista de predições de dentro do dicionário
        print("\n\n--- COMPARAÇÃO (PREDICT vs. GROUND TRUTH) ---")
        print(f"Valores Reais (Ground Truth):     {ground_truth_labels}")
        print(f"Resultado Bruto da Crew (Predict): {result.raw}")
        print("-------------------------------------------------")


        #print("\n--- COMPARAÇÃO (PREDICT vs. GROUND TRUTH) ---")
        #print(f"Valores Reais (Ground Truth): {ground_truth_labels}")
        #print(f"Previsões da Equipe (Crew):   {predicted_labels}")
        #print("------------------------------------------\n")

        # Calcula e imprime a acurácia
        #accuracy = accuracy_score(ground_truth_labels, predicted_labels)
        #print(f"Acurácia: {accuracy:.2%}")
        
        #print("\n--- RELATÓRIO DE CLASSIFICAÇÃO ---\n")
        #print(classification_report(
        #    ground_truth_labels, 
        #    predicted_labels, 
        #    target_names=['Normal (0)', 'Ataque (1)'])
        #)

    except Exception as e:
        print(f"Erro ao processar o resultado final: {e}")
        print(f"Resultado bruto recebido da crew: {result.raw}")

if __name__ == "__main__":
    run()