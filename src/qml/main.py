import pandas as pd

from qml.crew import CyberPredict

def run():
    """
    Executa a crew e avalia seu desempenho.
    """
    data_path = "C:\\Users\\pedro\\Documents\\PROJETOS\\QML\\data\\dados_de_teste.csv"

    sampled_data = pd.read_csv(data_path).sample(3, random_state=42)

    ground_truth_labels = sampled_data["Attack_label"].tolist()

    sample_data_test = sampled_data.drop(["Attack_label"], axis=1)
    sample_data_test = sample_data_test.reset_index(drop=True)
    input_records = sample_data_test.to_dict(orient='records')
    inputs_for_crew = {
        'samples': input_records # Trocamos 'argument' por 'samples'
    }

    result = CyberPredict().crew().kickoff(inputs=inputs_for_crew)

    try:
        final_output_dict = result.raw

        # Agora, pegamos a lista de predições de dentro do dicionário
        print("\n\n--- COMPARAÇÃO (PREDICT vs. GROUND TRUTH) ---")
        print(f"Valores Reais (Ground Truth):     {ground_truth_labels}")
        print(f"Resultado Bruto da Crew (Predict): {final_output_dict}")
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