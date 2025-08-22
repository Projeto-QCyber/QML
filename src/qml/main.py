from typing import Optional
from qml.crew import CyberPredict
from schemas.agents_traditional import Specialist

def run():
    """
    Run the research crew.
    """
    
    data_path = "./data/dados_de_teste.csv"
    """
    data = pd.read_csv(data_path).sample(10)
    sample_data_test = data.drop(["Attack_label"], axis=1)
    sample_data_test = sample_data_test.reset_index(drop=True)
    dictionary = sample_data_test.to_dict()
    inputs = {
        'argument': dictionary
    }
    """

    result = CyberPredict().crew().kickoff(inputs=data_path)

    # RECOMENDADO: dê name="validate_results" no Task para identificar fácil
    final_task = next(
        (t for t in result.tasks_output if (t.name == "validate_results") or
                                          (t.description and "Especialista valida os votos" in t.description)),
        None
    )

    if final_task is None:
        raise RuntimeError("Task 'validate_results' não foi encontrado em result.tasks_output.")

    # Aqui estará o objeto Pydantic Specialist (se o parsing deu certo)
    spec: Optional[Specialist] = getattr(final_task, "pydantic", None)

    print("\n\n=== FINAL RESULT ===\n")
    if spec is None:
        # fallback pra inspecionar o que veio cru
        print("Não foi possível construir o Pydantic 'Specialist'. Conteúdo bruto do task:")
        print(final_task.raw)              # texto puro gerado pelo LLM
        print("\nJSON parseado (se disponível):", getattr(final_task, "json_dict", None))
    else:
        print("Predictions:", spec.predictions)
        print("Report:", spec.report)

if __name__ == "__main__":
    run()