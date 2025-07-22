import operator
from pennylane import numpy as np
from typing import List
from langchain_core.tools import tool
from sklearn.metrics import confusion_matrix

@tool
def vqc_predict(
    y_test_dummy: List[int],
    X_test: List[List[float]],
    n_layers,
    n_qubits):
    """
    Executa o Classificador Quântico Variacional (VQC) no conjunto de dados de teste pré-carregado.
    Calcula e retorna um resumo dos resultados, incluindo acurácia e uma análise simples da matriz de confusão.
    Use esta ferramenta quando o usuário pedir para realizar uma predição, rodar o teste ou avaliar o modelo quântico.
    """
    print("\n--- Executando a ferramenta de predição VQC... ---")
    weights_init = 0.01 * np.random.randn(n_layers, n_qubits, 3, requires_grad=True)
    bias_init = np.array(0.0, requires_grad=True)

    weights = weights_init
    bias = bias_init
    
    # Esta é a sua lógica de predição
    predictions = [np.sign(variational_classifier(weights, bias, x)) for x in X_test]

    # Calcular métricas para uma resposta mais rica
    accuracy = accuracy_score(y_test_dummy, predictions)
    cm = confusion_matrix(y_test_dummy, predictions)
    
    # Extrair os valores da matriz de confusão
    # Assumindo que as classes são -1 (negativo) e 1 (positivo)
    # e que confusion_matrix as ordena como 0 (-1) e 1 (1)
    tn, fp, fn, tp = cm.ravel() if len(cm.ravel()) == 4 else (cm[0,0], 0, 0, 0)

    # Formatar a resposta para o LLM
    response = (
        f"A predição com o VQC foi concluída com sucesso.\n"
        f"Acurácia no conjunto de teste: {accuracy:.2%}\n"
        f"Resultados:\n"
        f"  - Verdadeiros Positivos (previu 1, era 1): {tp}\n"
        f"  - Verdadeiros Negativos (previu 0, era 0): {tn}\n"
        f"  - Falsos Positivos (previu 1, era 0): {fp}\n"
        f"  - Falsos Negativos (previu 0, era 1): {fn}\n"
    )
    print("--- Ferramenta concluiu a execução. ---\n")
    return response