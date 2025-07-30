#### PROTOTIPO PARA DA API QUE RECEBE O DADO E ANALISA SE É ATAQUE OU NÃO


import pennylane as qml
from pennylane import numpy as np
import os
import json
from flask import Flask, request, jsonify
from sklearn.metrics import confusion_matrix  # <-- IMPORTANTE: Nova importação

# --- 1. CONFIGURAÇÃO E CARREGAMENTO DO MODELO ---

MODEL_DIR = "modelo"
print(">>> Iniciando servidor da API...")

config_path = os.path.join(MODEL_DIR, "vqc_config.json")
try:
    with open(config_path, "r") as f:
        config = json.load(f)
    num_qubits = config["num_qubits"]

    weights_path = os.path.join(MODEL_DIR, "vqc_weights.npy")
    bias_path = os.path.join(MODEL_DIR, "vqc_bias.npy")
    weights = np.load(weights_path)
    bias = np.load(bias_path)

    print(f">>> Modelo treinado com {num_qubits} qubits carregado com sucesso.")

except FileNotFoundError:
    print(f"ERRO: Arquivos do modelo não encontrados em '{MODEL_DIR}'. O servidor não pode iniciar.")
    exit()


# --- 2. DEFINIÇÃO DO CIRCUITO QUÂNTICO (VQC) ---

def statepreparation(x, num_qubits):
    qml.BasisEmbedding(x, wires=range(num_qubits))


def layer(W):
    qml.Rot(W[0, 0], W[0, 1], W[0, 2], wires=0)
    qml.Rot(W[1, 0], W[1, 1], W[1, 2], wires=1)
    qml.Rot(W[2, 0], W[2, 1], W[2, 2], wires=2)
    qml.Rot(W[3, 0], W[3, 1], W[3, 2], wires=3)
    qml.CNOT(wires=[0, 1]);
    qml.CNOT(wires=[1, 2]);
    qml.CNOT(wires=[2, 3]);
    qml.CNOT(wires=[3, 0])


dev = qml.device("default.qubit", wires=num_qubits)


@qml.qnode(dev, interface="autograd")
def circuit(weights_input, x_input):
    statepreparation(x_input, num_qubits)
    for W in weights_input:
        layer(W)
    return qml.expval(qml.PauliZ(0))


def variational_classifier(weights_input, bias_input, x_input):
    return circuit(weights_input, x_input) + bias_input


print(">>> Circuito Quântico (VQC) pronto.")

# --- 3. INICIALIZAÇÃO DA API FLASK ---

app = Flask(__name__)


# --- 4. CRIAÇÃO DO ENDPOINT DE PREDIÇÃO ---

@app.route("/api-quantum/predict", methods=["POST"])
def predict():
    if not request.is_json:
        return jsonify({"erro": "Requisição inválida. O corpo deve ser um JSON."}), 400

    data = request.get_json()
    X_test_list = data.get("X")
    y_test_list = data.get("y")

    if X_test_list is None or y_test_list is None:
        return jsonify({"erro": "Dados ausentes. O JSON deve conter as chaves 'X' e 'y'."}), 400

    if len(X_test_list[0]) != num_qubits:
        return jsonify({
            "erro": f"Incompatibilidade de dimensões. O modelo espera {num_qubits} features, mas recebeu {len(X_test_list[0])}."
        }), 400

    print(f"\n>>> Recebida requisição para predição com {len(X_test_list)} amostras.")

    X_test_np = np.array(X_test_list, requires_grad=False)
    predictions = [np.sign(variational_classifier(weights, bias, x)) for x in X_test_np]

    # --- LÓGICA DA MATRIZ DE CONFUSÃO ATUALIZADA ---
    # Calcula a acurácia
    correct_predictions = sum(p == l for p, l in zip(predictions, y_test_list))
    accuracy = correct_predictions / len(y_test_list) if y_test_list else 0

    # Calcula a matriz de confusão. As classes são -1 (Negativo) e 1 (Positivo)
    # confusion_matrix retorna [[TN, FP], [FN, TP]]
    try:
        cm = confusion_matrix(y_test_list, predictions, labels=[-1, 1])
        tn, fp, fn, tp = cm.ravel()
    except ValueError:
        # Caso de segurança se uma das classes não estiver presente nos dados
        tn, fp, fn, tp = 0, 0, 0, 0
        if len(y_test_list) > 0:
            # Lógica manual simples para o caso de apenas uma classe
            if y_test_list[0] == 1:  # Apenas positivos
                tp = sum(p == 1 for p in predictions)
                fn = len(predictions) - tp
            else:  # Apenas negativos
                tn = sum(p == -1 for p in predictions)
                fp = len(predictions) - tn

    # --- RESPOSTA JSON ATUALIZADA ---
    response_data = {
        "dados_recebidos": {
            "X": X_test_list,
            "y": y_test_list
        },
        "predicoes_vqc": [int(p) for p in predictions],
        "analise": {
            "acuracia": f"{accuracy:.2%}",
            "verdadeiros_positivos (previu 1, era 1)": int(tp),
            "verdadeiros_negativos (previu -1, era -1)": int(tn),
            "falsos_positivos (previu 1, era -1)": int(fp),
            "falsos_negativos (previu -1, era 1)": int(fn)
        }
    }

    print(f">>> Predição concluída. Acurácia: {accuracy:.2%}")
    return jsonify(response_data)


# --- 5. EXECUÇÃO DO SERVIDOR ---

if __name__ == "__main__":
    port = 5000
    print(f">>> Servidor Flask está sendo executado em http://127.0.0.1:{port}")
    print(">>> Pressione CTRL+C para encerrar.")
    app.run(host='0.0.0.0', port=port, debug=False)