import pennylane as qml
from pennylane import numpy as np
from pennylane.optimize import AdamOptimizer
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix
import math
import json
import os
import re
import uuid
from typing import TypedDict, Annotated, List, operator

# Imports para o Agente LangChain (mantenha os seus se forem diferentes)
from langchain_core.messages import HumanMessage, ToolMessage, SystemMessage, AIMessage
from langchain_core.tools import tool
from langgraph.graph import StateGraph, END

# --- PARTE 0: DEFINIÇÃO DO DIRETÓRIO DO MODELO ---
MODEL_DIR = "modelo"


# --- PARTE 1: DEFINIÇÕES GLOBAIS DO VQC ---
# Estas definições são necessárias tanto para o treino quanto para a predição.

def statepreparation(x, num_qubits):
    """Codifica os dados de entrada nos estados quânticos."""
    qml.BasisEmbedding(x, wires=range(num_qubits))


def layer(W):
    """Define uma camada do circuito variacional."""
    qml.Rot(W[0, 0], W[0, 1], W[0, 2], wires=0)
    qml.Rot(W[1, 0], W[1, 1], W[1, 2], wires=1)
    qml.Rot(W[2, 0], W[2, 1], W[2, 2], wires=2)
    qml.Rot(W[3, 0], W[3, 1], W[3, 2], wires=3)

    qml.CNOT(wires=[0, 1])
    qml.CNOT(wires=[1, 2])
    qml.CNOT(wires=[2, 3])
    qml.CNOT(wires=[3, 0])


# O QNode será definido dinamicamente após carregar a configuração
circuit = None
dev = None


def variational_classifier(weights, bias, x):
    """Cria o classificador quântico variacional completo."""
    if circuit is None:
        raise RuntimeError("O circuito quântico (QNode) não foi inicializado.")
    return circuit(weights, x) + bias


# --- PARTE 2: FUNÇÃO DE TREINAMENTO E SALVAMENTO DO MODELO ---

def train_and_save_model():
    """
    Carrega os dados, treina o modelo VQC e salva os pesos, bias e configuração
    dentro do diretório especificado em MODEL_DIR.
    """
    print("--- Iniciando Treinamento do Modelo VQC ---")

    # Carregando e preparando os dados (Titanic)
    # ATENÇÃO: Atualize o caminho para o seu arquivo de dados, se necessário.
    df_train = pd.read_csv("D:/Documentos/Projetos/QML/data/train.csv")
    df_train['Pclass'] = df_train['Pclass'].astype(str)
    df_train = pd.concat([df_train, pd.get_dummies(df_train[['Pclass', 'Sex', 'Embarked']])], axis=1)
    df_train['Age'] = df_train['Age'].fillna(df_train['Age'].median())
    df_train['is_child'] = df_train['Age'].map(lambda x: 1 if x < 12 else 0)

    cols_model = ['is_child', 'Pclass_1', 'Pclass_2', 'Pclass_3']
    x_data = df_train[cols_model]
    y_data = df_train['Survived']

    X_train, _, y_train, _ = train_test_split(x_data, y_data, test_size=0.10, random_state=42, stratify=y_data)

    X_train = np.array(X_train.values, requires_grad=False)
    Y_train = np.array(y_train.values * 2 - np.ones(len(y_train)), requires_grad=False)

    # Parâmetros do modelo baseados nos dados
    num_qubits = X_train.shape[1]
    num_layers = 3

    # Define o dispositivo e o QNode para o treinamento
    global dev, circuit
    dev = qml.device("default.qubit", wires=num_qubits)

    @qml.qnode(dev, interface="autograd")
    def training_circuit(weights, x):
        statepreparation(x, num_qubits)
        for W in weights:
            layer(W)
        return qml.expval(qml.PauliZ(0))

    circuit = training_circuit

    # Funções de custo e acurácia
    def square_loss(labels, predictions):
        return np.mean((labels - predictions) ** 2)

    def cost(weights, bias, X, Y):
        predictions = [variational_classifier(weights, bias, x) for x in X]
        return square_loss(Y, predictions)

    # Loop de treinamento
    np.random.seed(0)
    weights_init = 0.01 * np.random.randn(num_layers, num_qubits, 3, requires_grad=True)
    bias_init = np.array(0.0, requires_grad=True)

    opt = AdamOptimizer(0.125)
    num_it = 30 ### Alterar para melhorar o treinamento
    batch_size = 32

    weights = weights_init
    bias = bias_init

    for it in range(num_it):
        batch_index = np.random.randint(0, len(X_train), (batch_size,))
        X_batch, Y_batch = X_train[batch_index], Y_train[batch_index]
        weights, bias, _, _ = opt.step(cost, weights, bias, X_batch, Y_batch)

        if (it + 1) % 10 == 0:
            current_cost = cost(weights, bias, X_train, Y_train)
            print(f"Iter: {it + 1:3d} | Custo: {current_cost:.7f}")

    print("--- Treinamento Concluído ---")

    # >>> NOVO: Criar o diretório do modelo se não existir
    os.makedirs(MODEL_DIR, exist_ok=True)
    print(f"Salvando modelo treinado no diretório '{MODEL_DIR}'...")

    # >>> NOVO: Salvar arquivos dentro do diretório
    np.save(os.path.join(MODEL_DIR, "vqc_weights.npy"), weights.unwrap())
    np.save(os.path.join(MODEL_DIR, "vqc_bias.npy"), bias.unwrap())

    model_config = {"num_qubits": num_qubits, "num_layers": num_layers}
    with open(os.path.join(MODEL_DIR, "vqc_config.json"), "w") as f:
        json.dump(model_config, f)

    print("Modelo salvo com sucesso!")


# --- PARTE 3: FERRAMENTA DE PREDIÇÃO (LANGCHAIN TOOL) ---

@tool
def vqc_predict(y_test_dummy: List[int], X_test: List[List[float]]):
    """
    Carrega um modelo VQC pré-treinado do diretório 'modelo' e executa predições.
    """
    print(f"\n--- Carregando modelo VQC do diretório '{MODEL_DIR}' e executando predição ---")

    # >>> NOVO: Caminhos para os arquivos do modelo
    config_path = os.path.join(MODEL_DIR, "vqc_config.json")
    weights_path = os.path.join(MODEL_DIR, "vqc_weights.npy")
    bias_path = os.path.join(MODEL_DIR, "vqc_bias.npy")

    try:
        # Carrega a configuração e os parâmetros do modelo
        with open(config_path, "r") as f:
            config = json.load(f)
        weights = np.load(weights_path)
        bias = np.load(bias_path)
        num_qubits = config["num_qubits"]

        if len(X_test[0]) != num_qubits:
            return f"Erro: O modelo foi treinado com {num_qubits} features, mas os dados fornecidos têm {len(X_test[0])}."

        global dev, circuit
        dev = qml.device("default.qubit", wires=num_qubits)

        @qml.qnode(dev, interface="autograd")
        def prediction_circuit(weights, x):
            statepreparation(x, num_qubits)
            for W in weights:
                layer(W)
            return qml.expval(qml.PauliZ(0))

        circuit = prediction_circuit

    except FileNotFoundError:
        return f"Erro: Arquivos do modelo não encontrados no diretório '{MODEL_DIR}'. Por favor, execute o treinamento primeiro."

    X_test_np = np.array(X_test, requires_grad=False)
    predictions = [np.sign(variational_classifier(weights, bias, x)) for x in X_test_np]

    accuracy = accuracy_score(y_test_dummy, predictions)
    cm = confusion_matrix(y_test_dummy, predictions)
    tn, fp, fn, tp = cm.ravel() if len(cm.ravel()) == 4 else (cm[0, 0], 0, 0, 0)

    response = (
        f"Predição com VQC concluída.\n"
        f"Acurácia: {accuracy:.2%}\n"
        f"Resultados:\n"
        f"  - Verdadeiros Positivos (previu 1, era 1): {tp}\n"
        f"  - Verdadeiros Negativos (previu -1, era -1): {tn}\n"
        f"  - Falsos Positivos (previu 1, era -1): {fp}\n"
        f"  - Falsos Negativos (previu -1, era 1): {fn}\n"
    )
    print("--- Ferramenta de predição concluiu. ---")
    return response


# --- PARTE 4: LÓGICA DO AGENTE LANGGRAPH ---
# (Nenhuma mudança necessária aqui)
class AgentState(TypedDict):
    messages: Annotated[List, operator.add]
    step_count: int


def extract_data_from_message(content: str):
    y_match = re.search(r'y\s*=\s*(\[.*?\])', content)
    x_match = re.search(r'X\s*=\s*(\[\[.*?\]\])', content)
    if y_match and x_match:
        try:
            y = json.loads(y_match.group(1))
            X = json.loads(x_match.group(1))
            return X, y
        except json.JSONDecodeError:
            return None, None
    return None, None


def agent_node(state: AgentState):
    if state.get("step_count", 0) == 0:
        last_message = state["messages"][-1]
        X, y = extract_data_from_message(last_message.content)
        if X is not None and y is not None:
            tool_call = {
                "name": "vqc_predict",
                "args": {"y_test_dummy": y, "X_test": X},
                "id": str(uuid.uuid4())
            }
            response = AIMessage(content="", tool_calls=[tool_call])
        else:
            response = AIMessage(
                content="Formato de entrada inválido. Use: Faça predições com VQC para y=[1,-1], X=[[0,0,1,0],[0,0,1,0]]")
    else:
        response = AIMessage(content="Processo finalizado.")

    step_count = state.get("step_count", 0) + 1
    return {"messages": [response], "step_count": step_count}


def tool_node(state: AgentState):
    last_message = state["messages"][-1]
    tool_call = last_message.tool_calls[0]
    result = vqc_predict.invoke(tool_call["args"])
    msg = ToolMessage(content=result, tool_call_id=tool_call["id"])
    return {"messages": [msg]}


def should_continue(state: AgentState):
    if isinstance(state["messages"][-1], ToolMessage):
        return END
    if hasattr(state["messages"][-1], 'tool_calls') and state["messages"][-1].tool_calls:
        return "tool"
    return END


builder = StateGraph(AgentState)
builder.add_node("agent", agent_node)
builder.add_node("tool", tool_node)
builder.set_entry_point("agent")
builder.add_conditional_edges("agent", should_continue, {"tool": "tool", END: END})
builder.add_edge("tool", END)
graph = builder.compile()

# --- PARTE 5: EXECUÇÃO PRINCIPAL ---
if __name__ == "__main__":

    # Passo 1: Treinar e salvar o modelo.
    # Descomente a linha abaixo e execute o script UMA VEZ.
    # --------------------------------------------------------------------
    # train_and_save_model()
    # --------------------------------------------------------------------

    # >>> NOVO: Verifica se o arquivo de config existe DENTRO do diretório
    config_path = os.path.join(MODEL_DIR, "vqc_config.json")
    if not os.path.exists(config_path):
        print(
            f"🚨 Modelo não encontrado em '{MODEL_DIR}'! Por favor, descomente e execute 'train_and_save_model()' no código.")
    else:
        # Passo 2: Usar o agente para fazer predições com o modelo salvo.
        print(f"\n🤖 Agente de Predição VQC pronto (usando modelo de '{MODEL_DIR}').")
        print("--------------------------------------------------")

        user_input = "Faça predições com VQC para y=[1,-1], X=[[0,0,1,0],[0,0,1,0]]"

        initial_state = {"messages": [HumanMessage(content=user_input)], "step_count": 0}
        final_state = graph.invoke(initial_state, {"recursion_limit": 5})

        result_message = final_state['messages'][-1]
        print(result_message.content)