import operator
from pennylane import numpy as np
from typing import TypedDict, Annotated, List
from langchain_ollama.chat_models import ChatOllama
from langchain_core.messages import HumanMessage, ToolMessage, SystemMessage
from langchain_core.tools import tool
from langgraph.graph import StateGraph, END
from sklearn.metrics import confusion_matrix

from langchain.schema import HumanMessage, AIMessage
import re
import uuid

# IP DO RASP
# 192.168.8.65
#
# Criar o executor que saberá como chamar a ferramenta
tools = [vqc_predict]

class Agent:
    def

# --- 1) Estado do agente ---
class AgentState(TypedDict):
    messages: Annotated[List, operator.add]
    step_count: int

# --- 2) Modelo SEM bind_tools ---
model = ChatOllama(model="deepseek-r1:7b", temperature=0, streaming=True)

def extract_data_from_message(content: str):
    """Extrai X e y da mensagem do usuário usando regex"""
    # Procura por padrões como y=[1,-1] ou y = [1, -1]
    y_pattern = r'y\s*=\s*\[([-\d\s,\.]+)\]'
    y_match = re.search(y_pattern, content)
    
    # Procura por padrões como X=[[...],[...]] ou X = [[...], [...]]
    x_pattern = r'X\s*=\s*\[(\[[-\d\s,\.]+\](?:\s*,\s*\[[-\d\s,\.]+\])*)\]'
    x_match = re.search(x_pattern, content)
    
    if y_match and x_match:
        try:
            # Extrai y
            y_str = y_match.group(1)
            y = [float(x.strip()) for x in y_str.split(',')]
            
            # Extrai X
            x_str = x_match.group(1)
            # Remove espaços e quebra em arrays individuais
            x_arrays = re.findall(r'\[([-\d\s,\.]+)\]', x_str)
            X = []
            for arr_str in x_arrays:
                arr = [float(x.strip()) for x in arr_str.split(',')]
                X.append(arr)
            
            return X, y
        except (ValueError, AttributeError):
            return None, None
    
    return None, None

def agent_node(state: AgentState):
    # Se é a primeira execução, processa a mensagem do usuário
    if state.get("step_count", 0) == 0:
        last_message = state["messages"][-1]
        if hasattr(last_message, 'content'):
            X, y = extract_data_from_message(last_message.content)
            if X is not None and y is not None:
                # Cria uma mensagem AI com tool call para o VQC
                tool_call_id = str(uuid.uuid4())
                tool_call = {
                    "name": "vqc_predict",
                    "args": {"y_test_dummy": y, "X_test": X},
                    "id": tool_call_id
                }
                response = AIMessage(content="", tool_calls=[tool_call])
            else:
                response = AIMessage(content="Não foi possível extrair X e y da mensagem. Certifique-se de usar o formato: y=[1,-1], X=[[0.1,0.2,0,0],[0,0.1,0.2,0.1]]")
        else:
            response = AIMessage(content="Mensagem inválida.")
    else:
        # Para execuções subsequentes, usa o modelo normalmente
        response = model.invoke(state["messages"])
    
    step_count = state.get("step_count", 0) + 1
    return {"messages": [response], "step_count": step_count}

# --- 5) Nó da ferramenta VQC ---
def tool_node(state: AgentState):
    last_message = state["messages"][-1]
    
    if not hasattr(last_message, 'tool_calls') or not last_message.tool_calls:
        return {"messages": [ToolMessage(content="Nenhuma tool call encontrada", tool_call_id="error")]}
    
    tool_call = last_message.tool_calls[0]
    print(f"Executando tool call: {tool_call['name']} com args: {tool_call['args']}")
    
    try:
        # Chama sua função vqc_predict diretamente (assumindo que ela está disponível)
        # A função vqc_predict retorna uma string, não precisa ser invocada como tool
        result = vqc_predict.func(
            y_test_dummy=tool_call["args"]["y_test_dummy"],
            X_test=tool_call["args"]["X_test"]
        )
        
        msg = ToolMessage(content=result, tool_call_id=tool_call["id"])
        
    except Exception as e:
        print(f"Erro ao executar VQC: {e}")
        msg = ToolMessage(content=f"Erro ao executar VQC: {str(e)}", tool_call_id=tool_call["id"])
    
    return {"messages": [msg]}

# --- 6) Função de roteamento condicional ---
def should_continue(state: AgentState):
    """Decide se deve continuar ou finalizar"""
    step_count = state.get("step_count", 0)
    
    # Limite de segurança
    if step_count >= 5:
        print("Limite de passos atingido, finalizando...")
        return END
    
    if not state["messages"]:
        return "tool"
    
    last_message = state["messages"][-1]
    
    # Se a última mensagem é uma ToolMessage, finaliza
    if isinstance(last_message, ToolMessage):
        return END
    
    # Se a última mensagem tem tool_calls, vai para a ferramenta
    if hasattr(last_message, 'tool_calls') and last_message.tool_calls:
        return "tool"
    
    return END

# --- 7) Montagem do grafo ---
builder = StateGraph(AgentState)
builder.add_node("agent", agent_node)
builder.add_node("tool", tool_node)
builder.set_entry_point("agent")

builder.add_conditional_edges(
    "agent",
    should_continue,
    {
        "tool": "tool",
        END: END
    }
)

builder.add_edge("tool", "agent")
graph = builder.compile()