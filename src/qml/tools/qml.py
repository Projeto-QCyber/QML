import joblib
import pandas as pd
from pennylane import numpy as np
from typing import List, Dict, Any, Type
from crewai.tools import BaseTool
from pydantic import BaseModel, Field

from IA.models.vqc import VQC

class VQCModelInput(BaseModel):
    """Input schema for VQC model."""
    samples: List[Dict[str, Any]] = Field(..., description="A list of network data samples (as dictionaries) to be used for prediction.")

class VQCModel(BaseTool):
    name: str = "Variational Quantum Classifier"
    description: str = """
        This tool uses a pre-trained Variational Classifier model for binary classification based on quantum computing.
        It accepts a complete list of data samples (dictionaries) and returns a list of binary predictions (0 or 1).
    """
    args_schema: Type[BaseModel] = VQCModelInput
    model: object

    def __init__(self, model_path:str = None, **kwargs):
        if model_path is None:
            model_path = "IA/weights/quantum/variational_classifier_model.joblib"
        loaded_model = joblib.load(model_path)

        self.loaded_weights = loaded_model['weights']
        self.loaded_bias = loaded_model['bias']
        self.n_qubits = loaded_model['n_qubits']
        self.n_layers = loaded_model['n_layers']

        super().__init__(model=loaded_model, **kwargs)
        
    def _run(self, samples: List[Dict[str, Any]]) -> str:
        try:
            df_input = pd.DataFrame(samples)
            
            # Linhas de Debug (opcionais, mas úteis)
            # print(f"DEBUG: Ferramenta recebeu um lote de {len(df_input)} amostras.")
            # print(f"DEBUG: Colunas para predição: {df_input.columns.tolist()}")
            variational_classifier = VQC(
                n_qubits=self.n_qubits,
                n_layers=self.n_layers,
                data_shape=df_input.shape
            ).variational_classifier

            # Supondo também que df_input.values seja um batch de dados e 2D
            predictions = [np.sign(variational_classifier(self.loaded_weights, self.loaded_bias, x)) for x in df_input.values]
            predictions_list = [int(p) for p in predictions]

            return str(predictions_list)

        except Exception as e:
            return f"Error during batch prediction: {str(e)}"