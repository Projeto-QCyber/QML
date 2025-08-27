import joblib
import pandas as pd
from typing import List, Dict, Any, Type
from crewai.tools import BaseTool
from pydantic import BaseModel, Field

class RFModelInput(BaseModel):
    """Input schema for RFModel for batch predictions."""
    samples: List[Dict[str, Any]] = Field(..., description="A list of network data samples (as dictionaries) to be used for prediction.")

class RFModel(BaseTool):
    name: str = "Model"
    description: str = """
        This tool uses a pre-trained Random Forest model for binary classification.
        It accepts a complete list of data samples (dictionaries) and returns a list of binary predictions (0 or 1).
    """
    args_schema: Type[BaseModel] = RFModelInput
    model: object

    def __init__(self, model_path: str = None, **kwargs):
        # 1. Primeiro, carregamos o modelo joblib em uma variável
        if model_path is None:
            model_path = "IA/weights/traditional/random_forest_model.joblib"
        loaded_model = joblib.load(model_path)

        # A ferramenta RFModel herda todos os métodos do modelo carregado usando joblib
        super().__init__(model=loaded_model, **kwargs)

    def _run(self, samples: List[Dict[str, Any]]) -> str:
        try:
            df_input = pd.DataFrame(samples)
            
            # Linhas de Debug (opcionais, mas úteis)
            # print(f"DEBUG: Ferramenta recebeu um lote de {len(df_input)} amostras.")
            # print(f"DEBUG: Colunas para predição: {df_input.columns.tolist()}")

            predictions = self.model.predict(df_input)
            predictions_list = [int(p) for p in predictions]

            return str(predictions_list)

        except Exception as e:
            return f"Error during batch prediction: {str(e)}"