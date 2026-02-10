import joblib
import pandas as pd
from pathlib import Path
from typing import List, Dict, Any, Literal, Type
from crewai.tools import BaseTool
from pydantic import BaseModel, Field


ROOT_DIR = Path(__file__).resolve().parents[3]

class RFModelInput(BaseModel):
    samples: List[Dict[str, Any]] = Field(..., description="Lista de amostras para predição.")

class RFModel(BaseTool):
    name: str = "Model"
    description: str = """
        Ferramenta que usa um Random Forest pré-treinado.
        Recebe uma lista de amostras (dicionários) e retorna lista de predições 0/1.
    """
    args_schema: Type[BaseModel] = RFModelInput
    model: object

    def __init__(self, model_path: str = None, classification: Literal["multiclass", "binary"] = "binary", **kwargs):
        if classification == "multiclass":
            model_path = ROOT_DIR / "IA/weights/traditional/random_forest_model_mult_q.joblib"
        elif classification == "binary" and model_path is None:
            model_path = ROOT_DIR / "IA/weights/traditional/random_forest_model_bin_q.joblib"

        loaded_model = joblib.load(model_path)
        super().__init__(model=loaded_model, **kwargs)

    def _run(self, samples: List[Dict[str, Any]]) -> List[int]:
        try:
            df_input = pd.DataFrame(samples)

            # Remove label / metadata columns that are not model features
            drop_cols = [c for c in df_input.columns
                         if c.lower() in {"attack_label", "attack_type", "split"}]
            df_input = df_input.drop(columns=drop_cols, errors="ignore")

            # Garante que as colunas estão na ordem usada no treino
            if hasattr(self.model, "feature_names_in_"):
                missing = set(self.model.feature_names_in_) - set(df_input.columns)
                if missing:
                    print(f"[ERRO RFModel] Faltam colunas no input: {missing}")
                df_input = df_input.reindex(columns=self.model.feature_names_in_)

            # Converte para float (importante se vierem strings)
            df_input = df_input.astype(float)

            # Debug para verificar se a entrada está correta
            print(f"[DEBUG RFModel] Shape entrada: {df_input.shape}")
            print(f"[DEBUG RFModel] Colunas: {list(df_input.columns)}")
            print(f"[DEBUG RFModel] Primeira linha:\n{df_input.head(1)}")

            preds = self.model.predict(df_input)
            preds_list = [int(p) for p in preds]

            print(f"[DEBUG RFModel] Predições: {preds_list}")
            return preds_list

        except Exception as e:
            # Surface upstream; caller must decide fallback (not silently 'normal')
            raise RuntimeError(f"RFModel prediction failed: {e}") from e