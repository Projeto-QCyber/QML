import os
import joblib
import warnings
import pandas as pd
from pathlib import Path
from typing import List, Dict, Any, Literal, Type
from crewai.tools import BaseTool
from pydantic import BaseModel, Field
from sklearn.exceptions import InconsistentVersionWarning


ROOT_DIR = Path(__file__).resolve().parents[3]
_MODEL_CACHE: dict[str, object] = {}


def _resolve_model_path(
    model_path: str | Path | None,
    classification: Literal["multiclass", "binary"],
) -> Path:
    if classification == "multiclass":
        return ROOT_DIR / "IA/weights/traditional/random_forest_model_mult.joblib"
    if model_path is None or classification == "binary":
        return ROOT_DIR / "IA/weights/traditional/random_forest_model_bin.joblib"
    return Path(model_path)


def _load_model_cached(model_path: str | Path) -> object:
    resolved = str(Path(model_path).resolve())
    if resolved not in _MODEL_CACHE:
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=InconsistentVersionWarning) # Ignorar mismatch entre o modelo binário salvo com scikit-learn=1.6.1 e o modelo multiclasse salvo com scikit-learn=1.7.1
            _MODEL_CACHE[resolved] = joblib.load(resolved)
    return _MODEL_CACHE[resolved]


def _prepare_dataframe(model: object, samples: List[Dict[str, Any]]) -> pd.DataFrame:
    df_input = pd.DataFrame(samples)
    if df_input.empty:
        raise ValueError("Nenhuma amostra foi enviada para predição.")

    if hasattr(model, "feature_names_in_"):
        missing = set(model.feature_names_in_) - set(df_input.columns)
        if missing:
            missing_preview = ", ".join(sorted(missing)[:10])
            raise ValueError(f"Faltam colunas obrigatórias no input: {missing_preview}")
        df_input = df_input.reindex(columns=model.feature_names_in_)

    return df_input.astype(float)

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
        model_path = _resolve_model_path(model_path, classification)
        loaded_model = _load_model_cached(model_path)
        super().__init__(model=loaded_model, **kwargs)

    def predict(self, samples: List[Dict[str, Any]]) -> List[int]:
        df_input = _prepare_dataframe(self.model, samples)

        if os.getenv("QCYBER_MODEL_DEBUG", "").lower() in {"1", "true", "yes"}:
            print(f"[DEBUG RFModel] Shape entrada: {df_input.shape}")
            print(f"[DEBUG RFModel] Colunas: {list(df_input.columns)}")
            print(f"[DEBUG RFModel] Primeira linha:\n{df_input.head(1)}")

        model_input = df_input if hasattr(self.model, "feature_names_in_") else df_input.values
        preds = self.model.predict(model_input)
        return [int(p) for p in preds]

    def predict_proba(self, samples: List[Dict[str, Any]]) -> list[list[float]] | None:
        if not hasattr(self.model, "predict_proba"):
            return None
        df_input = _prepare_dataframe(self.model, samples)
        model_input = df_input if hasattr(self.model, "feature_names_in_") else df_input.values
        probabilities = self.model.predict_proba(model_input)
        return [[float(value) for value in row] for row in probabilities]

    def _run(self, samples: List[Dict[str, Any]]) -> List[int]:
        try:
            preds_list = self.predict(samples)
            if os.getenv("QCYBER_MODEL_DEBUG", "").lower() in {"1", "true", "yes"}:
                print(f"[DEBUG RFModel] Predições: {preds_list}")
            return preds_list

        except Exception as e:
            # Surface upstream; caller must decide fallback (not silently 'normal')
            raise RuntimeError(f"RFModel prediction failed: {e}") from e
