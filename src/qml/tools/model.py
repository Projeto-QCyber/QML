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
MODEL_MULTICLASS_CLASS_IDS = list(range(16))
APP_MULTICLASS_PROBABILITY_CLASS_IDS = [*range(15), 99]
MODEL_TO_APP_MULTICLASS_ID = {
    0: 99,   # NORMAL
    1: 0,    # Backdoor
    2: 1,    # DDoS_HTTP
    3: 2,    # DDoS_ICMP
    4: 3,    # DDoS_TCP
    5: 4,    # DDoS_UDP
    6: 5,    # Fingerprinting
    7: 6,    # MITM
    8: 7,    # Password
    9: 8,    # Port_Scanning
    10: 9,   # Ransomware
    11: 10,  # SQL_injection
    12: 11,  # Uploading
    13: 12,  # Vulnerability_scanner
    14: 13,  # XSS
    15: 14,  # Others
}
APP_TO_MODEL_MULTICLASS_ID = {
    app_id: model_id for model_id, app_id in MODEL_TO_APP_MULTICLASS_ID.items()
}


def _resolve_model_path(
    model_path: str | Path | None,
    classification: Literal["multiclass", "binary"],
) -> Path:
    if model_path is not None:
        return Path(model_path)

    if classification == "multiclass":
        env_path = os.getenv("QCYBER_MULTICLASS_MODEL_PATH") or os.getenv("QML_RF_MODEL_MULT_PATH")
        if env_path:
            return Path(env_path)
        return ROOT_DIR / "IA/weights/traditional/random_forest_model_mult.joblib"

    env_path = os.getenv("QCYBER_BINARY_MODEL_PATH") or os.getenv("QML_RF_MODEL_BIN_PATH")
    if env_path:
        return Path(env_path)
    return ROOT_DIR / "IA/weights/traditional/random_forest_model_bin.joblib"


def _load_model_cached(model_path: str | Path) -> object:
    resolved = str(Path(model_path).resolve())
    if resolved not in _MODEL_CACHE:
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=InconsistentVersionWarning)
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
        Recebe uma lista de amostras (dicionários) e retorna uma lista de predições.
    """
    args_schema: Type[BaseModel] = RFModelInput
    model: object
    classification: Literal["multiclass", "binary"] = "binary"

    def __init__(self, model_path: str = None, classification: Literal["multiclass", "binary"] = "binary", **kwargs):
        model_path = _resolve_model_path(model_path, classification)
        loaded_model = _load_model_cached(model_path)
        super().__init__(model=loaded_model, classification=classification, **kwargs)

    def _map_prediction(self, prediction: int) -> int:
        if self.classification == "multiclass":
            return MODEL_TO_APP_MULTICLASS_ID.get(prediction, prediction)
        return prediction

    def _map_probability_row(self, row: Any) -> list[float]:
        values = [float(value) for value in row]
        if self.classification != "multiclass":
            return values

        model_classes = [int(value) for value in getattr(self.model, "classes_", [])]
        if set(model_classes) != set(MODEL_MULTICLASS_CLASS_IDS):
            return values

        probabilities_by_class = {
            int(class_id): values[idx]
            for idx, class_id in enumerate(model_classes)
            if idx < len(values)
        }
        return [
            float(probabilities_by_class.get(APP_TO_MODEL_MULTICLASS_ID[class_id], 0.0))
            for class_id in APP_MULTICLASS_PROBABILITY_CLASS_IDS
        ]

    def predict(self, samples: List[Dict[str, Any]]) -> List[int]:
        df_input = _prepare_dataframe(self.model, samples)

        if os.getenv("QCYBER_MODEL_DEBUG", "").lower() in {"1", "true", "yes"}:
            print(f"[DEBUG RFModel] Shape entrada: {df_input.shape}")
            print(f"[DEBUG RFModel] Colunas: {list(df_input.columns)}")
            print(f"[DEBUG RFModel] Primeira linha:\n{df_input.head(1)}")

        model_input = df_input if hasattr(self.model, "feature_names_in_") else df_input.values
        preds = self.model.predict(model_input)
        return [self._map_prediction(int(p)) for p in preds]

    def predict_proba(self, samples: List[Dict[str, Any]]) -> list[list[float]] | None:
        if not hasattr(self.model, "predict_proba"):
            return None
        df_input = _prepare_dataframe(self.model, samples)
        model_input = df_input if hasattr(self.model, "feature_names_in_") else df_input.values
        probabilities = self.model.predict_proba(model_input)
        return [self._map_probability_row(row) for row in probabilities]

    def _run(self, samples: List[Dict[str, Any]]) -> List[int]:
        try:
            preds_list = self.predict(samples)
            if os.getenv("QCYBER_MODEL_DEBUG", "").lower() in {"1", "true", "yes"}:
                print(f"[DEBUG RFModel] Predições: {preds_list}")
            return preds_list

        except Exception as e:
            # Surface upstream; caller must decide fallback (not silently 'normal')
            raise RuntimeError(f"RFModel prediction failed: {e}") from e
