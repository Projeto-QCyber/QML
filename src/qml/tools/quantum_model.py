from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Type

import numpy as np
import pandas as pd
from crewai.tools import BaseTool
from pydantic import BaseModel, Field, PrivateAttr


ROOT_DIR = Path(__file__).resolve().parents[3]
DEFAULT_MODEL_STEM = ROOT_DIR / "IA/weights/quantum/quantum_angle_embedding_y_ring_rot_cnot_L4_s42_8806630d6b"
DEFAULT_MODEL_PATH = DEFAULT_MODEL_STEM.with_suffix(".npz")
PREDICT_SCRIPT_PATH = ROOT_DIR / "IA/scripts/predict.py"


class QuantumModelInput(BaseModel):
    samples: List[Dict[str, Any]] = Field(..., description="Lista de amostras para predição.")


class QuantumModel(BaseTool):
    name: str = "quantum_model"
    description: str = """
        Ferramenta de classificação binária com modelo quântico (.npz/.json ou .pt legado).
        Recebe uma lista de amostras (dicionários) e retorna predições 0/1.
    """
    args_schema: Type[BaseModel] = QuantumModelInput
    model_path: str = str(DEFAULT_MODEL_PATH)
    device: Optional[str] = None

    _classifier: Any = PrivateAttr(default=None)
    _feature_names: List[str] = PrivateAttr(default_factory=list)

    def _resolve_model_path(self) -> Path:
        model_file = Path(self.model_path)
        if model_file.exists():
            return model_file

        json_sidecar = model_file.with_suffix(".json")
        if model_file.suffix.lower() == ".npz" and json_sidecar.exists():
            return model_file

        legacy_pt = model_file.with_suffix(".pt")
        if legacy_pt.exists():
            return legacy_pt

        quantum_dir = ROOT_DIR / "IA/weights/quantum"
        candidates = sorted(quantum_dir.glob("*.npz"))
        if len(candidates) == 1:
            return candidates[0]
        if len(candidates) > 1:
            raise FileNotFoundError(
                f"Configured model path not found: '{model_file}'. Multiple .npz candidates found; "
                f"pass model_path explicitly. Candidates: {[str(p) for p in candidates]}"
            )

        candidates = sorted(quantum_dir.glob("*.pt"))
        if len(candidates) == 1:
            return candidates[0]
        if len(candidates) > 1:
            raise FileNotFoundError(
                f"Configured model path not found: '{model_file}'. Multiple .pt candidates found; "
                f"pass model_path explicitly. Candidates: {[str(p) for p in candidates]}"
            )
        raise FileNotFoundError(f"Quantum model file not found at '{model_file}'.")

    def __init__(self, model_path: Optional[str] = None, device: Optional[str] = None, **kwargs):
        resolved_path = Path(model_path) if model_path else DEFAULT_MODEL_PATH
        super().__init__(model_path=str(resolved_path), device=device, **kwargs)

    def _load_predict_module(self):
        if not PREDICT_SCRIPT_PATH.exists():
            raise FileNotFoundError(f"predict.py not found at '{PREDICT_SCRIPT_PATH}'.")

        module_name = "qml_portable_predict"
        spec = importlib.util.spec_from_file_location(module_name, str(PREDICT_SCRIPT_PATH))
        if spec is None or spec.loader is None:
            raise RuntimeError(f"Could not create import spec for '{PREDICT_SCRIPT_PATH}'.")

        module = importlib.util.module_from_spec(spec)
        # Required for dataclass/type resolution during module execution on py3.12.
        sys.modules[module_name] = module
        spec.loader.exec_module(module)
        return module

    def _ensure_loaded(self) -> None:
        if self._classifier is not None:
            return

        model_file = self._resolve_model_path()

        predict_module = self._load_predict_module()
        load_quantum_model = getattr(predict_module, "load_quantum_model", None)
        if load_quantum_model is None:
            raise RuntimeError("predict.py does not expose 'load_quantum_model'.")

        self._classifier = load_quantum_model(str(model_file), device_override=self.device)
        self._feature_names = list(getattr(self._classifier, "features", []) or [])

    def _run(self, samples: List[Dict[str, Any]]) -> List[int]:
        try:
            self._ensure_loaded()

            df_input = pd.DataFrame(samples)
            if df_input.empty:
                return []

            # Remove labels accidentally present in payload.
            df_input = df_input.drop(
                columns=[c for c in df_input.columns if str(c).strip().lower() in {"attack_label", "attack_type", "label"}],
                errors="ignore",
            )

            if self._feature_names:
                missing = [c for c in self._feature_names if c not in df_input.columns]
                if missing:
                    raise ValueError(f"Missing required feature columns: {missing}")
                df_input = df_input.reindex(columns=self._feature_names)

            preds = self._classifier.predict01(df_input)
            preds_np = np.asarray(preds, dtype=np.int32).reshape(-1)
            return [int(p) for p in preds_np.tolist()]

        except Exception as e:
            raise RuntimeError(f"quantum_model prediction failed: {e}") from e
