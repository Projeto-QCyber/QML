import joblib
import pandas as pd
import numpy as np
import time
from pathlib import Path
from typing import List, Dict, Any, Literal, Type, Tuple
from crewai.tools import BaseTool
from pydantic import BaseModel, Field, PrivateAttr
import shap
import os


ROOT_DIR = Path(__file__).resolve().parents[3]

# Simple in-process caches to avoid reloading model and rebuilding the SHAP explainer
_MODEL_CACHE: dict[str, object] = {}
_EXPLAINER_CACHE: dict[str, shap.TreeExplainer] = {}


class ExplainInput(BaseModel):
    """Input schema for Top-2 feature attribution tool."""
    samples: List[Dict[str, Any]] = Field(
        ..., description="A list of network data samples (dicts) for attribution."
    )


class ExplainTop2SHAP(BaseTool):
    name: str = "ExplainTop2"
    description: str = (
        "Compute per-sample top-2 influential inputs using SHAP for the Random Forest model. "
        "Returns concise lines ready for the leader, e.g.: \n"
        "[1] tcp.flags, dst_port seem far from normal."
    )
    args_schema: Type[BaseModel] = ExplainInput
    model: object
    # Pydantic private attrs (won't be stripped/reset by BaseModel)
    _mode: str = PrivateAttr(default="on")
    _is_multiclass: bool = PrivateAttr(default=False)
    _explainer: shap.TreeExplainer | None = PrivateAttr(default=None)

    def __init__(
        self,
        model_path: str | None = None,
        classification: Literal["multiclass", "binary"] = "binary",
        mode: Literal["on", "off", "fast"] | None = None,
        **kwargs,
    ) -> None:
        # Resolve mode from env if not provided (assign to attr after BaseModel init)
        env_mode = os.getenv("QCYBER_SHAP_MODE", "on").strip().lower()

        if classification == "multiclass":
            model_path = ROOT_DIR / "IA/weights/traditional/random_forest_model_mult.joblib"
        elif classification == "binary" and model_path is None:
            model_path = ROOT_DIR / "IA/weights/traditional/random_forest_model_bin.joblib"

        key = f"{classification}:{Path(model_path).resolve()}"

        # Load model once
        if key not in _MODEL_CACHE:
            print(f"[ExplainTop2] Loading model from {model_path}")
            t_load = time.time()
            _MODEL_CACHE[key] = joblib.load(model_path)
            print(f"[ExplainTop2] Model loaded in {time.time() - t_load:.2f}s")
        loaded_model = _MODEL_CACHE[key]
        super().__init__(model=loaded_model, **kwargs)

        # Set private attrs after BaseModel init
        self._mode = (mode or env_mode)
        if self._mode not in ("on", "off", "fast"):
            self._mode = "on"
        self._is_multiclass = classification == "multiclass"
        self._explainer = None
        if self._mode == "on":
            if key not in _EXPLAINER_CACHE:
                print("[ExplainTop2] Building TreeExplainer ...")
                t_exp = time.time()
                _EXPLAINER_CACHE[key] = shap.TreeExplainer(
                    self.model,
                    feature_perturbation="tree_path_dependent"
                )
                print(f"[ExplainTop2] TreeExplainer ready in {time.time() - t_exp:.2f}s")
            self._explainer = _EXPLAINER_CACHE[key]

    def _format_line(self, idx: int, feat_info: List[Tuple[str, str]]) -> str:
        # One-liner tailored for the leader's quick read with directions
        if len(feat_info) == 0:
            return f"[{idx}], no salient inputs detected."
        if len(feat_info) == 1:
            name, direction = feat_info[0]
            return f"[{idx}], {name} {direction}."
        (n1, d1), (n2, d2) = feat_info[0], feat_info[1]
        return f"[{idx}], {n1} {d1}, {n2} {d2}."

    def _topk_from_shap(self, df: pd.DataFrame) -> List[List[Tuple[str, str]]]:
        feature_names = list(df.columns)

        # Get predictions to select the class for multiclass SHAP values
        try:
            y_pred = self.model.predict(df)
        except Exception:
            y_pred = None

        # Compute SHAP values
        print(f"[ExplainTop2] Computing shap_values for df shape={df.shape} ...")
        t_sv = time.time()
        values = self._explainer.shap_values(df, check_additivity=False)  # type: ignore
        dt_sv = time.time() - t_sv

        n_samples, n_feat = df.shape
        # Build a per-sample SHAP matrix for the predicted class: [n_samples, n_features]
        shap_mat = np.zeros((n_samples, n_feat), dtype=float)

        if isinstance(values, list):
            print(f"[ExplainTop2] shap_values computed in {dt_sv:.2f}s (~{dt_sv/n_samples:.4f}s/sample) (multiclass list={len(values)})")
            classes = getattr(self.model, "classes_", None)
            for i in range(n_samples):
                if y_pred is not None:
                    c_pred = y_pred[i]
                    if classes is not None:
                        try:
                            c_idx = int(np.where(classes == c_pred)[0][0])
                        except Exception:
                            c_idx = int(c_pred)
                    else:
                        c_idx = int(c_pred)
                else:
                    c_idx = 0
                shap_mat[i, :] = values[c_idx][i, :]
        else:
            shape_info = getattr(values, "shape", "unknown")
            print(f"[ExplainTop2] shap_values computed in {dt_sv:.2f}s (~{dt_sv/n_samples:.4f}s/sample) (array shape={shape_info})")
            if values.ndim == 2:
                shap_mat = values
            elif values.ndim == 3:
                # Identify axes for (samples, features, classes), reorder to that
                dims = values.shape
                s_axis = int(np.argmin([abs(d - n_samples) for d in dims]))
                f_axis = int(np.argmin([abs(d - n_feat) for d in dims]))
                c_axis = [ax for ax in range(3) if ax not in (s_axis, f_axis)][0]
                values_re = np.moveaxis(values, [s_axis, f_axis, c_axis], [0, 1, 2])
                classes = getattr(self.model, "classes_", None)
                for i in range(n_samples):
                    if y_pred is not None:
                        c_pred = y_pred[i]
                        if classes is not None:
                            try:
                                c_idx = int(np.where(classes == c_pred)[0][0])
                            except Exception:
                                c_idx = int(c_pred)
                        else:
                            c_idx = int(c_pred)
                    else:
                        c_idx = 0
                    shap_mat[i, :] = values_re[i, :, c_idx]
            else:
                raise ValueError(f"Unexpected shap_values ndim: {values.ndim} shape={values.shape}")

        # Direction severity thresholds
        abs_mat = np.abs(shap_mat)
        large_batch = n_samples >= 20

        if large_batch:
            # Use global per-feature thresholds across the batch
            q99_global = np.quantile(abs_mat, 0.99, axis=0)

            def dir_label_global(val: float, idx_f: int) -> str:
                mag = abs(val)
                if mag >= q99_global[idx_f]:
                    return "absolutely abnormal"
                if val > 0:
                    return "higher than normal"
                if val < 0:
                    return "lower than normal"
                return "near normal"

            results: List[List[Tuple[str, str]]] = []
            for i in range(n_samples):
                sv = shap_mat[i, :]
                order = np.argsort(np.abs(sv))[-2:][::-1]
                info: List[Tuple[str, str]] = []
                for j in order:
                    j_int = int(j)
                    info.append((feature_names[j_int], dir_label_global(sv[j_int], j_int)))
                results.append(info)
        else:
            # Small batch (e.g. 1 sample in multiclass specialists). Use per-sample thresholds across features.
            results = []
            for i in range(n_samples):
                sv = shap_mat[i, :]
                abs_sv = np.abs(sv)
                # 95th and 99th percentiles across this sample's features
                p99 = np.quantile(abs_sv, 0.99)

                def dir_label_local(val: float) -> str:
                    mag = abs(val)
                    if mag >= p99:
                        return "absolutely abnormal"
                    if val > 0:
                        return "higher than normal"
                    if val < 0:
                        return "lower than normal"
                    return "near normal"

                order = np.argsort(abs_sv)[-2:][::-1]
                info: List[Tuple[str, str]] = []
                for j in order:
                    j_int = int(j)
                    info.append((feature_names[j_int], dir_label_local(sv[j_int])))
                results.append(info)

        return results

    def _run(self, samples: List[Dict[str, Any]]) -> str:
        df_input = pd.DataFrame(samples)
        if df_input.empty:
            return "[]"

        print(f"[ExplainTop2] _run: mode={self._mode} samples={len(samples)}, features={df_input.shape[1]}")

        # OFF mode: skip entirely
        if self._mode == "off":
            return ""

        # FAST mode: lightweight heuristic using feature importances and batch medians
        if self._mode == "fast":
            if not hasattr(self.model, "feature_importances_"):
                return "\n".join(self._format_line(i + 1, []) for i in range(len(samples)))
            importances = np.asarray(getattr(self.model, "feature_importances_"))
            # Guard if feature counts mismatch
            if importances.shape[0] != df_input.shape[1]:
                return "\n".join(self._format_line(i + 1, []) for i in range(len(samples)))
            order = np.argsort(importances)[-2:][::-1]
            top_idx = [int(x) for x in order]
            top_feats = [df_input.columns[j] for j in top_idx]
            medians = {f: float(df_input[f].median()) for f in top_feats}
            lines = []
            for i in range(len(df_input)):
                info: List[Tuple[str, str]] = []
                for f in top_feats:
                    val = float(df_input.iloc[i][f])
                    direction = "higher than normal" if val >= medians[f] else "lower than normal"
                    info.append((f, direction))
                lines.append(self._format_line(i + 1, info))
            return "\n".join(lines)

        # ON mode: compute SHAP with chunking to reduce peak memory
        t0 = time.time()
        chunk_env = os.getenv("QCYBER_SHAP_CHUNK", "64")
        try:
            chunk_size = max(1, int(chunk_env))
        except Exception:
            chunk_size = 64
        n = len(df_input)
        all_lines: List[str] = []
        for start in range(0, n, chunk_size):
            end = min(start + chunk_size, n)
            sub_df = df_input.iloc[start:end]
            top2_per_sample = self._topk_from_shap(sub_df)
            lines = [self._format_line(start + i + 1, feats) for i, feats in enumerate(top2_per_sample)]
            all_lines.extend(lines)
        print(f"[ExplainTop2] Top-2 selection done in {time.time() - t0:.2f}s (chunk_size={chunk_size})")
        return "\n".join(all_lines)
