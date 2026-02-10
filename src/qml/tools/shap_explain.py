import os
from pathlib import Path
from typing import Optional, List, Dict, Any, Literal

import joblib
import numpy as np
import pandas as pd
import shap

# Ensure headless rendering for Matplotlib export
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


ROOT_DIR = Path(__file__).resolve().parents[3]


def load_model(model_path: str | Path) -> object:
    """Load a joblib model from disk."""
    return joblib.load(Path(model_path))


def align_features_to_model(model, X: pd.DataFrame) -> pd.DataFrame:
    """Reorder/select columns to match the model's expected features if available.

    Raises if any expected feature is missing.
    """
    if hasattr(model, "feature_names_in_"):
        feats = list(getattr(model, "feature_names_in_"))
        missing = [f for f in feats if f not in X.columns]
        if missing:
            raise ValueError(f"Missing features required by the model: {missing}")
        return X[feats]
    return X


def small_background(X: pd.DataFrame, background_size: int = 100, random_state: int = 0) -> pd.DataFrame:
    """Return a small background subset of the given DataFrame."""
    if len(X) <= background_size:
        return X
    return X.sample(background_size, random_state=random_state)


def build_explainer(model, background: pd.DataFrame) -> shap.Explainer:
    """Create a SHAP Explainer with a small background set."""
    return shap.Explainer(model, background)


def save_waterfalls(
    model,
    X: pd.DataFrame,
    out_dir: Path,
    n_samples: int = 10,
    background_size: int = 100,
    max_display: int = 10,
    random_state: int = 0,
    label_column: Optional[str] = "Attack_label",
) -> list[Path]:
    """Save up to n_samples SHAP waterfall plots using the given model and features.

    - Uses a small background subset to keep things light.
    - For multiclass models, picks the predicted class per sample for plotting.
    Returns list of written SVG paths.
    """
    out_dir.mkdir(parents=True, exist_ok=True)

    # Keep original data to recover labels for titles/filenames
    X_orig = X.copy()
    # Align features to what the model expects
    X = align_features_to_model(model, X)

    # Only take the first n_samples to keep it light and deterministic
    X_sel = X.iloc[: max(1, min(n_samples, len(X)))]
    # Corresponding original rows (for labels)
    orig_for_sel = None
    if label_column and isinstance(X_orig, pd.DataFrame) and label_column in X_orig.columns:
        try:
            orig_for_sel = X_orig.loc[X_sel.index]
        except Exception:
            orig_for_sel = None

    # Build explainer with a small background
    background = small_background(X, background_size=background_size, random_state=random_state)
    explainer = build_explainer(model, background)

    # Compute SHAP explanations
    exp = explainer(X_sel)

    # If available, get predicted class per sample
    pred_proba: Optional[np.ndarray]
    try:
        pred_proba = model.predict_proba(X_sel)  # type: ignore[attr-defined]
        if isinstance(pred_proba, list):
            # Guard against multi-output list form; use the first output
            pred_proba = pred_proba[0]
    except Exception:
        pred_proba = None

    written: list[Path] = []
    classes = getattr(model, "classes_", None)

    # Also try to get class predictions for titles
    try:
        preds = model.predict(X_sel)  # type: ignore[attr-defined]
    except Exception:
        preds = None

    for i in range(len(X_sel)):
        row_exp = exp[i]

        # If multi-output, select the predicted class per-sample when possible
        if hasattr(row_exp, "values") and getattr(row_exp.values, "ndim", 1) == 2 and row_exp.values.shape[1] > 1:
            if pred_proba is not None and pred_proba.ndim == 2 and pred_proba.shape[0] == len(X_sel):
                j = int(np.argmax(pred_proba[i]))
            else:
                j = 0
            row_exp = row_exp[:, j]

        # Plot and save
        shap.plots.waterfall(row_exp, max_display=max_display, show=False)
        # Build an informative title
        pred_lbl = None
        pred_prob = None
        if preds is not None:
            try:
                pred_lbl = preds[i]
            except Exception:
                pred_lbl = None
        if pred_proba is not None:
            try:
                pred_prob = float(np.max(pred_proba[i]))
            except Exception:
                pred_prob = None
        true_lbl = None
        if label_column and orig_for_sel is not None and label_column in orig_for_sel.columns:
            try:
                true_lbl = orig_for_sel.iloc[i][label_column]
            except Exception:
                true_lbl = None

        # Compute base value and model output for this slice
        try:
            base_val = float(np.squeeze(row_exp.base_values))
        except Exception:
            base_val = None
        try:
            shap_sum = float(np.sum(np.squeeze(row_exp.values)))
            fx_val = (base_val + shap_sum) if base_val is not None else None
        except Exception:
            fx_val = None

        sample_id = X_sel.index[i] if hasattr(X_sel, "index") else (i + 1)

        ax = plt.gca()
        ax.set_xlabel("SHAP value (impact on model output)")
        ax.set_ylabel("Features")
        # Descriptive filename
        def _s(x):
            try:
                return str(x).replace("/", "-").replace(" ", "_")
            except Exception:
                return str(x)

        name_bits = [f"shap_waterfall_{i+1:02d}"]
        if pred_lbl is not None:
            name_bits.append(f"pred-{_s(pred_lbl)}")
        if true_lbl is not None:
            name_bits.append(f"label-{_s(true_lbl)}")
        fname = out_dir / ("_".join(name_bits) + ".svg")
        plt.tight_layout()
        plt.savefig(fname, dpi=300, bbox_inches="tight", format="svg")
        plt.close()
        written.append(fname)

    return written


def _select_predicted_class_shap(exp: shap.Explanation, model, X: pd.DataFrame) -> np.ndarray:
    """Return a (n_samples, n_features) SHAP matrix by selecting per-sample
    the SHAP values of the predicted class when multi-output.
    """
    vals = exp.values
    # Already 2D (n, d)
    if getattr(vals, "ndim", 2) == 2:
        return vals

    # Expect (n, d, c). Choose argmax predicted class for each sample.
    try:
        proba = model.predict_proba(X)  # type: ignore[attr-defined]
        if isinstance(proba, list):
            proba = proba[0]
    except Exception:
        proba = None

    n, d, *_ = vals.shape
    out = np.zeros((n, d), dtype=float)
    for i in range(n):
        j = int(np.argmax(proba[i])) if proba is not None else 0
        out[i, :] = vals[i, :, j]
    return out


def save_violin_by_label(
    model,
    df: pd.DataFrame,
    label_column: str,
    out_dir: Path,
    sample_per_label: int = 50,
    background_size: int = 200,
    layered: bool = False,
    max_display: int = 20,
    random_state: int = 0,
) -> list[Path]:
    """Create balanced subsets (up to sample_per_label per label) and save a SHAP
    violin summary plot per label value.
    """
    out_dir.mkdir(parents=True, exist_ok=True)

    if label_column not in df.columns:
        raise ValueError(f"Label column '{label_column}' not found in DataFrame")

    # Balanced sampling per label
    written: list[Path] = []
    for label_val, group in df.groupby(label_column):
        sub = group.sample(min(sample_per_label, len(group)), random_state=random_state)

        # Features must include whatever the model expects (may include label columns)
        X = align_features_to_model(model, sub)

        # Background and explainer
        bg = small_background(X, background_size=background_size, random_state=random_state)
        explainer = build_explainer(model, bg)

        # Explanations for this label subset
        exp = explainer(X)

        # Build a 2D shap matrix selecting per-sample predicted class if needed
        shap_mat = _select_predicted_class_shap(exp, model, X)

        # Plot
        plt.figure()
        plot_type = "layered_violin" if layered else "violin"
        shap.plots.violin(
            shap_mat,
            features=X,
            feature_names=list(X.columns),
            plot_type=plot_type,
            max_display=max_display,
            color_bar_label="Coloration represents magnitude",
            show=False,
        )
        # Title and axes
        plt.xlabel("Shap Value (impact on model output) of 100 samples")
        plt.ylabel("Features")
        fname = out_dir / f"shap_violin_label_{label_val}.svg"
        plt.tight_layout()
        plt.savefig(fname, dpi=300, bbox_inches="tight", format="svg")
        plt.close()
        written.append(fname)

    return written


class ExplainTop2SHAP:
    """
    Lightweight helper that loads the random-forest model (binary or multiclass),
    computes SHAP values for the provided samples and returns a human-readable
    summary listing the most influential features.
    """

    def __init__(
        self,
        classification: Literal["multiclass", "binary"] = "binary",
        model_path: Optional[str | Path] = None,
        background_csv: Optional[str | Path] = None,
        top_k: int = 2,
        background_size: int = 200,
    ) -> None:
        self.classification = classification
        self.top_k = max(1, top_k)
        self.background_size = max(1, background_size)
        self.model_path = self._resolve_model_path(model_path)
        self.model = load_model(self.model_path)
        self._background_csv = Path(background_csv) if background_csv else None

    # ------------------------------------------------------------------ helpers
    def _resolve_model_path(self, explicit: Optional[str | Path]) -> Path:
        def _candidate(path_like: Optional[str | Path]) -> Optional[Path]:
            if not path_like:
                return None
            p = Path(path_like)
            return p if p.exists() else None

        if explicit:
            explicit_path = Path(explicit)
            if not explicit_path.exists():
                raise FileNotFoundError(f"Specified SHAP model not found: {explicit_path}")
            return explicit_path

        env_key = "QML_RF_MODEL_MULT_PATH" if self.classification == "multiclass" else "QML_RF_MODEL_BIN_PATH"
        env_path = _candidate(os.getenv(env_key))
        if env_path:
            return env_path

        default_path = (
            ROOT_DIR / "IA/weights/traditional/random_forest_model_mult_q.joblib"
            if self.classification == "multiclass"
            else ROOT_DIR / "IA/weights/traditional/random_forest_model_bin_q.joblib"
        )
        if default_path.exists():
            return default_path

        raise FileNotFoundError(
            f"Could not locate SHAP model for classification='{self.classification}'. "
            "Set QML_RF_MODEL_MULT_PATH / QML_RF_MODEL_BIN_PATH or provide model_path."
        )

    def _prepare_dataframe(self, samples: List[Dict[str, Any]]) -> pd.DataFrame:
        if not samples:
            raise ValueError("ExplainTop2SHAP requires at least one sample.")
        df = pd.DataFrame(samples)
        if df.empty:
            raise ValueError("ExplainTop2SHAP received empty samples.")
        df = df.apply(pd.to_numeric, errors="coerce").fillna(0.0)
        return align_features_to_model(self.model, df)

    def _load_background_source(self) -> Optional[pd.DataFrame]:
        candidates: List[Path] = []
        if self._background_csv:
            candidates.append(self._background_csv)
        env_bg = os.getenv("QML_SHAP_BACKGROUND_CSV")
        if env_bg:
            candidates.append(Path(env_bg))
        default_bg = ROOT_DIR / "data/dados_de_teste_q.csv"
        if default_bg.exists():
            candidates.append(default_bg)

        for path in candidates:
            if not path.exists():
                continue
            try:
                df = pd.read_csv(path)
                if df.empty:
                    continue
                df = df.apply(pd.to_numeric, errors="coerce").fillna(0.0)
                return align_features_to_model(self.model, df)
            except Exception as exc:
                print(f"[ExplainTop2SHAP] Failed to use background '{path}': {exc}")
        return None

    def _prepare_background(self, df: pd.DataFrame) -> pd.DataFrame:
        bg = self._load_background_source()
        if bg is not None and not bg.empty:
            return small_background(bg, background_size=min(len(bg), self.background_size))
        return small_background(df, background_size=min(len(df), self.background_size))

    def _compute_explanation(self, df: pd.DataFrame) -> shap.Explanation:
        try:
            explainer = shap.TreeExplainer(self.model)
            return explainer(df)
        except Exception:
            background = self._prepare_background(df)
            explainer = shap.Explainer(self.model, background)
            return explainer(df)

    def _summarize(self, exp: shap.Explanation, df: pd.DataFrame) -> str:
        shap_matrix = _select_predicted_class_shap(exp, self.model, df)
        row = shap_matrix[0]
        feature_names = list(df.columns)

        order = np.argsort(np.abs(row))[::-1][: min(self.top_k, len(feature_names))]
        try:
            prediction = self.model.predict(df)[0]
        except Exception:
            prediction = None

        header = (
            f"Top {len(order)} features influencing the {self.classification} prediction"
            + (f" (predicted class: {prediction})" if prediction is not None else "")
            + ":"
        )

        lines = [header]
        for idx in order:
            value = float(row[idx])
            feature = feature_names[idx]
            trend = "increases" if value >= 0 else "decreases"
            lines.append(
                f"- {feature}: SHAP={value:.4f} ({trend} the model score for this sample)"
            )

        return "\n".join(lines)

    # ---------------------------------------------------------------- public API
    def _run(self, samples: List[Dict[str, Any]]) -> str:
        """
        Mirrors the crew tool interface so the API layer can re-use it directly.
        """
        df = self._prepare_dataframe(samples)
        explanation = self._compute_explanation(df)
        return self._summarize(explanation, df)
