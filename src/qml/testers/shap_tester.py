from pathlib import Path
import os
import time
import pandas as pd
from typing import Optional

from qml.tools.shap_explain import load_model, save_waterfalls, save_violin_by_label


ROOT_DIR = Path(__file__).resolve().parents[3]
DATA_PATH = ROOT_DIR / "data" / "test" / "dados_de_teste.csv"


def _resolve_model_path() -> Path:
    """Choose an existing model path.

    Preference order:
    1) Env override QML_SHAP_MODEL in {"mult", "bin"}
    2) Env path QML_RF_MODEL_MULT_PATH / QML_RF_MODEL_BIN_PATH
    3) Default files under IA/weights/traditional/
    """
    prefer = os.getenv("QML_SHAP_MODEL", "mult").strip().lower()
    env_mult = os.getenv("QML_RF_MODEL_MULT_PATH")
    env_bin = os.getenv("QML_RF_MODEL_BIN_PATH")
    default_mult = ROOT_DIR / "IA/weights/traditional/random_forest_model_mult_q.joblib"
    default_bin = ROOT_DIR / "IA/weights/traditional/random_forest_model_bin_q.joblib"

    def ok(p: Optional[str | Path]) -> Optional[Path]:
        if not p:
            return None
        pp = Path(p)
        return pp if pp.exists() else None

    cand_mult = ok(env_mult) or (default_mult if default_mult.exists() else None)
    cand_bin = ok(env_bin) or (default_bin if default_bin.exists() else None)

    if prefer.startswith("mult") and cand_mult is not None:
        return cand_mult
    if prefer.startswith("bin") and cand_bin is not None:
        return cand_bin

    # Fallback: first available
    if cand_mult is not None:
        return cand_mult
    if cand_bin is not None:
        return cand_bin
    raise FileNotFoundError(
        "No model file found. Expected one of: "
        f"{default_mult} or {default_bin}, or env QML_RF_MODEL_MULT_PATH/QML_RF_MODEL_BIN_PATH."
    )


def run() -> None:
    t0 = time.time()
    print(f"[SHAP Tester] Loading CSV: {DATA_PATH}")
    df = pd.read_csv(DATA_PATH)
    # Do not drop columns here; the model may have been trained with them present.
    # Column alignment is handled inside qml.tools.shap_explain.align_features_to_model.
    X = df
    if X.empty:
        raise ValueError("No feature columns available after dropping labels.")

    model_path = _resolve_model_path()
    print(f"[SHAP Tester] Using model: {model_path}")
    model = load_model(model_path)

    out_dir = ROOT_DIR / "src/qml/output/shap"
    written = save_waterfalls(
        model=model,
        X=X,
        out_dir=out_dir,
        n_samples=10,
        background_size=100,
        max_display=10,
        random_state=0,
    )

    print(f"[SHAP Tester] Wrote {len(written)} SVGs to {out_dir}")
    for p in written:
        print(f" - {p}")

    # Violin plots by Attack_label (balanced sample up to 50 per label)
    violin_dir = out_dir / "violin"
    violin_written = save_violin_by_label(
        model=model,
        df=X,  # include all columns; aligner will pick what model expects
        label_column="Attack_label",
        out_dir=violin_dir,
        sample_per_label=50,
        background_size=200,
        layered=False,
        max_display=20,
        random_state=0,
    )
    print(f"[SHAP Tester] Wrote {len(violin_written)} violin SVGs to {violin_dir}")
    for p in violin_written:
        print(f" - {p}")
    print(f"[SHAP Tester] Done in {time.time() - t0:.2f}s")


if __name__ == "__main__":
    run()
