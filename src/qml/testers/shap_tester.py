from pathlib import Path
import os
import time
import pandas as pd
import re
from typing import List

from qml.tools.shap_explain import ExplainTop2SHAP


ROOT_DIR = Path(__file__).resolve().parents[3]
DATA_PATH = ROOT_DIR / "data/dados_de_teste.csv"


def _parse_features_from_line(line: str) -> List[str]:
    # Matches: "[1], feat1 seems far from normal." or "[1], feat1, feat2 seem far from normal."
    m = re.match(r"^\[\d+\],\s+(.*)\s+seem?s\s+far\s+from\s+normal\.$", line.strip())
    if not m:
        return []
    feat_blob = m.group(1)
    # split by ", ", but only at top level (feature names are simple)
    parts = [p.strip() for p in feat_blob.split(",")]
    return [p for p in parts if p]


def run() -> None:
    t0 = time.time()
    print(f"[Tester] Loading dataset from {DATA_PATH} ...")
    df = pd.read_csv(DATA_PATH)
    print(f"[Tester] CSV loaded: shape={df.shape}")

    drop_cols = [c for c in ["Attack_label", "Attack_type"] if c in df.columns]
    if drop_cols:
        print(f"[Tester] Dropping label columns: {drop_cols}")
    features_df = df.drop(columns=drop_cols) if drop_cols else df.copy()
    print(f"[Tester] Features prepared: shape={features_df.shape}")

    if features_df.empty:
        raise ValueError("No feature columns available after dropping labels.")

    samples = features_df.to_dict(orient="records")
    n_samples, n_features = features_df.shape
    print(f"[Tester] Samples ready: n_samples={n_samples}, n_features={n_features}")

    # Resolve model path
    env_model_path = os.getenv("QML_RF_MODEL_MULT_PATH")
    if env_model_path:
        mp = Path(env_model_path)
        if not mp.exists():
            raise FileNotFoundError(f"Model path set in QML_RF_MODEL_MULT_PATH not found: {mp}")
        print(f"[Tester] Using model from env: {mp}")
        print("[Tester] Initializing ExplainTop2SHAP (multiclass)...")
        t_init = time.time()
        explainer_tool = ExplainTop2SHAP(model_path=str(mp), classification="multiclass")
        print(f"[Tester] Tool initialized in {time.time() - t_init:.2f}s")
    else:
        default_mp = ROOT_DIR / "IA/weights/traditional/random_forest_model_mult.joblib"
        if not default_mp.exists():
            raise FileNotFoundError(
                f"Random Forest model file not found at default path: {default_mp}. "
                f"Provide it there or set QML_RF_MODEL_MULT_PATH to the absolute path to your .joblib."
            )
        print(f"[Tester] Using default model path: {default_mp}")
        print("[Tester] Initializing ExplainTop2SHAP (multiclass)...")
        t_init = time.time()
        explainer_tool = ExplainTop2SHAP(classification="multiclass")
        print(f"[Tester] Tool initialized in {time.time() - t_init:.2f}s")

    print(f"[Tester] Running SHAP explain for {n_samples} samples x {n_features} features ...")
    t_run = time.time()
    batch_size = 32 if n_samples > 32 else n_samples
    all_lines_list = []
    for start in range(0, n_samples, batch_size):
        end = min(start + batch_size, n_samples)
        batch_idx = (start // batch_size) + 1
        print(f"[Tester]  - Processing batch {start+1}-{end} (batch #{batch_idx}) ...")
        tb = time.time()
        batch_lines = explainer_tool._run(samples[start:end])
        dtb = time.time() - tb
        n_batch = end - start
        print(f"[Tester]  - Batch {start+1}-{end} done in {dtb:.2f}s (~{dtb/n_batch:.4f}s/sample)")
        # Every 10 batches, print an example output line from the explainer
        if batch_idx % 10 == 0:
            preview_line = next((ln for ln in batch_lines.split("\n") if ln.strip()), "")
            if preview_line:
                print(f"[Tester]  - Example (batch #{batch_idx}): {preview_line}")
        all_lines_list.append(batch_lines)
    lines = "\n".join(all_lines_list)
    total_dt = time.time() - t_run
    print(f"[Tester] SHAP explain completed in {total_dt:.2f}s (~{total_dt/max(n_samples,1):.4f}s/sample)")

    # Print results
    print("[Tester] ===== SHAP Top-2 Lines =====")
    print(lines)
    print("[Tester] ===== End of Lines =====")

    # Unit-test style checks
    print("[Tester] Validating output invariants ...")
    out_lines = [ln for ln in lines.split("\n") if ln.strip()]
    assert len(out_lines) == len(features_df), "Output line count must match sample count."

    cols = set(features_df.columns.tolist())
    for i, ln in enumerate(out_lines, start=1):
        assert ln.startswith(f"[{i}], "), f"Line {i} is not correctly indexed/formatted: {ln}"
        feats = _parse_features_from_line(ln)
        assert 1 <= len(feats) <= 2, f"Line {i} must contain 1 or 2 features: {ln}"
        for f in feats:
            assert f in cols, f"Feature '{f}' in line {i} not found in input columns."

    print(f"[Tester] OK. Total time: {time.time() - t0:.2f}s")


if __name__ == "__main__":
    run()
