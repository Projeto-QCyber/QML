#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any, Dict, List

import pandas as pd

ROOT_DIR = Path(__file__).resolve().parents[2]
SRC_DIR = ROOT_DIR / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from qml.tools.quantum_model import QuantumModel


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Run real quantum_model inference using one row from CSV.",
    )
    p.add_argument("--csv", default="data/dados_de_teste_q.csv", help="CSV source path.")
    p.add_argument("--row", type=int, default=0, help="Row index to inject into the tool.")
    p.add_argument(
        "--model-path",
        default=str(ROOT_DIR / "IA/weights/quantum/quantum_angle_embedding_y_ring_rot_cnot_L4_s42_8806630d6b.pt"),
        help="Path to .pt quantum model.",
    )
    p.add_argument("--device", default=None, help="Optional PennyLane device override.")
    return p.parse_args()


def _drop_label_cols(sample: Dict[str, Any]) -> Dict[str, Any]:
    out = dict(sample)
    for k in list(out.keys()):
        if str(k).strip().lower() in {"attack_label", "attack_type", "label"}:
            out.pop(k, None)
    return out


def main() -> int:
    args = parse_args()
    df = pd.read_csv(args.csv)
    if df.empty:
        raise SystemExit("CSV is empty.")
    if args.row < 0 or args.row >= len(df):
        raise SystemExit(f"Row index out of range: {args.row} (rows={len(df)}).")

    raw_sample = df.iloc[[args.row]].to_dict(orient="records")[0]
    sample = _drop_label_cols(raw_sample)

    qm = QuantumModel(model_path=args.model_path, device=args.device)
    preds: List[int] = qm._run([sample])

    # Expose whether preprocessing stages exist in loaded model.
    clf = qm._classifier
    print("sample_row:", args.row)
    print("rows_in:", 1)
    print("preds_out:", preds)
    print("ok_format:", isinstance(preds, list) and all(isinstance(x, int) for x in preds))
    print("model_path_used:", qm.model_path)
    print("has_quantile:", bool(getattr(clf, "quantile", None) is not None))
    print("has_pls:", bool(getattr(clf, "pls", None) is not None))
    print("has_pca:", bool(getattr(clf, "pca", None) is not None))
    print("has_scaler:", bool(getattr(clf, "scaler", None) is not None))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
