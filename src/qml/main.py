import json
import pandas as pd
from pathlib import Path
from typing import Any

from qml.services.detection import BatchDetectionService, build_detection_report


ROOT_DIR = Path(__file__).resolve().parents[2]
DATA_PATH = ROOT_DIR / "data" / "dados_de_teste.csv"
OUTPUT_DIR = ROOT_DIR / "src" / "qml" / "output"


def _load_default_samples(count: int = 5) -> tuple[list[dict[str, Any]], list[Any]]:
    if not DATA_PATH.exists():
        raise FileNotFoundError(f"Default sample dataset not found: {DATA_PATH}")

    df = pd.read_csv(DATA_PATH)
    if df.empty:
        raise ValueError(f"Default sample dataset is empty: {DATA_PATH}")

    label_columns = [column for column in ("Attack_type", "Attack_label") if column in df.columns]

    if "Attack_type" in df.columns:
        attacks = df[df["Attack_type"] != 7]
        if len(attacks) >= min(3, count):
            first = attacks.sample(min(3, count), random_state=42)
            rest = df.drop(first.index)
            remaining = count - len(first)
            second = rest.sample(remaining, replace=len(rest) < remaining, random_state=43)
            selected = pd.concat([first, second], ignore_index=True)
        else:
            selected = df.sample(count, replace=len(df) < count, random_state=42)

        labels = selected["Attack_type"].tolist()
        features = selected.drop(columns=label_columns)
    else:
        selected = df.sample(count, replace=len(df) < count, random_state=42)
        labels = []
        features = selected.drop(columns=label_columns)

    return features.to_dict(orient="records"), labels


def run() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    samples, labels = _load_default_samples()

    if labels:
        print(f"Loaded sample labels: {labels}")
    print(f"Running optimized batch detection with {len(samples)} samples...")

    result = BatchDetectionService().predict(samples)
    payload = result.to_dict()
    payload["report"] = build_detection_report(result)
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    run()
