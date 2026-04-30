import json
import os
import pandas as pd
from pathlib import Path
from typing import Any

os.environ.setdefault("CREWAI_DISABLE_TELEMETRY", "true")
os.environ.setdefault("OTEL_SDK_DISABLED", "true")
os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

from qml.services.detection import BatchDetectionService, build_detection_report
from qml.utils.runtime import env_flag


ROOT_DIR = Path(__file__).resolve().parents[2]
DATA_PATH = ROOT_DIR / "data" / "test" / "dados_de_teste.csv"
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
    os.environ.setdefault("QCYBER_CREW_VERBOSE", "1")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    samples, labels = _load_default_samples()
    use_binary_crew = env_flag("QCYBER_USE_BINARY_CREW", default=True)
    use_multiclass_crew = env_flag("QCYBER_USE_MULTICLASS_CREW", default=True)
    use_incident_response_crew = env_flag("QCYBER_USE_INCIDENT_RESPONSE_CREW", default=True)
    use_remediation_crew = env_flag("QCYBER_USE_REMEDIATION_CREW", default=True)

    if labels:
        print(f"Loaded sample labels: {labels}")
    print(f"Running optimized two-stage CrewAI detection with {len(samples)} samples...")
    print(
        "Workflow config: "
        f"binary_crew={use_binary_crew}, "
        f"multiclass_crew={use_multiclass_crew}, "
        f"incident_response_crew={use_incident_response_crew}, "
        f"remediation_crew={use_remediation_crew}"
    )

    result = BatchDetectionService(
        use_binary_crew=use_binary_crew,
        use_multiclass_crew=use_multiclass_crew,
        use_incident_response_crew=use_incident_response_crew,
        use_remediation_crew=use_remediation_crew,
    ).predict(samples)
    payload = result.to_dict()
    payload["report"] = build_detection_report(result)
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    run()
