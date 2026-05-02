import json
import os
import pandas as pd
from pathlib import Path
from typing import Any

os.environ.setdefault("CREWAI_DISABLE_TELEMETRY", "true")
os.environ.setdefault("OTEL_SDK_DISABLED", "true")
os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

from qml.services.detection import BatchDetectionService, build_detection_report
from qml.utils.runtime import default_crew_enabled, env_flag


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
        attacks = df[df["Attack_type"] != 0]
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


def _save_detection_payload(payload: dict[str, Any]) -> Path:
    output_path = OUTPUT_DIR / "detection_result.json"
    with output_path.open("w", encoding="utf-8") as file:
        json.dump(payload, file, ensure_ascii=False, indent=2)
        file.write("\n")
    return output_path


def run() -> None:
    os.environ.setdefault("QCYBER_CREW_VERBOSE", "1")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    samples, labels = _load_default_samples()
    crew_default = default_crew_enabled()
    use_binary_crew = env_flag("QCYBER_USE_BINARY_CREW", default=crew_default)
    use_multiclass_crew = env_flag("QCYBER_USE_MULTICLASS_CREW", default=crew_default)
    use_incident_response_crew = env_flag(
        "QCYBER_USE_INCIDENT_RESPONSE_CREW",
        default=crew_default,
    )
    use_remediation_crew = env_flag("QCYBER_USE_REMEDIATION_CREW", default=crew_default)

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
    output_path = _save_detection_payload(payload)
    print(f"Saved detection output to: {output_path}")
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    run()
