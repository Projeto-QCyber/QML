import os
import pandas as pd
from typing import Any
from pathlib import Path

from qml.crew import CyberPredict


ROOT_DIR = Path(__file__).resolve().parents[2]
DEFAULT_DATASET = ROOT_DIR / "data" / "dados_de_teste_q.csv"


def _load_samples(samples_path: str | None = None, limit: int = 5) -> list[dict[str, Any]]:
    path = Path(samples_path).expanduser() if samples_path else DEFAULT_DATASET
    if not path.is_absolute():
        path = ROOT_DIR / path

    df = pd.read_csv(path)
    samples = df.head(limit).to_dict(orient="records")

    if len(samples) < limit:
        raise ValueError(f"Expected at least {limit} samples in {path}, found {len(samples)}.")

    return samples


def run() -> None:

    samples = _load_samples(samples_path=DEFAULT_DATASET)
    samples = samples[:5]

    print(f"QML_BINARY_TOOL_PROVIDER={os.getenv('QML_BINARY_TOOL_PROVIDER', 'both')}")
    result = CyberPredict().crew().kickoff(inputs={"samples": samples})
    print(result)


def run_quantum() -> None:
    os.environ["QML_BINARY_TOOL_PROVIDER"] = "quantum"
    run()


def run_rf() -> None:
    os.environ["QML_BINARY_TOOL_PROVIDER"] = "rf"
    run()


if __name__ == "__main__":
    run()
