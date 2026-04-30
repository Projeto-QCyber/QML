from __future__ import annotations

import json
import re
from collections import Counter
from dataclasses import dataclass, field
from typing import Any

from qml.tools.model import RFModel


ATTACK_LABELS: dict[int, str] = {
    0: "Backdoor",
    1: "DDoS_HTTP",
    2: "DDoS_ICMP",
    3: "DDoS_TCP",
    4: "DDoS_UDP",
    5: "Fingerprinting",
    6: "MITM",
    7: "Password",
    8: "Port_Scanning",
    9: "Ransomware",
    10: "SQL_injection",
    11: "Uploading",
    12: "Vulnerability_scanner",
    13: "XSS",
    99: "Normal",
}


@dataclass(slots=True)
class ClassifiedSample:
    batch_index: int
    attack_type_id: int
    attack_type_label: str
    probabilities: list[float] | None = None


@dataclass(slots=True)
class BatchDetectionResult:
    binary_predictions: list[int]
    attack_indices: list[int]
    classified_attacks: list[ClassifiedSample] = field(default_factory=list)

    @property
    def window_has_attack(self) -> bool:
        return bool(self.attack_indices)

    @property
    def primary_attack_type_id(self) -> int:
        if not self.classified_attacks:
            return 99

        counts = Counter(item.attack_type_id for item in self.classified_attacks)
        most_common_count = counts.most_common(1)[0][1]
        tied = {attack_id for attack_id, count in counts.items() if count == most_common_count}

        for item in self.classified_attacks:
            if item.attack_type_id in tied:
                return item.attack_type_id
        return self.classified_attacks[0].attack_type_id

    @property
    def primary_attack_type_label(self) -> str:
        return ATTACK_LABELS.get(self.primary_attack_type_id, "Normal")

    def to_dict(self) -> dict[str, Any]:
        return {
            "binary_predictions": self.binary_predictions,
            "window_has_attack": self.window_has_attack,
            "attack_indices": self.attack_indices,
            "classified_attacks": [
                {
                    "batch_index": item.batch_index,
                    "attack_type_id": item.attack_type_id,
                    "attack_type_label": item.attack_type_label,
                    "probabilities": item.probabilities,
                }
                for item in self.classified_attacks
            ],
            "primary_attack_type_id": self.primary_attack_type_id,
            "primary_attack_type_label": self.primary_attack_type_label,
        }


class BatchDetectionService:
    """Deterministic model pipeline used before optional LLM interpretation."""

    def __init__(self, use_binary_crew: bool = False) -> None:
        self.use_binary_crew = use_binary_crew
        self.binary_model = None if use_binary_crew else RFModel(classification="binary")
        self._multiclass_model: RFModel | None = None

    @property
    def multiclass_model(self) -> RFModel:
        if self._multiclass_model is None:
            self._multiclass_model = RFModel(classification="multiclass")
        return self._multiclass_model

    def _predict_binary(self, samples: list[dict[str, Any]]) -> list[int]:
        if self.use_binary_crew:
            return self._predict_binary_with_crew(samples)
        if self.binary_model is None:
            raise RuntimeError("Modelo binário direto não foi inicializado.")
        return self.binary_model.predict(samples)

    def _predict_binary_with_crew(self, samples: list[dict[str, Any]]) -> list[int]:
        from qml.crew import CyberPredict

        result = CyberPredict().crew().kickoff(inputs={"samples": samples})
        raw_output = getattr(result, "raw", str(result))
        predictions = _extract_binary_predictions(raw_output)
        if not predictions:
            raise RuntimeError(f"A crew binária não retornou predições válidas. Saída: {raw_output}")
        return predictions

    def predict(self, samples: list[dict[str, Any]]) -> BatchDetectionResult:
        binary_predictions = self._predict_binary(samples)
        if len(binary_predictions) != len(samples):
            raise RuntimeError(
                "O modelo binário retornou uma quantidade de predições diferente "
                "da quantidade de amostras recebidas."
            )
        attack_indices = [idx for idx, pred in enumerate(binary_predictions) if int(pred) == 1]

        if not attack_indices:
            return BatchDetectionResult(
                binary_predictions=binary_predictions,
                attack_indices=[],
            )

        suspicious_samples = [samples[idx] for idx in attack_indices]
        multiclass_predictions = self.multiclass_model.predict(suspicious_samples)
        if len(multiclass_predictions) != len(suspicious_samples):
            raise RuntimeError(
                "O modelo multiclasse retornou uma quantidade de predições diferente "
                "da quantidade de amostras suspeitas."
            )
        probabilities = self.multiclass_model.predict_proba(suspicious_samples)

        classified_attacks = []
        for local_idx, attack_type_id in enumerate(multiclass_predictions):
            classified_attacks.append(
                ClassifiedSample(
                    batch_index=attack_indices[local_idx],
                    attack_type_id=int(attack_type_id),
                    attack_type_label=ATTACK_LABELS.get(int(attack_type_id), f"Classe_{attack_type_id}"),
                    probabilities=probabilities[local_idx] if probabilities else None,
                )
            )

        return BatchDetectionResult(
            binary_predictions=binary_predictions,
            attack_indices=attack_indices,
            classified_attacks=classified_attacks,
        )


def build_detection_report(result: BatchDetectionResult, shap_explanation: str = "") -> str:
    if not result.window_has_attack:
        return "A janela analisada não apresentou amostras classificadas como ataque pelo modelo binário."

    grouped = Counter(item.attack_type_label for item in result.classified_attacks)
    grouped_text = ", ".join(f"{label}: {count}" for label, count in grouped.most_common())
    indices_text = ", ".join(str(idx) for idx in result.attack_indices)
    report = (
        f"A janela contém {len(result.attack_indices)} amostra(s) suspeita(s), "
        f"nos índices [{indices_text}]. Classificação multiclasse consolidada: {grouped_text}. "
        f"Tipo primário selecionado: {result.primary_attack_type_label}."
    )

    if shap_explanation.strip():
        report += f"\n\nPrincipais evidências SHAP:\n{shap_explanation.strip()}"

    return report


def _extract_binary_predictions(text: str) -> list[int]:
    if not text:
        return []

    candidate = _extract_json_object(text)
    if isinstance(candidate, dict):
        raw_predictions = candidate.get("predictions") or candidate.get("votes")
        if isinstance(raw_predictions, list):
            return [int(value) for value in raw_predictions]
        if isinstance(raw_predictions, str):
            parsed = _extract_first_int_list(raw_predictions)
            if parsed:
                return parsed

    return _extract_first_int_list(text)


def _extract_json_object(text: str) -> dict[str, Any] | None:
    code_block = re.search(r"```(?:json)?\s*(.*?)```", text, re.DOTALL | re.IGNORECASE)
    if code_block:
        try:
            parsed = json.loads(code_block.group(1).strip())
            if isinstance(parsed, dict):
                return parsed
        except json.JSONDecodeError:
            pass

    try:
        parsed = json.loads(text)
        if isinstance(parsed, dict):
            return parsed
    except json.JSONDecodeError:
        pass

    for match in re.findall(r"\{.*?\}", text, flags=re.DOTALL):
        try:
            parsed = json.loads(match)
            if isinstance(parsed, dict):
                return parsed
        except json.JSONDecodeError:
            continue

    return None


def _extract_first_int_list(text: str) -> list[int]:
    match = re.search(r"\[(?:\s*[-+]?\d+\s*,?\s*)+\]", text)
    if not match:
        return []
    try:
        parsed = json.loads(match.group(0))
        if isinstance(parsed, list):
            return [int(value) for value in parsed]
    except json.JSONDecodeError:
        pass
    return [int(value) for value in re.findall(r"[-+]?\d+", match.group(0))]
