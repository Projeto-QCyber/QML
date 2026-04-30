from __future__ import annotations

import json
import os
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
    binary_probabilities: list[float] | None = None
    binary_confidence: float | None = None
    model_attack_type_id: int | None = None
    workflow_attack_type_id: int | None = None
    probabilities: list[float] | None = None
    confidence: float | None = None
    decision_status: str = "accepted"
    rejection_reason: str = ""
    report: str | None = None


@dataclass(slots=True)
class BinarySampleDecision:
    batch_index: int
    prediction: int
    probabilities: list[float] | None = None
    confidence: float | None = None
    decision_status: str = "accepted"
    reason: str = ""


@dataclass(slots=True)
class BatchDetectionResult:
    binary_predictions: list[int]
    attack_indices: list[int]
    binary_decisions: list[BinarySampleDecision] = field(default_factory=list)
    review_indices: list[int] = field(default_factory=list)
    classified_attacks: list[ClassifiedSample] = field(default_factory=list)
    incident_response_plan: str = ""
    incident_response_status: str = "not_applicable"
    remediation_suggestion: dict[str, Any] = field(default_factory=dict)
    remediation_suggestion_status: str = "not_applicable"
    workflow_trace: list[dict[str, Any]] = field(default_factory=list)

    @property
    def window_has_attack(self) -> bool:
        return bool(self.attack_indices)

    @property
    def primary_attack_type_id(self) -> int:
        accepted_attacks = [
            item for item in self.classified_attacks if item.decision_status == "accepted"
        ]
        if not accepted_attacks:
            return 99

        counts = Counter(item.attack_type_id for item in accepted_attacks)
        most_common_count = counts.most_common(1)[0][1]
        tied = {attack_id for attack_id, count in counts.items() if count == most_common_count}

        for item in accepted_attacks:
            if item.attack_type_id in tied:
                return item.attack_type_id
        return accepted_attacks[0].attack_type_id

    @property
    def primary_attack_type_label(self) -> str:
        return ATTACK_LABELS.get(self.primary_attack_type_id, "Normal")

    def to_dict(self) -> dict[str, Any]:
        return {
            "binary_predictions": self.binary_predictions,
            "binary_decisions": [
                {
                    "batch_index": item.batch_index,
                    "prediction": item.prediction,
                    "probabilities": item.probabilities,
                    "confidence": item.confidence,
                    "decision_status": item.decision_status,
                    "reason": item.reason,
                }
                for item in self.binary_decisions
            ],
            "window_has_attack": self.window_has_attack,
            "attack_indices": self.attack_indices,
            "review_indices": self.review_indices,
            "classified_attacks": [
                {
                    "batch_index": item.batch_index,
                    "attack_type_id": item.attack_type_id,
                    "attack_type_label": item.attack_type_label,
                    "binary_probabilities": item.binary_probabilities,
                    "binary_confidence": item.binary_confidence,
                    "model_attack_type_id": item.model_attack_type_id,
                    "workflow_attack_type_id": item.workflow_attack_type_id,
                    "probabilities": item.probabilities,
                    "confidence": item.confidence,
                    "decision_status": item.decision_status,
                    "rejection_reason": item.rejection_reason,
                    "report": item.report,
                }
                for item in self.classified_attacks
            ],
            "primary_attack_type_id": self.primary_attack_type_id,
            "primary_attack_type_label": self.primary_attack_type_label,
            "incident_response_status": self.incident_response_status,
            "incident_response_plan": self.incident_response_plan,
            "remediation_suggestion_status": self.remediation_suggestion_status,
            "remediation_suggestion": self.remediation_suggestion,
            "workflow_trace": self.workflow_trace,
        }


class BatchDetectionService:
    """Deterministic model pipeline used before optional LLM interpretation."""

    def __init__(
        self,
        use_binary_crew: bool = False,
        use_multiclass_crew: bool = False,
        use_incident_response_crew: bool = False,
        use_remediation_crew: bool = False,
        binary_min_confidence: float | None = None,
        multiclass_min_confidence: float | None = None,
    ) -> None:
        self.use_binary_crew = use_binary_crew
        self.use_multiclass_crew = use_multiclass_crew
        self.use_incident_response_crew = use_incident_response_crew
        self.use_remediation_crew = use_remediation_crew
        self.binary_min_confidence = (
            binary_min_confidence
            if binary_min_confidence is not None
            else _read_float_env("QCYBER_BINARY_MIN_CONFIDENCE", 0.60)
        )
        self.multiclass_min_confidence = (
            multiclass_min_confidence
            if multiclass_min_confidence is not None
            else _read_float_env("QCYBER_MULTICLASS_MIN_CONFIDENCE", 0.55)
        )
        self.binary_model = RFModel(classification="binary")
        self._multiclass_model: RFModel | None = None

    @property
    def multiclass_model(self) -> RFModel:
        if self._multiclass_model is None:
            self._multiclass_model = RFModel(classification="multiclass")
        return self._multiclass_model

    def _predict_binary(
        self,
        samples: list[dict[str, Any]],
        probabilities: list[list[float]] | None,
    ) -> list[int]:
        if self.use_binary_crew:
            return self._predict_binary_with_crew(samples, probabilities)
        return self.binary_model.predict(samples)

    def _predict_binary_with_crew(
        self,
        samples: list[dict[str, Any]],
        probabilities: list[list[float]] | None,
    ) -> list[int]:
        from qml.crew import CyberPredict

        probability_context = _build_binary_probability_context(
            probabilities=probabilities,
            min_confidence=self.binary_min_confidence,
        )
        result = CyberPredict().crew().kickoff(
            inputs={
                "samples": samples,
                "binary_probability_context": json.dumps(probability_context, ensure_ascii=False),
            }
        )
        raw_output = getattr(result, "raw", str(result))
        predictions = _extract_binary_predictions(raw_output)
        if not predictions:
            raise RuntimeError(f"A crew binária não retornou predições válidas. Saída: {raw_output}")
        return predictions

    def _build_binary_decisions(
        self,
        predictions: list[int],
        probabilities: list[list[float]] | None,
    ) -> list[BinarySampleDecision]:
        decisions: list[BinarySampleDecision] = []
        for idx, prediction in enumerate(predictions):
            sample_probabilities = probabilities[idx] if probabilities and idx < len(probabilities) else None
            confidence = _max_probability(sample_probabilities)
            pred = int(prediction)

            if confidence is not None and confidence < self.binary_min_confidence:
                status = "needs_specialist_review"
                reason = (
                    f"Baixa confiança binária: {confidence:.3f} "
                    f"< {self.binary_min_confidence:.3f}."
                )
            elif pred == 1:
                status = "accepted_attack"
                reason = "Amostra classificada como ataque com confiança suficiente."
            else:
                status = "accepted_normal"
                reason = "Amostra classificada como normal com confiança suficiente."

            decisions.append(
                BinarySampleDecision(
                    batch_index=idx,
                    prediction=pred,
                    probabilities=sample_probabilities,
                    confidence=confidence,
                    decision_status=status,
                    reason=reason,
                )
            )
        return decisions

    def _run_multiclass_workflow(
        self,
        sample: dict[str, Any],
        probability_context: dict[str, Any],
    ) -> tuple[int | None, str | None]:
        from qml.crew_multiclass import CyberPredictMult

        result = CyberPredictMult().crew().kickoff(
            inputs={
                "samples": [sample],
                "shap_explanation": "",
                "probability_context": json.dumps(probability_context, ensure_ascii=False),
            }
        )
        raw_output = getattr(result, "raw", str(result))
        predictions = _extract_binary_predictions(raw_output)
        report = ""
        parsed = _extract_json_object(raw_output)
        if isinstance(parsed, dict):
            report_value = parsed.get("report") or parsed.get("explanation")
            if report_value is not None:
                report = str(report_value)

        if not predictions:
            return None, report or raw_output
        return int(predictions[0]), report or raw_output

    def _run_incident_response_workflow(self, attack_label: str) -> tuple[str, str]:
        if not self.use_incident_response_crew:
            return "", "disabled"
        if not attack_label or attack_label.lower() == "normal":
            return "", "not_applicable"

        from qml.crew_response import IncidentResponseCrew

        result = IncidentResponseCrew().crew().kickoff(inputs={"attack_label": attack_label})
        plan = getattr(result, "raw", str(result)).strip()
        if not plan:
            return "", "empty"
        return plan, "generated"

    def _run_remediation_workflow(
        self,
        attack_label: str,
        result: BatchDetectionResult,
    ) -> tuple[dict[str, Any], str]:
        if not self.use_remediation_crew:
            return {}, "disabled"
        if not attack_label or attack_label.lower() == "normal":
            return {}, "not_applicable"

        from qml.services.remediation import run_initial_remediation_suggestion

        context = {
            "primary_attack_type": attack_label,
            "attack_indices": result.attack_indices,
            "review_indices": result.review_indices,
            "classified_attacks": result.to_dict().get("classified_attacks", []),
            "confidence_policy": {
                "binary_min_confidence": self.binary_min_confidence,
                "multiclass_min_confidence": self.multiclass_min_confidence,
            },
        }
        suggestion = run_initial_remediation_suggestion(
            attack_label=attack_label,
            context=context,
        )
        return suggestion, "generated"

    def predict(self, samples: list[dict[str, Any]]) -> BatchDetectionResult:
        workflow_trace: list[dict[str, Any]] = [
            {
                "stage": "input",
                "status": "received",
                "sample_count": len(samples),
            }
        ]
        binary_probabilities = self.binary_model.predict_proba(samples)
        binary_predictions = self._predict_binary(samples, binary_probabilities)
        if len(binary_predictions) != len(samples):
            raise RuntimeError(
                "O modelo binário retornou uma quantidade de predições diferente "
                "da quantidade de amostras recebidas."
            )
        binary_decisions = self._build_binary_decisions(binary_predictions, binary_probabilities)
        attack_indices = [
            item.batch_index
            for item in binary_decisions
            if item.prediction == 1 and item.decision_status == "accepted_attack"
        ]
        review_indices = [
            item.batch_index
            for item in binary_decisions
            if item.decision_status == "needs_specialist_review"
        ]
        workflow_trace.append(
            {
                "stage": "binary",
                "status": "crew" if self.use_binary_crew else "direct_model",
                "sample_count": len(samples),
                "accepted_attack_count": len(attack_indices),
                "review_count": len(review_indices),
            }
        )

        if not attack_indices:
            return BatchDetectionResult(
                binary_predictions=binary_predictions,
                attack_indices=[],
                binary_decisions=binary_decisions,
                review_indices=review_indices,
                incident_response_status="not_applicable",
                workflow_trace=workflow_trace,
            )

        suspicious_samples = [samples[idx] for idx in attack_indices]
        workflow_trace.append(
            {
                "stage": "multiclass_input",
                "status": "filtered_from_binary_attacks",
                "source_indices": attack_indices,
                "sample_count": len(suspicious_samples),
            }
        )
        multiclass_predictions = self.multiclass_model.predict(suspicious_samples)
        if len(multiclass_predictions) != len(suspicious_samples):
            raise RuntimeError(
                "O modelo multiclasse retornou uma quantidade de predições diferente "
                "da quantidade de amostras suspeitas."
            )
        probabilities = self.multiclass_model.predict_proba(suspicious_samples)

        classified_attacks = []
        for local_idx, attack_type_id in enumerate(multiclass_predictions):
            batch_index = attack_indices[local_idx]
            binary_decision = binary_decisions[batch_index]
            sample_probabilities = probabilities[local_idx] if probabilities else None
            confidence = _max_probability(sample_probabilities)
            workflow_attack_type_id = None
            workflow_report = None
            probability_context = _build_probability_context(
                batch_index=batch_index,
                binary_decision=binary_decision,
                multiclass_prediction=int(attack_type_id),
                multiclass_probabilities=sample_probabilities,
                binary_min_confidence=self.binary_min_confidence,
                multiclass_min_confidence=self.multiclass_min_confidence,
            )
            if self.use_multiclass_crew:
                workflow_attack_type_id, workflow_report = self._run_multiclass_workflow(
                    suspicious_samples[local_idx],
                    probability_context,
                )

            final_attack_type_id = (
                int(workflow_attack_type_id)
                if workflow_attack_type_id is not None
                else int(attack_type_id)
            )
            decision_status = "accepted"
            rejection_reason = ""
            if confidence is not None and confidence < self.multiclass_min_confidence:
                decision_status = "needs_specialist_review"
                rejection_reason = (
                    f"Baixa confiança multiclasse: {confidence:.3f} "
                    f"< {self.multiclass_min_confidence:.3f}."
                )
            elif workflow_attack_type_id is not None and int(workflow_attack_type_id) != int(attack_type_id):
                decision_status = "needs_specialist_review"
                rejection_reason = (
                    "Divergência entre modelo probabilístico "
                    f"({int(attack_type_id)}) e workflow multiclasse ({int(workflow_attack_type_id)})."
                )
            elif final_attack_type_id == 99:
                decision_status = "needs_specialist_review"
                rejection_reason = "Workflow multiclasse retornou rejeição/Normal para amostra suspeita."

            classified_attacks.append(
                ClassifiedSample(
                    batch_index=batch_index,
                    attack_type_id=final_attack_type_id,
                    attack_type_label=ATTACK_LABELS.get(final_attack_type_id, f"Classe_{final_attack_type_id}"),
                    binary_probabilities=binary_decision.probabilities,
                    binary_confidence=binary_decision.confidence,
                    model_attack_type_id=int(attack_type_id),
                    workflow_attack_type_id=workflow_attack_type_id,
                    probabilities=sample_probabilities,
                    confidence=confidence,
                    decision_status=decision_status,
                    rejection_reason=rejection_reason,
                    report=workflow_report,
                )
            )
        workflow_trace.append(
            {
                "stage": "multiclass",
                "status": "crew" if self.use_multiclass_crew else "direct_model",
                "sample_count": len(suspicious_samples),
                "accepted_count": len([
                    item for item in classified_attacks if item.decision_status == "accepted"
                ]),
                "review_count": len([
                    item for item in classified_attacks if item.decision_status != "accepted"
                ]),
            }
        )

        result = BatchDetectionResult(
            binary_predictions=binary_predictions,
            attack_indices=attack_indices,
            binary_decisions=binary_decisions,
            review_indices=review_indices,
            classified_attacks=classified_attacks,
            workflow_trace=workflow_trace,
        )
        response_plan, response_status = self._run_incident_response_workflow(
            result.primary_attack_type_label
        )
        result.incident_response_plan = response_plan
        result.incident_response_status = response_status
        workflow_trace.append(
            {
                "stage": "incident_response",
                "status": response_status,
                "attack_label": result.primary_attack_type_label,
            }
        )
        remediation_suggestion, remediation_status = self._run_remediation_workflow(
            result.primary_attack_type_label,
            result,
        )
        result.remediation_suggestion = remediation_suggestion
        result.remediation_suggestion_status = remediation_status
        workflow_trace.append(
            {
                "stage": "remediation",
                "status": remediation_status,
                "attack_label": result.primary_attack_type_label,
            }
        )
        return result


def build_detection_report(result: BatchDetectionResult, shap_explanation: str = "") -> str:
    if not result.window_has_attack:
        if result.review_indices:
            review_text = ", ".join(str(idx) for idx in result.review_indices)
            return (
                "A janela analisada não teve amostras aceitas como ataque com confiança suficiente. "
                f"As amostras [{review_text}] devem ser revisadas por especialista por baixa confiança."
            )
        return "A janela analisada não apresentou amostras classificadas como ataque pelo modelo binário."

    accepted_classifications = [
        item for item in result.classified_attacks if item.decision_status == "accepted"
    ]
    review_classifications = [
        item for item in result.classified_attacks if item.decision_status != "accepted"
    ]
    grouped = Counter(item.attack_type_label for item in accepted_classifications)
    grouped_text = ", ".join(f"{label}: {count}" for label, count in grouped.most_common())
    indices_text = ", ".join(str(idx) for idx in result.attack_indices)
    review_text = ", ".join(str(idx) for idx in result.review_indices)
    multiclass_review_text = ", ".join(str(item.batch_index) for item in review_classifications)
    grouped_text = grouped_text or "nenhuma classificação aceita por confiança suficiente"
    report = (
        f"A janela contém {len(result.attack_indices)} amostra(s) suspeita(s), "
        f"nos índices [{indices_text}]. Classificação multiclasse consolidada: {grouped_text}. "
        f"Tipo primário selecionado: {result.primary_attack_type_label}."
    )
    if review_text:
        report += f" As amostras [{review_text}] foram encaminhadas para revisão na etapa binária."
    if multiclass_review_text:
        report += (
            f" As amostras [{multiclass_review_text}] foram encaminhadas para revisão "
            "na etapa multiclasse."
        )
    if result.incident_response_status == "generated":
        report += " Um plano de resposta ao incidente foi gerado para o tipo primário aceito."
    elif result.incident_response_status == "disabled":
        report += " A geração automática do plano de resposta está desabilitada neste fluxo."
    if result.remediation_suggestion_status == "generated":
        report += " Uma sugestão inicial de remediação assistida foi gerada para o operador."

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


def _max_probability(probabilities: list[float] | None) -> float | None:
    if not probabilities:
        return None
    return max(float(value) for value in probabilities)


def _build_probability_context(
    batch_index: int,
    binary_decision: BinarySampleDecision,
    multiclass_prediction: int,
    multiclass_probabilities: list[float] | None,
    binary_min_confidence: float,
    multiclass_min_confidence: float,
) -> dict[str, Any]:
    return {
        "batch_index": batch_index,
        "binary_stage": {
            "prediction": binary_decision.prediction,
            "probabilities": binary_decision.probabilities,
            "confidence": binary_decision.confidence,
            "min_confidence": binary_min_confidence,
            "decision_status": binary_decision.decision_status,
        },
        "multiclass_stage": {
            "model_prediction": multiclass_prediction,
            "model_prediction_label": ATTACK_LABELS.get(
                multiclass_prediction,
                f"Classe_{multiclass_prediction}",
            ),
            "probabilities": _label_probabilities(multiclass_probabilities),
            "raw_probabilities": multiclass_probabilities,
            "confidence": _max_probability(multiclass_probabilities),
            "min_confidence": multiclass_min_confidence,
        },
        "rejection_policy": {
            "return_99_when_multiclass_confidence_below_threshold": True,
            "send_to_specialist_when_workflow_disagrees_with_probability_model": True,
        },
    }


def _build_binary_probability_context(
    probabilities: list[list[float]] | None,
    min_confidence: float,
) -> dict[str, Any]:
    rows = []
    for idx, row in enumerate(probabilities or []):
        confidence = _max_probability(row)
        predicted_class = _argmax(row)
        rows.append(
            {
                "batch_index": idx,
                "model_prediction": predicted_class,
                "model_prediction_label": "attack" if predicted_class == 1 else "normal",
                "probabilities": {
                    "normal": float(row[0]) if len(row) > 0 else None,
                    "attack": float(row[1]) if len(row) > 1 else None,
                },
                "confidence": confidence,
                "min_confidence": min_confidence,
                "reject_if_below_min_confidence": True,
            }
        )
    return {
        "samples": rows,
        "rejection_policy": {
            "return_prediction_from_probability_model_when_confident": True,
            "send_to_specialist_review_when_confidence_below_threshold": True,
        },
    }


def _argmax(values: list[float] | None) -> int | None:
    if not values:
        return None
    return max(range(len(values)), key=lambda idx: values[idx])


def _label_probabilities(probabilities: list[float] | None) -> list[dict[str, Any]]:
    if not probabilities:
        return []
    return [
        {
            "class_id": idx,
            "label": ATTACK_LABELS.get(idx, f"Classe_{idx}"),
            "probability": float(probability),
        }
        for idx, probability in enumerate(probabilities)
    ]


def _read_float_env(name: str, default: float) -> float:
    raw_value = os.getenv(name)
    if raw_value is None or raw_value.strip() == "":
        return default
    try:
        return float(raw_value)
    except ValueError:
        raise ValueError(f"{name} must be a float, got {raw_value!r}.") from None
