from __future__ import annotations

import json
import re
from typing import Any

from qml.crew_remediation import RemediationChatCrew


def run_remediation_chat(
    attack_label: str,
    operator_message: str,
    context: dict[str, Any] | list[Any] | str | None = None,
) -> dict[str, Any]:
    context_text = _serialize_context(context)
    result = RemediationChatCrew().crew().kickoff(
        inputs={
            "attack_label": attack_label,
            "operator_message": operator_message,
            "context": context_text,
        }
    )
    raw_output = getattr(result, "raw", str(result))
    parsed = _extract_json_object(raw_output)
    if parsed is None:
        return {
            "assistant_message": raw_output,
            "attack_label": attack_label,
            "recommended_mode": "manual",
            "confidence_note": "Não foi possível estruturar a resposta do LLM como JSON.",
            "commands": [],
            "manual_steps": [],
            "questions_for_operator": ["Revise a resposta textual antes de executar qualquer ação."],
            "do_not_do": ["Não execute comandos sem validação e aprovação."],
            "raw_output": raw_output,
        }
    parsed.setdefault("raw_output", raw_output)
    parsed["automatic_execution_available"] = False
    parsed["automatic_execution_note"] = (
        "Esta versão sugere comandos, mas não executa shell automaticamente. "
        "A execução automática deve ser implementada por adaptadores allowlistados "
        "e aprovação explícita do operador."
    )
    return parsed


def run_initial_remediation_suggestion(
    attack_label: str,
    context: dict[str, Any] | list[Any] | str | None = None,
) -> dict[str, Any]:
    operator_message = (
        "Um ataque foi confirmado pelo workflow de detecção. "
        "Sugira imediatamente os comandos defensivos iniciais para reduzir danos, "
        "preservar evidências e orientar o operador sobre execução manual ou assistida."
    )
    return run_remediation_chat(
        attack_label=attack_label,
        operator_message=operator_message,
        context=context,
    )


def _serialize_context(context: dict[str, Any] | list[Any] | str | None) -> str:
    if context is None:
        return "{}"
    if isinstance(context, str):
        return context
    return json.dumps(context, ensure_ascii=False, indent=2)


def _extract_json_object(text: str) -> dict[str, Any] | None:
    if not text:
        return None

    code_block = re.search(r"```(?:json)?\s*(.*?)```", text, re.DOTALL | re.IGNORECASE)
    if code_block:
        candidate = code_block.group(1).strip()
        try:
            parsed = json.loads(candidate)
            return parsed if isinstance(parsed, dict) else None
        except json.JSONDecodeError:
            pass

    try:
        parsed = json.loads(text)
        return parsed if isinstance(parsed, dict) else None
    except json.JSONDecodeError:
        pass

    for candidate in re.findall(r"\{.*\}", text, flags=re.DOTALL):
        try:
            parsed = json.loads(candidate)
            return parsed if isinstance(parsed, dict) else None
        except json.JSONDecodeError:
            continue

    return None
