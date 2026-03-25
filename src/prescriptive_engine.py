"""
Motor de Análisis Prescriptivo (What-If) para Riesgo Crediticio.

Sistema de Simulación Secuencial de Riesgo — evalúa alternativas de crédito
cuando una solicitud es rechazada por alto riesgo. Emula el comportamiento de
un comité de crédito humano: recorre los plazos disponibles en orden ascendente
desde el plazo actual, y para cada combinación verifica simultáneamente que la
probabilidad de impago sea aceptable y que la relación cuota/ingreso (DTI) no
supere el límite de política institucional.

Ejemplo de uso:
    from src.prescriptive_engine import simulate_alternatives

    result = simulate_alternatives(
        input_df=current_input,
        current_credit=100000,
        current_months=36,
        current_income=5000,
        model=trained_model,
        threshold=0.3
    )
"""

import logging
from typing import Any

import pandas as pd

from utils.config import (
    AVAILABLE_TERMS,
    HIGH_RISK_THRESHOLD,
    POLICY_MAX_DTI,
    SIMULATION_ANNUAL_RATE,
)

logger = logging.getLogger(__name__)

# Cantidad reducida como último recurso cuando ningún plazo funciona al monto
# completo.  Se aplica sobre el crédito original y se evalúa solo en el plazo
# máximo (60 meses).
REDUCED_AMOUNT_RATIO: float = 0.70


def calculate_monthly_payment(
    credit: float,
    months: int,
    annual_rate: float = SIMULATION_ANNUAL_RATE,
) -> float:
    """Calcula el pago mensual usando fórmula de amortización francesa.

    Args:
        credit: Monto del préstamo.
        months: Plazo en meses.
        annual_rate: Tasa de interés anual (default: 15%).

    Returns:
        Pago mensual calculado.
    """
    if months <= 0 or credit <= 0:
        return 0.0

    monthly_rate = annual_rate / 12
    if monthly_rate == 0:
        return credit / months

    # Fórmula de amortización francesa
    payment = (
        credit
        * (monthly_rate * (1 + monthly_rate) ** months)
        / ((1 + monthly_rate) ** months - 1)
    )

    return payment


def evaluate_scenario(
    input_df: pd.DataFrame,
    credit: float,
    months: int,
    income: float,
    model: Any,
    threshold: float,
) -> dict:
    """Evalúa un escenario de crédito específico.

    Args:
        input_df: DataFrame con las características del solicitante.
        credit: Monto del préstamo a evaluar.
        months: Plazo en meses a evaluar.
        income: Ingresos mensuales del solicitante.
        model: Modelo de predicción de riesgo entrenado.
        threshold: Umbral de riesgo para aprobación.

    Returns:
        Diccionario con resultados del escenario.

    Raises:
        ValueError: Si el modelo no puede realizar predicciones.
    """
    try:
        scenario_df = input_df.copy()

        monthly_payment = calculate_monthly_payment(credit, months)
        annual_payment = monthly_payment * 12

        if "AMT_CREDIT" in scenario_df.columns:
            scenario_df["AMT_CREDIT"] = credit
        if "AMT_ANNUITY" in scenario_df.columns:
            scenario_df["AMT_ANNUITY"] = annual_payment
        if "AMT_INCOME_TOTAL" in scenario_df.columns:
            scenario_df["AMT_INCOME_TOTAL"] = income * 12

        probability = model.predict_proba(scenario_df)[0][1]
        is_approved = probability <= threshold

        debt_to_income = monthly_payment / income if income > 0 else float("inf")
        is_affordable = debt_to_income <= POLICY_MAX_DTI

        return {
            "viable": is_approved and is_affordable,
            "probability": probability,
            "monthly_payment": monthly_payment,
            "debt_to_income_ratio": debt_to_income,
            "credit": credit,
            "months": months,
        }

    except Exception as e:
        logger.error(f"Error evaluando escenario: {e}")
        raise ValueError(f"Error en predicción del modelo: {e}") from e


def _build_result(scenario: dict, scenario_name: str) -> dict:
    """Construye el diccionario de respuesta estandarizado."""
    return {
        "viable": True,
        "scenario": scenario_name,
        "new_credit": scenario["credit"],
        "new_months": scenario["months"],
        "new_monthly_payment": scenario["monthly_payment"],
        "new_probability": scenario["probability"],
        "debt_to_income_ratio": scenario["debt_to_income_ratio"],
    }


def _no_viable_result() -> dict:
    """Retorna el diccionario de respuesta cuando no hay alternativa viable."""
    return {
        "viable": False,
        "scenario": "none",
        "new_credit": None,
        "new_months": None,
        "new_monthly_payment": None,
        "new_probability": None,
        "debt_to_income_ratio": None,
    }


def simulate_alternatives(
    input_df: pd.DataFrame,
    current_credit: float,
    current_months: int,
    current_income: float,
    model: Any,
    threshold: float = HIGH_RISK_THRESHOLD,
) -> dict:
    """Simula alternativas de crédito mediante búsqueda secuencial por plazos.

    Sistema de Simulación Secuencial de Riesgo: emula la dinámica de un comité
    de crédito que evalúa opciones de manera escalonada, comenzando por el
    plazo más cercano al solicitado y avanzando hasta el máximo permitido.
    Para cada plazo se verifican dos condiciones simultáneas:
      1. Probabilidad de impago ≤ umbral de riesgo
      2. Relación cuota/ingreso (DTI) ≤ POLICY_MAX_DTI

    Si ningún plazo funciona con el monto original al 100 %, se aplica un
    último intento reduciendo el crédito al 70 % y evaluándolo al plazo máximo
    (60 meses).

    Args:
        input_df: DataFrame con características originales del solicitante.
        current_credit: Monto del crédito original solicitado.
        current_months: Plazo original en meses.
        current_income: Ingresos mensuales del solicitante.
        model: Modelo de predicción de riesgo entrenado.
        threshold: Umbral de riesgo para aprobación (default: 30%).

    Returns:
        Diccionario con resultados de la simulación:
        - viable: True si alguna alternativa fue aprobada
        - scenario: Nombre del escenario que funcionó
        - new_credit: Monto aprobado (si viable)
        - new_months: Plazo aprobado (si viable)
        - new_monthly_payment: Cuota mensual (si viable)
        - new_probability: Nueva probabilidad de impago
        - debt_to_income_ratio: Relación cuota/ingreso

    Ejemplo:
        >>> result = simulate_alternatives(df, 100000, 36, 5000, model)
        >>> if result['viable']:
        ...     print(
        ...         f"Aprobado: ${result['new_credit']:,.0f} "
        ...         f"a {result['new_months']} meses"
        ...     )
    """
    logger.info(f"Iniciando simulación What-If para crédito: ${current_credit:,.0f}")

    # ── Evaluación original (referencia) ────────────────────────────────────
    original = evaluate_scenario(
        input_df, current_credit, current_months, current_income, model, threshold
    )

    if original["viable"]:
        logger.info("Escenario original ya es viable, no requiere simulación")
        return _build_result(original, "original")

    logger.info(
        f"Escenario original rechazado (prob={original['probability']:.2%}, "
        f"dti={original['debt_to_income_ratio']:.2%}). "
        f"Iniciando búsqueda secuencial por plazos..."
    )

    # ── Fase 1: Búsqueda secuencial a monto completo ────────────────────────
    # Filtrar plazos disponibles que sean >= al plazo actual y recorrerlos
    # en orden ascendente (el primero que funcione gana).
    candidate_terms = [t for t in AVAILABLE_TERMS if t >= current_months]

    for term in candidate_terms:
        scenario = evaluate_scenario(
            input_df, current_credit, term, current_income, model, threshold
        )

        logger.info(
            f"  Plazo {term:>2d} meses — prob={scenario['probability']:.2%}, "
            f"dti={scenario['debt_to_income_ratio']:.2%}"
        )

        if scenario["viable"]:
            logger.info(
                f"✓ Alternativa viable encontrada: {term} meses, "
                f"${current_credit:,.0f} al 100%"
            )
            return _build_result(scenario, "extended_term")

    # ── Fase 2: Reducción de monto (70 %) al plazo máximo ──────────────────
    max_term = AVAILABLE_TERMS[-1]
    reduced_credit = current_credit * REDUCED_AMOUNT_RATIO

    reduced_pct = f"{REDUCED_AMOUNT_RATIO:.0%}"
    logger.info(
        f"Ningún plazo viable al 100%. "
        f"Intentando monto reducido (${reduced_credit:,.0f} = {reduced_pct}) "
        f"con plazo máximo de {max_term} meses"
    )

    scenario = evaluate_scenario(
        input_df, reduced_credit, max_term, current_income, model, threshold
    )

    logger.info(
        f"  Plazo {max_term:>2d} meses (monto reducido) — "
        f"prob={scenario['probability']:.2%}, "
        f"dti={scenario['debt_to_income_ratio']:.2%}"
    )

    if scenario["viable"]:
        logger.info(
            f"✓ Alternativa viable encontrada: {max_term} meses, "
            f"${reduced_credit:,.0f} al {REDUCED_AMOUNT_RATIO:.0%}"
        )
        return _build_result(scenario, "reduced_amount")

    # ── Sin alternativa viable ──────────────────────────────────────────────
    logger.info("✗ Simulación completada: Ninguna alternativa viable")
    return _no_viable_result()
