import os
import sys

# ==============================================================================
# CONFIGURACIÓN DE PATH ABSOLUTO (CORE ARCHITECTURE)
# ==============================================================================
# Este bloque permite a Streamlit ejecutar el script desde el subdirectorio /app
# mientras mantiene la capacidad de importar módulos hermanos (como /utils y /src)
# localizados en la raíz del proyecto. Previene el error ModuleNotFoundError.
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import json
import logging

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
import streamlit as st

from src.prescriptive_engine import simulate_alternatives
from utils.config import (
    ANNUAL_INTEREST_RATE,
    AVAILABLE_TERMS,
    FEATURE_MAPPING,
    FEATURE_MEDIANS_FILE_PATH,
    HIGH_RISK_THRESHOLD,
    MODEL_FILE_PATH,
    POLICY_MAX_AGE_AT_END,
    POLICY_MAX_DTI,
    POLICY_MAX_LTI,
    SHAP_PLOT_MARGIN_MULTIPLIER,
    SIMULATION_REDUCTION_STEP,
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

FACTORES_INMUTABLES = [
    "DAYS_BIRTH",
    "CODE_GENDER",
    "FLAG_OWN_CAR",
    "NAME_FAMILY_STATUS",
]


# ==============================================================================
# HELPER FUNCTIONS - Credit Intelligence System 3.0
# ==============================================================================


def consolidate_shap_values(
    feature_names: list[str],
    display_names: list[str],
    values: np.ndarray,
) -> tuple[list[str], list[str], np.ndarray]:
    """Consolidate EXT_SOURCE_2 and EXT_SOURCE_3 into 'Historial Crediticio'.

    Sums the SHAP values of both external credit scores into one consolidated
    factor, removing the individual entries from the arrays.

    Args:
        feature_names: Original model feature names.
        display_names: Human-readable feature names.
        values: SHAP values array for a single prediction.

    Returns:
        Tuple of consolidated (feature_names, display_names, values).
    """
    bureau_features = {"EXT_SOURCE_2", "EXT_SOURCE_3"}
    bureau_indices = [
        i for i, name in enumerate(feature_names) if name in bureau_features
    ]

    if len(bureau_indices) < 2:
        return feature_names, display_names, values

    bureau_sum = sum(values[i] for i in bureau_indices)
    keep_indices = [i for i in range(len(feature_names)) if i not in bureau_indices]

    new_names = [feature_names[i] for i in keep_indices]
    new_display = [display_names[i] for i in keep_indices]
    new_values = np.array([values[i] for i in keep_indices])

    new_names.append("EXT_SOURCE_BUREAU")
    new_display.append("Historial Crediticio (Buró)")
    new_values = np.append(new_values, bureau_sum)

    return new_names, new_display, new_values


def convert_days_to_years(
    feature_names: list[str],
    display_names: list[str],
    values: np.ndarray,
) -> tuple[list[str], list[str], np.ndarray]:
    """Convert DAYS_* features from negative days to positive years.

    Renames display names to indicate years and converts values so the
    SHAP sign is preserved while the magnitude reflects the year equivalent.

    Args:
        feature_names: Original model feature names.
        display_names: Human-readable feature names.
        values: SHAP values array.

    Returns:
        Tuple of (feature_names, display_names, values) with DAYS converted.
    """
    new_display = list(display_names)
    new_values = values.copy()

    for i, name in enumerate(feature_names):
        if name.startswith("DAYS_"):
            new_display[i] = new_display[i].replace("(Años)", "").strip() + " (Años)"
            # Preserve SHAP sign; magnitude stays the same since it's the
            # model's sensitivity, not the raw feature value.
            new_values[i] = abs(values[i]) if values[i] != 0 else 0.0
            # Re-apply original sign
            if values[i] < 0:
                new_values[i] = -new_values[i]

    return feature_names, new_display, new_values


def filter_top_factors(
    display_names: list[str],
    values: np.ndarray,
    top_n: int = 6,
    min_impact: float = 0.001,
) -> tuple[list[str], np.ndarray]:
    """Keep only the Top N factors with absolute impact above min_impact.

    Args:
        display_names: Human-readable feature names.
        values: SHAP values array.
        top_n: Maximum number of factors to retain.
        min_impact: Minimum absolute SHAP value to consider.

    Returns:
        Tuple of filtered (display_names, values) sorted by absolute impact descending.
    """
    abs_values = np.abs(values)
    valid_mask = abs_values >= min_impact
    valid_indices = np.where(valid_mask)[0]

    if len(valid_indices) == 0:
        return [], np.array([])

    sorted_indices = valid_indices[np.argsort(abs_values[valid_indices])[-top_n:][::-1]]
    filtered_names = [display_names[i] for i in sorted_indices]
    filtered_values = values[sorted_indices]

    return filtered_names, filtered_values


# ==============================================================================
# LEGACY / EXISTING FUNCTIONS
# ==============================================================================


@st.cache_resource
def load_assets():
    try:
        model = joblib.load(MODEL_FILE_PATH)
        with open(FEATURE_MEDIANS_FILE_PATH) as f:
            medians = json.load(f)
        classifier = model.named_steps["classifier"]
        bg_data = pd.DataFrame([medians])
        explainer = shap.TreeExplainer(
            classifier,
            data=bg_data,
            model_output="probability",
            feature_perturbation="interventional",
        )
        return model, medians, explainer
    except Exception as e:
        logger.error(f"Error: {e}")
        st.error(f"Error: {e}")
        return None, None, None


def transform_feature_names(feature_names):
    return [FEATURE_MAPPING.get(name, name) for name in feature_names]


def get_financial_advice(
    shap_values: object,
    feature_names: list[str],
    display_names: list[str | None],
) -> dict[str, object]:
    """Generate professional banking narrative from SHAP factor analysis.

    Identifies the most mitigating and most adversarial credit risk factors
    and produces a human-readable advisory summary. Differentiates between
    immutable factors (e.g. age, gender) and actionable ones to provide
    tailored recommendations.

    Args:
        shap_values: Array of SHAP values for a single prediction.
        feature_names: Original model feature names.
        display_names: Human-readable feature names for presentation.

    Returns:
        Dictionary with keys:
            - advice_text: Professional narrative string.
            - top_positive: Tuple of (display_name, value) for best mitigating factor.
            - top_negative: Tuple of (display_name, value) for worst adversarial factor.
            - worst_feature_name: Original feature name of the worst adversarial factor.
    """
    values = (
        shap_values.values
        if hasattr(shap_values, "values")
        else np.asarray(shap_values)
    )

    positive_mask = values < 0  # Negative SHAP → reduces risk (mitigating)
    negative_mask = values > 0  # Positive SHAP → increases risk (adversarial)

    pos_indices = np.where(positive_mask)[0]
    neg_indices = np.where(negative_mask)[0]

    # --- Top mitigating factor ---
    if len(pos_indices) > 0:
        best_idx = pos_indices[np.argmin(values[pos_indices])]
        top_positive = (display_names[best_idx], float(values[best_idx]))
    else:
        top_positive = ("Ninguno identificado", 0.0)

    # --- Top adversarial factor ---
    if len(neg_indices) > 0:
        worst_idx = neg_indices[np.argmax(values[neg_indices])]
        top_negative = (display_names[worst_idx], float(values[worst_idx]))
        worst_feature_name = feature_names[worst_idx]
    else:
        top_negative = ("Ninguno identificado", 0.0)
        worst_feature_name = ""

    # --- Narrative construction ---
    pos_name, pos_val = top_positive
    neg_name, neg_val = top_negative
    is_immutable = worst_feature_name in FACTORES_INMUTABLES

    lines: list[str] = []
    lines.append("**Resumen Ejecutivo de Factores de Riesgo**\n")

    if neg_name != "Ninguno identificado":
        lines.append(
            f"El factor que más incrementa el riesgo en tu solicitud "
            f"es **{neg_name}**, con un impacto de +{neg_val:.4f} "
            "en la probabilidad de impago."
        )

    if pos_name != "Ninguno identificado":
        lines.append(
            f"Por otro lado, tu principal fortaleza es **{pos_name}**, "
            f"que reduce el riesgo en {abs(pos_val):.4f} puntos."
        )

    if neg_name != "Ninguno identificado" and pos_name != "Ninguno identificado":
        if is_immutable:
            lines.append(
                f"\n**Recomendación:** Tu **{neg_name}** es una variable "
                "estadística fija. Para compensar este riesgo, te recomendamos "
                "enfocarte en variables que sí puedes controlar, como reducir "
                "el monto del préstamo o incrementar el plazo de pago."
            )
        else:
            lines.append(
                f"\n**Recomendación:** Tienes una oportunidad directa de mejora: "
                f"si ajustas el **{neg_name}** (por ejemplo, solicitando un "
                "monto menor), tu perfil crediticio se fortalecerá de inmediato."
            )

    if neg_name == "Ninguno identificado" and pos_name == "Ninguno identificado":
        lines.append(
            "No se identificaron factores con impacto significativo. "
            "Tu perfil se comporta de acuerdo con la media poblacional."
        )

    advice_text = "\n\n".join(lines)

    return {
        "advice_text": advice_text,
        "top_positive": top_positive,
        "top_negative": top_negative,
        "worst_feature_name": worst_feature_name,
    }


def validate_business_policy(
    annuity_m: float,
    income_m: float,
    credit: float,
    age: int,
    months: int,
) -> list[str]:
    """Validate loan parameters against business policy limits.

    Args:
        annuity_m: Monthly annuity (payment).
        income_m: Monthly income.
        credit: Total credit amount.
        age: Applicant age in years.
        months: Loan term in months.

    Returns:
        List of policy violation messages. Empty if all checks pass.
    """
    violations: list[str] = []

    dti = annuity_m / income_m if income_m > 0 else float("inf")
    if dti > POLICY_MAX_DTI:
        msg = (
            f"Relación Deuda/Ingreso (DTI): {dti:.2%} excede"
            f" el máximo permitido de {POLICY_MAX_DTI:.0%}."
        )
        violations.append(msg)

    lti = credit / (income_m * 12) if income_m > 0 else float("inf")
    if lti > POLICY_MAX_LTI:
        msg = (
            f"Relación Préstamo/Ingreso (LTI): {lti:.2f}x excede"
            f" el máximo permitido de {POLICY_MAX_LTI:.1f}x."
        )
        violations.append(msg)

    age_at_end = age + months / 12
    if age_at_end > POLICY_MAX_AGE_AT_END:
        msg = (
            f"Edad al finalizar el préstamo: {age_at_end:.1f} años"
            f" excede el máximo permitido de {POLICY_MAX_AGE_AT_END}."
        )
        violations.append(msg)

    return violations


# ==============================================================================
# ENHANCED NARRATIVE GENERATION (System 3.0)
# ==============================================================================


def generate_veredicto(
    prob: float,
    risk_level: str,
    capacity_status: str,
    fortaleza: tuple[str, float],
    riesgo: tuple[str, float],
    policy_reject: bool,
) -> str:
    """Generate a one-liner veredicto summary for the credit decision.

    Args:
        prob: Default probability.
        risk_level: Bajo / Medio / Alto.
        capacity_status: Sostenible / Crítica.
        fortaleza: (name, value) of strongest green bar.
        riesgo: (name, value) of strongest red bar.
        policy_reject: Whether business policy was violated.

    Returns:
        Single-sentence veredicto string.
    """
    if policy_reject:
        return (
            f"Solicitud rechazada por política interna. "
            f"Riesgo estadístico: {risk_level} ({prob:.1%}). "
            f"Capacidad de pago: {capacity_status}."
        )

    if risk_level == "Alto":
        return (
            f"Riesgo elevado ({prob:.1%}). "
            f"El factor adverso principal es **{riesgo[0]}**. "
            f"Se requiere ajuste de condiciones para aprobar."
        )

    fort_name = (
        fortaleza[0] if fortaleza[0] != "Ninguno identificado" else "perfil general"
    )
    return (
        f"Perfil aprobable con riesgo {risk_level.lower()} ({prob:.1%}). "
        f"Fortaleza principal: **{fort_name}**. "
        f"Capacidad de pago: {capacity_status}."
    )


# ==============================================================================
# MAIN APPLICATION
# ==============================================================================


def main():
    st.set_page_config(
        page_title="Consultor Crediticio Pro - Anthony",
        page_icon="🏦",
        layout="wide",
        initial_sidebar_state="expanded",
    )
    st.markdown(
        """
    <style>
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    .stDeployButton {display:none;}
    </style>
    """,
        unsafe_allow_html=True,
    )
    st.title("🏦 Simulador de Riesgo Crediticio")

    model, medians, explainer = load_assets()
    if not model:
        st.stop()

    feature_names = model.named_steps["classifier"].feature_name_
    norm_medians = {
        str(k).lower().replace(" ", "").replace("_", ""): v for k, v in medians.items()
    }

    st.sidebar.header("📋 Parámetros del Solicitante")
    bureau = st.sidebar.slider("📊 Score de Buró (0-1000)", 0, 1000, 500) / 1000.0
    credit = st.sidebar.number_input(
        "💰 Monto Préstamo ($)", min_value=0.0, value=100000.0, step=10000.0
    )
    months = st.sidebar.selectbox("📅 Plazo (Meses)", [12, 24, 36, 48, 60], index=2)
    income_m = st.sidebar.number_input(
        "💵 Ingresos Mensuales ($)", min_value=0.0, value=5000.0, step=500.0
    )

    rate = ANNUAL_INTEREST_RATE / 12
    annuity_m = credit * (rate * (1 + rate) ** months) / ((1 + rate) ** months - 1)
    annuity_y = annuity_m * 12

    st.sidebar.info(f"📅 Cuota Mensual: ${annuity_m:,.2f}")
    if annuity_m > 0.4 * income_m:
        st.sidebar.warning("⚠️ La cuota excede el 40% de los ingresos.")

    age = st.sidebar.slider("🎂 Edad", 18, 90, 35)

    input_values = [
        norm_medians.get(str(f).lower().replace(" ", "").replace("_", ""), 0.0)
        for f in feature_names
    ]
    input_df = pd.DataFrame([input_values], columns=feature_names)

    mapping = {
        "EXT_SOURCE_2": bureau,
        "EXT_SOURCE_3": bureau,
        "AMT_CREDIT": credit,
        "AMT_ANNUITY": annuity_y,
        "AMT_INCOME_TOTAL": income_m * 12,
        "DAYS_BIRTH": age * -365,
    }

    for col, val in mapping.items():
        if col in input_df.columns:
            input_df[col] = val

    if st.sidebar.button("🔍 Evaluar Riesgo", use_container_width=True):
        # ==============================================================================
        # SISTEMA DE SIMULACIÓN SECUENCIAL DE RIESGO
        # ==============================================================================
        violations = validate_business_policy(annuity_m, income_m, credit, age, months)
        policy_reject = bool(violations)

        if policy_reject:
            st.error(
                "**Solicitud con violaciones de políticas de negocio:**\n\n"
                + "\n".join(f"- {v}" for v in violations)
            )

        prob = model.predict_proba(input_df)[0][1]
        high_risk = prob > HIGH_RISK_THRESHOLD

        simulation_result = None
        if policy_reject or high_risk:
            st.markdown("---")
            st.subheader("🔄 Sistema de Simulación Secuencial: Análisis Prescriptivo")

            if policy_reject and high_risk:
                st.warning(
                    "⚠️ La solicitud presenta tanto violaciones de políticas como alto riesgo estadístico."
                )
            elif policy_reject:
                st.warning(
                    "⚠️ La solicitud presenta violaciones de políticas de negocio."
                )
            else:
                st.warning(
                    f"⚠️ La solicitud presenta alto riesgo estadístico (prob: {prob:.2%})."
                )

            with st.spinner("Simulando escenarios alternativos..."):
                simulation_result = simulate_alternatives(
                    input_df=input_df,
                    current_credit=credit,
                    current_months=months,
                    current_income=income_m,
                    model=model,
                    threshold=HIGH_RISK_THRESHOLD,
                )

            if simulation_result["viable"]:
                st.success("✅ **Se encontró una alternativa viable**")

                scenario_labels = {
                    "extended_term": "📅 Extensión de Plazo",
                    "reduced_amount": "📉 Reducción de Monto",
                }

                scenario_label = scenario_labels.get(
                    simulation_result["scenario"], simulation_result["scenario"]
                )

                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric(
                        "Nuevo Monto",
                        f"${simulation_result['new_credit']:,.2f}",
                        delta=f"${simulation_result['new_credit'] - credit:,.2f}",
                        delta_color="inverse",
                    )
                with col2:
                    st.metric(
                        "Nuevo Plazo",
                        f"{simulation_result['new_months']} meses",
                        delta=f"{simulation_result['new_months'] - months} meses",
                    )
                with col3:
                    st.metric(
                        "Nueva Cuota Mensual",
                        f"${simulation_result['new_monthly_payment']:,.2f}",
                        delta=f"${simulation_result['new_monthly_payment'] - annuity_m:,.2f}",
                        delta_color="inverse",
                    )

                st.info(f"**Escenario aplicado:** {scenario_label}")
                st.caption(
                    f"Nueva probabilidad de impago: {simulation_result['new_probability']:.2%} | "
                    f"Relación cuota/ingreso: {simulation_result['debt_to_income_ratio']:.1%}"
                )
            else:
                st.error(
                    "❌ No se encontraron alternativas viables dentro de los parámetros de simulación."
                )
                st.caption(
                    f"Parámetros evaluados: Plazo máximo {AVAILABLE_TERMS[-1]} meses, "
                    f"Reducción de monto al {SIMULATION_REDUCTION_STEP:.0%}"
                )
        else:
            st.success(f"🟢 CRÉDITO APROBADO - Probabilidad: {prob:.2%}")

        # ==============================================================================
        # KPI DASHBOARD (System 3.0)
        # ==============================================================================
        st.markdown("---")
        st.subheader("📈 Indicadores Clave")

        health_score = round((1 - prob) * 100, 1)

        if prob <= HIGH_RISK_THRESHOLD * 0.5:
            risk_level = "Bajo"
            risk_color = "#2A9D8F"
        elif prob <= HIGH_RISK_THRESHOLD:
            risk_level = "Medio"
            risk_color = "#E9C46A"
        else:
            risk_level = "Alto"
            risk_color = "#E63946"

        dti_ratio = annuity_m / income_m if income_m > 0 else float("inf")
        capacity_status = "Sostenible" if dti_ratio <= 0.4 else "Crítica"

        if prob <= HIGH_RISK_THRESHOLD and not policy_reject:
            recommendation = "Aprobar"
        elif prob <= HIGH_RISK_THRESHOLD:
            recommendation = "Ajustar"
        else:
            recommendation = "Rechazar"

        kpi1, kpi2, kpi3, kpi4 = st.columns(4)
        with kpi1:
            st.metric("🩺 Health Score", f"{health_score}/100")
        with kpi2:
            st.markdown(
                f"**Nivel de Riesgo**<br>"
                f'<span style="color:{risk_color};font-size:1.5rem;font-weight:bold">'
                f"{risk_level}</span>",
                unsafe_allow_html=True,
            )
        with kpi3:
            st.metric("📋 Recomendación", recommendation)
        with kpi4:
            cap_color = "#2A9D8F" if capacity_status == "Sostenible" else "#E63946"
            st.markdown(
                f"**Capacidad de Pago**<br>"
                f'<span style="color:{cap_color};font-size:1.5rem;font-weight:bold">'
                f"{capacity_status}</span><br>"
                f'<span style="font-size:0.85rem">DTI: {dti_ratio:.1%}</span>',
                unsafe_allow_html=True,
            )

        # ==============================================================================
        # ANÁLISIS SHAP - Credit Intelligence System 3.0
        # ==============================================================================
        st.markdown("---")

        with st.expander("¿Qué está decidiendo tu crédito?", expanded=True):
            st.markdown("""
            **Guía de Interpretación:**
            - 🔴 **Barras Rojas**: Aumentan el riesgo (factores adversos)
            - 🟢 **Barras Verdes**: Disminuyen el riesgo (factores mitigantes)
            - **Valores numéricos**: Impacto exacto en la probabilidad
            """)

            try:
                shap_values = explainer(input_df)

                if len(shap_values.values.shape) == 3:
                    exp_to_plot = shap_values[0, :, 1]
                else:
                    exp_to_plot = shap_values[0]

                raw_feature_names = list(feature_names)
                raw_display_names = transform_feature_names(raw_feature_names)
                raw_values = np.asarray(exp_to_plot.values)

                # --- Consolidation & filtering pipeline ---
                step_names, step_display, step_values = consolidate_shap_values(
                    raw_feature_names, raw_display_names, raw_values
                )
                step_names, step_display, step_values = convert_days_to_years(
                    step_names, step_display, step_values
                )

                top_display, top_values = filter_top_factors(
                    step_display, step_values, top_n=6, min_impact=0.001
                )

                if len(top_display) == 0:
                    st.info("No se encontraron factores con impacto significativo.")
                    return

                # --- Chart inside bordered container ---
                with st.container(border=True):
                    n_vars = len(top_display)
                    dynamic_height = max(4, n_vars * 0.55)

                    fig, ax = plt.subplots(figsize=(10, dynamic_height))

                    y_pos = np.arange(n_vars)
                    colors = ["#E63946" if v > 0 else "#2A9D8F" for v in top_values]
                    bars = ax.barh(y_pos, top_values, color=colors)

                    ax.set_yticks(y_pos)
                    ax.set_yticklabels(top_display, fontsize=10)
                    ax.invert_yaxis()
                    ax.tick_params(axis="y", pad=15)

                    ax.set_xlabel("Impacto en Probabilidad de Impago", fontsize=11)
                    ax.set_title(
                        "Top Factores de Riesgo Crediticio",
                        fontsize=14,
                        fontweight="bold",
                        pad=25,
                    )

                    ax.axvline(x=0, color="black", linestyle="-", linewidth=0.8)

                    max_abs_value = np.max(np.abs(top_values))
                    margin_multiplier = SHAP_PLOT_MARGIN_MULTIPLIER
                    x_min = -max_abs_value * margin_multiplier
                    x_max = max_abs_value * margin_multiplier
                    ax.set_xlim(x_min, x_max)

                    text_offset = max_abs_value * 0.02

                    for bar, value in zip(bars, top_values, strict=True):
                        width = bar.get_width()
                        y_center = bar.get_y() + bar.get_height() / 2

                        if width >= 0:
                            label_x = width + text_offset
                            ha = "left"
                        else:
                            label_x = width - text_offset
                            ha = "right"

                        ax.text(
                            label_x,
                            y_center,
                            f"{value:.3f}",
                            va="center",
                            ha=ha,
                            fontsize=9,
                            fontweight="bold",
                            color="black",
                        )

                    plt.subplots_adjust(left=0.4)
                    plt.tight_layout()

                    st.pyplot(fig)
                    plt.close(fig)

                # --- Narrative generation (System 3.0) ---
                green_indices = [i for i, v in enumerate(top_values) if v < 0]
                red_indices = [i for i, v in enumerate(top_values) if v > 0]

                if green_indices:
                    best_green = green_indices[np.argmin(top_values[green_indices])]
                    fortaleza = (top_display[best_green], float(top_values[best_green]))
                else:
                    fortaleza = ("Ninguno identificado", 0.0)

                if red_indices:
                    worst_red = red_indices[np.argmax(top_values[red_indices])]
                    riesgo = (top_display[worst_red], float(top_values[worst_red]))
                else:
                    riesgo = ("Ninguno identificado", 0.0)

                advice = get_financial_advice(
                    exp_to_plot, raw_feature_names, raw_display_names
                )

                veredicto = generate_veredicto(
                    prob=prob,
                    risk_level=risk_level,
                    capacity_status=capacity_status,
                    fortaleza=fortaleza,
                    riesgo=riesgo,
                    policy_reject=policy_reject,
                )

                with st.container(border=True):
                    st.markdown("### 🎯 Veredicto")
                    st.markdown(veredicto)

                with st.chat_message("assistant"):
                    st.markdown(advice["advice_text"])

                st.caption(
                    "Nota Técnica: Solo se muestran las variables que desvían el riesgo base. "
                    "Los campos no ingresados manualmente utilizan la mediana poblacional del "
                    "segmento, aportando un impacto neutral (0) en la desviación."
                )

            except Exception as e:
                logger.error(f"Error SHAP: {e}")
                st.warning("⚠️ No se pudo generar el análisis de factores.")

    st.sidebar.markdown("---")
    st.sidebar.info("""
    **ℹ️ Guía de Uso:**
    1. Ajuste los parámetros en la barra lateral
    2. Haga clic en 'Evaluar Riesgo'
    3. Revise los factores que afectan su riesgo
    """)


if __name__ == "__main__":
    main()
