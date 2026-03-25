"""
Centralized configuration module for the Credit Risk Analytics project.

This module contains all configuration constants, mappings, and parameters
used across the project, including the Streamlit app, data processing pipelines,
and model training scripts. Centralizing these values ensures consistency
and makes the codebase easier to maintain.
"""

from typing import Final

# =============================================================================
# FEATURE MAPPING: Maps internal feature names to user-friendly Spanish labels
# Used by the Streamlit app for displaying feature names in visualizations
# =============================================================================
FEATURE_MAPPING: Final[dict[str, str]] = {
    "AMT_CREDIT": "Monto Préstamo Solicitado ($)",
    "DAYS_BIRTH": "Edad (Años)",
    "AMT_ANNUITY": "Cuota Mensual Total ($)",
    "EXT_SOURCE_2": "Score de Crédito Externo 2",
    "EXT_SOURCE_3": "Score de Crédito Externo 3",
    "AMT_INCOME_TOTAL": "Ingreso Anual Solicitante ($)",
    "AMT_GOODS_PRICE": "Precio del Bien ($)",
    "AMT_REQ_CREDIT_BUREAU_HOUR": "Consultas Buró (Última Hora)",
    "AMT_REQ_CREDIT_BUREAU_DAY": "Consultas Buró (Último Día)",
    "AMT_REQ_CREDIT_BUREAU_WEEK": "Consultas Buró (Última Semana)",
    "AMT_REQ_CREDIT_BUREAU_MON": "Consultas Buró (Último Mes)",
    "AMT_REQ_CREDIT_BUREAU_QRT": "Consultas Buró (Último Trimestre)",
    "AMT_REQ_CREDIT_BUREAU_YEAR": "Consultas Buró (Último Año)",
    "DAYS_EMPLOYED": "Antigüedad Laboral (Años)",
    "DAYS_ID_PUBLISH": "Antigüedad Documento ID (Años)",
    "DAYS_REGISTRATION": "Antigüedad Registro (Años)",
    "DAYS_LAST_PHONE_CHANGE": "Días Último Cambio Teléfono",
    "CNT_CHILDREN": "Número de Hijos",
    "CNT_FAM_MEMBERS": "Miembros Familia",
    "NAME_HOUSING_TYPE_House___apartment": "Vivienda: Casa/Apartamento",
    "NAME_HOUSING_TYPE_Municipal_apartment": "Vivienda: Apartamento Municipal",
    "NAME_HOUSING_TYPE_With_parents": "Vivienda: Con Padres",
    "NAME_HOUSING_TYPE_Rented_apartment": "Vivienda: Apartamento Alquilado",
    "NAME_HOUSING_TYPE_Office_apartment": "Vivienda: Apartamento Oficina",
    "NAME_HOUSING_TYPE_Co-op_apartment": "Vivienda: Apartamento Cooperativa",
    "REG_REGION_NOT_LIVE_REGION": "Región ≠ Región Residencia",
    "REG_REGION_NOT_WORK_REGION": "Región ≠ Región Trabajo",
    "LIVE_REGION_NOT_WORK_REGION": "Residencia ≠ Trabajo",
    "REG_CITY_NOT_LIVE_CITY": "Ciudad ≠ Ciudad Residencia",
    "REG_CITY_NOT_WORK_CITY": "Ciudad ≠ Ciudad Trabajo",
    "LIVE_CITY_NOT_WORK_CITY": "Residencia ≠ Ciudad Trabajo",
    "REGION_POPULATION_RELATIVE": "Densidad Poblacional",
    "REGION_RATING_CLIENT": "Rating Región",
    "REGION_RATING_CLIENT_W_CITY": "Rating Región (con Ciudad)",
    "NAME_EDUCATION_TYPE_Secondary___secondary_special": "Secundaria Especial",
    "NAME_EDUCATION_TYPE_Higher_education": "Educación Superior",
    "NAME_EDUCATION_TYPE_Incomplete_higher": "Superior Incompleta",
    "NAME_EDUCATION_TYPE_Lower_secondary": "Secundaria Baja",
    "NAME_EDUCATION_TYPE_Academic_degree": "Grado Académico",
    "NAME_FAMILY_STATUS_Married": "Casado",
    "NAME_FAMILY_STATUS_Single___not_married": "Soltero",
    "NAME_FAMILY_STATUS_Civil_marriage": "Unión Civil",
    "NAME_FAMILY_STATUS_Separated": "Separado",
    "NAME_FAMILY_STATUS_Widow": "Viudo",
    "NAME_INCOME_TYPE_Working": "Trabajando",
    "NAME_INCOME_TYPE_Commercial_associate": "Asociado Comercial",
    "NAME_INCOME_TYPE_Pensioner": "Pensionado",
    "NAME_INCOME_TYPE_State_servant": "Servidor Público",
    "NAME_INCOME_TYPE_Unemployed": "Desempleado",
    "NAME_INCOME_TYPE_Student": "Estudiante",
    "NAME_INCOME_TYPE_Businessman": "Empresario",
    "NAME_INCOME_TYPE_Maternity_leave": "Licencia Maternidad",
    "NAME_CONTRACT_TYPE_Cash_loans": "Préstamo Efectivo",
    "NAME_CONTRACT_TYPE_Revolving_loans": "Préstamo Revolvente",
    "ORGANIZATION_TYPE_Trade__type_5": "Comercio Tipo 5",
    "ORGANIZATION_TYPE_Trade__type_3": "Comercio Tipo 3",
    "ORGANIZATION_TYPE_Trade__type_4": "Comercio Tipo 4",
    "ORGANIZATION_TYPE_Trade__type_1": "Comercio Tipo 1",
    "ORGANIZATION_TYPE_Trade__type_2": "Comercio Tipo 2",
    "ORGANIZATION_TYPE_Trade__type_6": "Comercio Tipo 6",
    "ORGANIZATION_TYPE_Trade__type_7": "Comercio Tipo 7",
    "ORGANIZATION_TYPE_Telecom": "Telecomunicaciones",
    "ORGANIZATION_TYPE_Industry__type_1": "Industria Tipo 1",
    "ORGANIZATION_TYPE_Industry__type_2": "Industria Tipo 2",
    "ORGANIZATION_TYPE_Industry__type_3": "Industria Tipo 3",
    "ORGANIZATION_TYPE_Industry__type_4": "Industria Tipo 4",
    "ORGANIZATION_TYPE_Industry__type_5": "Industria Tipo 5",
    "ORGANIZATION_TYPE_Industry__type_6": "Industria Tipo 6",
    "ORGANIZATION_TYPE_Industry__type_7": "Industria Tipo 7",
    "ORGANIZATION_TYPE_Industry__type_8": "Industria Tipo 8",
    "ORGANIZATION_TYPE_Industry__type_9": "Industria Tipo 9",
    "ORGANIZATION_TYPE_Industry__type_10": "Industria Tipo 10",
    "ORGANIZATION_TYPE_Industry__type_11": "Industria Tipo 11",
    "ORGANIZATION_TYPE_Industry__type_12": "Industria Tipo 12",
    "ORGANIZATION_TYPE_Industry__type_13": "Industria Tipo 13",
    "ORGANIZATION_TYPE_Transport__type_1": "Transporte Tipo 1",
    "ORGANIZATION_TYPE_Transport__type_2": "Transporte Tipo 2",
    "ORGANIZATION_TYPE_Transport__type_3": "Transporte Tipo 3",
    "ORGANIZATION_TYPE_Transport__type_4": "Transporte Tipo 4",
    "ORGANIZATION_TYPE_Services": "Servicios",
    "ORGANIZATION_TYPE_Construction": "Construcción",
    "ORGANIZATION_TYPE_Realtor": "Bienes Raíces",
    "ORGANIZATION_TYPE_Insurance": "Seguros",
    "ORGANIZATION_TYPE_Self-employed": "Autónomo",
    "ORGANIZATION_TYPE_Business_Entity_Type_1": "Empresa Tipo 1",
    "ORGANIZATION_TYPE_Business_Entity_Type_2": "Empresa Tipo 2",
    "ORGANIZATION_TYPE_Business_Entity_Type_3": "Empresa Tipo 3",
    "ORGANIZATION_TYPE_Education": "Educación",
    "ORGANIZATION_TYPE_Medicine": "Medicina",
    "ORGANIZATION_TYPE_Military": "Militar",
    "ORGANIZATION_TYPE_Police": "Policía",
    "ORGANIZATION_TYPE_Security_Ministries": "Ministerios Seguridad",
    "ORGANIZATION_TYPE_Security": "Seguridad",
    "ORGANIZATION_TYPE_Government": "Gobierno",
    "ORGANIZATION_TYPE_XNA": "No Especificado",
    "ORGANIZATION_TYPE_Other": "Otro",
    "EXT_SOURCE_1": "Score Crédito Externo 1",
    "FLAG_DOCUMENT_2": "Documento Soporte 2",
    "FLAG_DOCUMENT_3": "Documento Soporte 3",
    "FLAG_DOCUMENT_4": "Documento Soporte 4",
    "FLAG_DOCUMENT_5": "Documento Soporte 5",
    "FLAG_DOCUMENT_6": "Documento Soporte 6",
    "FLAG_DOCUMENT_7": "Documento Soporte 7",
    "FLAG_DOCUMENT_8": "Documento Soporte 8",
    "FLAG_DOCUMENT_9": "Documento Soporte 9",
    "FLAG_DOCUMENT_10": "Documento Soporte 10",
    "FLAG_DOCUMENT_11": "Documento Soporte 11",
    "FLAG_DOCUMENT_12": "Documento Soporte 12",
    "FLAG_DOCUMENT_13": "Documento Soporte 13",
    "FLAG_DOCUMENT_14": "Documento Soporte 14",
    "FLAG_DOCUMENT_15": "Documento Soporte 15",
    "FLAG_DOCUMENT_16": "Documento Soporte 16",
    "FLAG_DOCUMENT_17": "Documento Soporte 17",
    "FLAG_DOCUMENT_18": "Documento Soporte 18",
    "FLAG_DOCUMENT_19": "Documento Soporte 19",
    "FLAG_DOCUMENT_20": "Documento Soporte 20",
    "FLAG_DOCUMENT_21": "Documento Soporte 21",
    "OBS_30_CNT_SOCIAL_CIRCLE": "Obs. Círculo Social (30d)",
    "DEF_30_CNT_SOCIAL_CIRCLE": "Default Círculo Social (30d)",
    "OBS_60_CNT_SOCIAL_CIRCLE": "Obs. Círculo Social (60d)",
    "DEF_60_CNT_SOCIAL_CIRCLE": "Default Círculo Social (60d)",
    "FLAG_OWN_CAR": "Posee Automóvil",
    "FLAG_OWN_REALTY": "Posee Propiedad",
    "FLAG_MOBIL": "Teléfono Móvil",
    "FLAG_EMP_PHONE": "Teléfono Laboral",
    "FLAG_WORK_PHONE": "Teléfono Trabajo",
    "FLAG_CONT_MOBILE": "Contacto Móvil",
    "FLAG_PHONE": "Teléfono Fijo",
    "FLAG_EMAIL": "Email",
    "HOUR_APPR_PROCESS_START": "Hora Solicitud",
    "APARTMENTS_AVG": "Apartamentos (Zona)",
    "YEARS_BUILD_AVG": "Años Construcción (Zona)",
    "ELEVATORS_AVG": "Ascensores (Zona)",
    "LIVINGAREA_AVG": "Área Habitable (Zona)",
    "FLOORSMAX_AVG": "Pisos Máximos (Zona)",
    "LANDAREA_AVG": "Área Terreno (Zona)",
}

# =============================================================================
# MODEL HYPERPARAMETERS: LightGBM configuration for credit risk prediction
# =============================================================================
MODEL_RANDOM_STATE: Final[int] = 42
MODEL_N_ESTIMATORS: Final[int] = 100
MODEL_LEARNING_RATE: Final[float] = 0.1
MODEL_NUM_LEAVES: Final[int] = 31

# =============================================================================
# DATA PROCESSING PARAMETERS
# =============================================================================
# Train/test split ratio
TEST_SIZE: Final[float] = 0.2

# Missing value threshold for dropping columns (percentage)
MISSING_VALUE_THRESHOLD: Final[float] = 40.0

# Chunk size for database operations (PostgreSQL parameter limit)
DB_CHUNK_SIZE: Final[int] = 5000

# =============================================================================
# STREAMLIT APP CONFIGURATION
# =============================================================================
# Annual interest rate for loan calculations
ANNUAL_INTEREST_RATE: Final[float] = 0.15

# Risk threshold for high-risk classification
HIGH_RISK_THRESHOLD: Final[float] = 0.3

# Maximum number of features to display in SHAP visualization
MAX_FEATURES_DISPLAY: Final[int] = 20

# Margin multiplier for SHAP plot x-axis scaling
SHAP_PLOT_MARGIN_MULTIPLIER: Final[float] = 1.4

# =============================================================================
# PRESCRIPTIVE SIMULATION (WHAT-IF ANALYSIS) CONFIGURATION
# =============================================================================
# Maximum term in months derived from available terms
# SIMULATION_MAX_TERM_MONTHS: use AVAILABLE_TERMS[-1] instead

# Credit amount reduction factor when extending term fails (80% of original)
SIMULATION_REDUCTION_STEP: Final[float] = 0.80

# Annual interest rate used in simulation calculations
SIMULATION_ANNUAL_RATE: Final[float] = 0.15

# =============================================================================
# BUSINESS POLICY RULES (BANKING STANDARDS)
# =============================================================================
# "Vendedor" threshold: maximum DTI for automatic approval recommendations
POLICY_MAX_DTI: Final[float] = 0.45

# Hard limit: absolute maximum DTI (infranqueable safety belt)
POLICY_HARD_DTI: Final[float] = 0.80

# Maximum Loan-to-Income ratio: credit cannot exceed 5x annual income
POLICY_MAX_LTI: Final[float] = 5.0

# Maximum age at loan end: age + term (years) cannot exceed 85 years
POLICY_MAX_AGE_AT_END: Final[int] = 85

# Available loan terms for sequential simulation
AVAILABLE_TERMS: Final[list[int]] = [12, 24, 36, 48, 60]

# =============================================================================
# FILE PATHS
# =============================================================================
MODEL_FILE_PATH: Final[str] = "models/modelo_riesgo.pkl"
FEATURE_MEDIANS_FILE_PATH: Final[str] = "models/feature_medians.json"
