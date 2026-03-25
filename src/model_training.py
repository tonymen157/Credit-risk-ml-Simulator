import logging
import os
import re

import joblib
import lightgbm as lgb
import pandas as pd
from dotenv import load_dotenv
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sqlalchemy import create_engine, text

from utils.config import (
    DB_CHUNK_SIZE,
    MODEL_LEARNING_RATE,
    MODEL_N_ESTIMATORS,
    MODEL_NUM_LEAVES,
    MODEL_RANDOM_STATE,
    TEST_SIZE,
)

# Configuración de logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def load_environment() -> str:
    """
    Carga variables de entorno y retorna la URL de la base de datos.
    Returns:
        str: DATABASE_URL
    """
    load_dotenv()
    database_url = os.getenv("DATABASE_URL")
    if not database_url:
        raise OSError("DATABASE_URL no encontrada en variables de entorno")
    logger.info("Variables de entorno cargadas correctamente")
    return database_url


def create_db_engine(database_url: str):
    """
    Crea y retorna un motor SQLAlchemy.
    Args:
        database_url (str): URL de conexión a la base de datos
    Returns:
        Engine: Motor SQLAlchemy
    """
    try:
        engine = create_engine(database_url)
        # Prueba de conexión
        with engine.connect() as conn:
            conn.execute(text("SELECT 1"))
        logger.info("Motor SQLAlchemy creado y conexión verificada")
        return engine
    except Exception as e:
        logger.error(f"Error al crear el motor de base de datos: {e}")
        raise


def load_data_from_postgres(engine) -> pd.DataFrame:
    """
    Carga la tabla cleaned_application_train desde PostgreSQL en un DataFrame.
    Optimiza la memoria convirtiendo columnas numéricas a tipos apropiados.
    Args:
        engine: Motor SQLAlchemy
    Returns:
        pd.DataFrame: DataFrame con los datos limpios
    """
    try:
        # Primero, obtener la estructura de la tabla para optimizar tipos
        query = text("""
            SELECT column_name, data_type 
            FROM information_schema.columns 
            WHERE table_name = 'cleaned_application_train'
        """)
        with engine.connect() as conn:
            result = conn.execute(query)
            columns_info = result.fetchall()

        # Construir un diccionario de tipos para la conversión
        dtype_map = {}
        for col in columns_info:
            col_name = col[0]
            data_type = col[1].lower()
            if "integer" in data_type or "int" in data_type:
                dtype_map[col_name] = "int32"
            elif (
                "numeric" in data_type
                or "decimal" in data_type
                or "real" in data_type
                or "double" in data_type
            ):
                dtype_map[col_name] = "float32"
            else:
                # Para otros tipos (varchar, text, etc.) dejamos el tipo por defecto de pandas
                pass

        # Cargar los datos en chunks para optimizar memoria
        chunk_size = DB_CHUNK_SIZE  # Ajustado según la regla de chunking
        chunks = []

        # ELIMINAMOS el parámetro dtype de aquí adentro
        for chunk in pd.read_sql_table(
            "cleaned_application_train",
            con=engine,
            chunksize=chunk_size,
        ):
            # APLICAMOS la reducción de memoria inmediatamente después de leer el chunk
            try:
                chunk = chunk.astype(dtype_map)
            except Exception as e:
                logger.warning(f"Advertencia de casteo (ignorada): {e}")

            chunks.append(chunk)
            logger.info(f"Chunk leído y optimizado con {len(chunk)} filas")

        df = pd.concat(chunks, ignore_index=True)
        logger.info(f"Datos cargados exitosamente. Shape: {df.shape}")
        return df

    except Exception as e:
        logger.error(f"Error al cargar datos de PostgreSQL: {e}")
        raise


def prepare_features_target(df: pd.DataFrame):
    """
    Separa características (X) y objetivo (y).
    Asume que la columna objetivo se llama 'TARGET'.
    Limpia los nombres de las columnas para evitar errores en LightGBM.
    """
    if "TARGET" not in df.columns:
        raise ValueError("Columna 'TARGET' no encontrada en el DataFrame")

    y = df["TARGET"]
    X = df.drop("TARGET", axis=1)

    # --- CORRECCIÓN PARA LIGHTGBM ---
    # Reemplaza cualquier carácter que NO sea letra, número o guion bajo por un '_'
    X.columns = [re.sub(r"[^a-zA-Z0-9_]", "_", col) for col in X.columns]

    logger.info(f"Características: {X.shape[1]} columnas")
    logger.info(f"Distribución de TARGET: {y.value_counts().to_dict()}")
    return X, y


def train_model(X_train, y_train):
    """
    Entrena un modelo LightGBM dentro de un Pipeline.
    Maneja el desbalance de clases usando is_unbalance=True.
    Args:
        X_train: Características de entrenamiento
        y_train: Objetivo de entrenamiento
    Returns:
        Pipeline: Pipeline entrenado con el modelo LightGBM
    """
    try:
        # Crear el clasificador LightGBM
        lgb_classifier = lgb.LGBMClassifier(
            objective="binary",
            is_unbalance=True,  # Manejo de desbalance de clases
            random_state=MODEL_RANDOM_STATE,
            n_estimators=MODEL_N_ESTIMATORS,
            learning_rate=MODEL_LEARNING_RATE,
            num_leaves=MODEL_NUM_LEAVES,
            verbosity=-1,
        )

        # Crear pipeline (aunque no haya preprocesamiento, es obligatorio según reglas)
        pipeline = Pipeline([("classifier", lgb_classifier)])

        # Entrenar el modelo
        logger.info("Iniciando entrenamiento del modelo LightGBM...")
        pipeline.fit(X_train, y_train)
        logger.info("Entrenamiento completado")
        return pipeline

    except Exception as e:
        logger.error(f"Error durante el entrenamiento: {e}")
        raise


def evaluate_model(pipeline, X_test, y_test):
    """
    Evalúa el modelo en el conjunto de prueba.
    Calcula ROC-AUC, reporte de clasificación y matriz de confusión.
    Args:
        pipeline: Pipeline entrenado
        X_test: Características de prueba
        y_test: Objetivo de prueba
    """
    try:
        # Predecir probabilidades y clases
        y_pred_proba = pipeline.predict_proba(X_test)[:, 1]
        y_pred = pipeline.predict(X_test)

        # Calcular métricas
        roc_auc = roc_auc_score(y_test, y_pred_proba)
        report = classification_report(y_test, y_pred)
        cm = confusion_matrix(y_test, y_pred)

        # Imprimir resultados
        logger.info(f"ROC-AUC: {roc_auc:.4f}")
        logger.info("\nReporte de Clasificación:")
        logger.info(f"\n{report}")
        logger.info("\nMatriz de Confusión:")
        logger.info(f"\n{cm}")

        return roc_auc, report, cm

    except Exception as e:
        logger.error(f"Error durante la evaluación: {e}")
        raise


def save_model(pipeline, filepath: str = "modelo_riesgo.pkl"):
    """
    Guarda el pipeline entrenado en disco usando joblib.
    Args:
        pipeline: Pipeline entrenado
        filepath (str): Ruta donde guardar el modelo
    """
    try:
        joblib.dump(pipeline, filepath)
        logger.info(f"Modelo guardado exitosamente en {filepath}")
    except Exception as e:
        logger.error(f"Error al guardar el modelo: {e}")
        raise


def main():
    """Función principal que orquesta el proceso de entrenamiento."""
    try:
        logger.info("=== INICIANDO ENTRENAMIENTO DEL MODELO ===")

        # 1. Cargar variables de entorno
        database_url = load_environment()

        # 2. Crear motor de base de datos
        engine = create_db_engine(database_url)

        # 3. Cargar datos desde PostgreSQL
        df = load_data_from_postgres(engine)

        # 4. Preparar características y objetivo
        X, y = prepare_features_target(df)

        # 5. Dividir en train y test (80/20) con estratificación
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=TEST_SIZE, random_state=MODEL_RANDOM_STATE, stratify=y
        )
        logger.info(f"Datos divididos: Train={X_train.shape}, Test={X_test.shape}")

        # 6. Entrenar modelo
        pipeline = train_model(X_train, y_train)

        # 7. Evaluar modelo
        evaluate_model(pipeline, X_test, y_test)

        # 8. Guardar modelo
        save_model(pipeline, "../models/modelo_riesgo.pkl")

        logger.info("=== ENTRENAMIENTO COMPLETADO ===")

    except Exception as e:
        logger.error(f"Error crítico en el proceso de entrenamiento: {e}")
        raise
    finally:
        # Cerrar conexiones si es necesario
        if "engine" in locals():
            engine.dispose()
            logger.info("Conexiones a la base de datos cerradas")


if __name__ == "__main__":
    main()
