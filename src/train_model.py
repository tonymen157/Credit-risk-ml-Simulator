#!/usr/bin/env python3
"""
Script to train a LightGBM credit risk model on cleaned_application_train.

Applies data purging to remove low-value columns, trains with is_unbalance=True,
evaluates via ROC-AUC, saves the model and recalculates feature medians.
"""

import json
import logging
import os
import re

import joblib
import lightgbm as lgb
import pandas as pd
from dotenv import load_dotenv
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sqlalchemy import create_engine, text

MAX_CARDINALITY = 100
NULL_THRESHOLD = 0.85

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def load_environment() -> str:
    """Load DATABASE_URL from .env file.

    Returns:
        str: Database connection URL.

    Raises:
        OSError: If DATABASE_URL is not set.
    """
    load_dotenv()
    database_url = os.getenv("DATABASE_URL")
    if not database_url:
        raise OSError("DATABASE_URL not found in environment variables")
    logger.info("Environment variables loaded successfully")
    return database_url


def create_db_engine(database_url: str):
    """Create and verify a SQLAlchemy engine.

    Args:
        database_url: Database connection URL.

    Returns:
        Engine: SQLAlchemy engine instance.
    """
    engine = create_engine(database_url)
    with engine.connect() as conn:
        conn.execute(text("SELECT 1"))
    logger.info("Database connection verified")
    return engine


def load_data(engine) -> pd.DataFrame:
    """Load cleaned_application_train from PostgreSQL in chunks.

    Args:
        engine: SQLAlchemy engine.

    Returns:
        pd.DataFrame: Loaded dataset.
    """
    query = text("""
        SELECT column_name, data_type
        FROM information_schema.columns
        WHERE table_name = 'cleaned_application_train'
    """)
    with engine.connect() as conn:
        columns_info = conn.execute(query).fetchall()

    dtype_map: dict[str, str] = {}
    for col in columns_info:
        col_name, data_type = col[0], col[1].lower()
        if "integer" in data_type or "int" in data_type:
            dtype_map[col_name] = "int32"
        elif any(t in data_type for t in ("numeric", "decimal", "real", "double")):
            dtype_map[col_name] = "float32"

    chunk_size = 5000
    chunks: list[pd.DataFrame] = []
    for raw_chunk in pd.read_sql_table(
        "cleaned_application_train", con=engine, chunksize=chunk_size
    ):
        try:
            typed_chunk = raw_chunk.astype(dtype_map)
        except Exception as e:
            logger.warning(f"Type casting warning (ignored): {e}")
            typed_chunk = raw_chunk
        chunks.append(typed_chunk)
        logger.info(f"Chunk loaded and optimized: {len(typed_chunk)} rows")

    df = pd.concat(chunks, ignore_index=True)
    logger.info(f"Data loaded. Shape: {df.shape}")
    return df


def purge_columns(df: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    """Remove low-value columns from the DataFrame.

    Purging rules:
        - ID columns: names containing 'SK_ID', '_ID', or ending in '_ID'.
        - Zero variance: columns with nunique <= 1.
        - High cardinality object columns: object dtype with >100 unique values.
        - High null columns: >85% null values.

    Args:
        df: Input DataFrame.

    Returns:
        Tuple of (purged DataFrame, list of removed column names).
    """
    removed: list[str] = []
    initial_cols = set(df.columns)

    # 1. ID columns (keep TARGET if present)
    id_pattern = re.compile(r"SK_ID|_ID$", flags=re.IGNORECASE)
    id_cols = [c for c in df.columns if id_pattern.search(c) and c != "TARGET"]
    removed.extend(id_cols)
    df = df.drop(columns=id_cols, errors="ignore")
    logger.info(f"Removed {len(id_cols)} ID columns")

    # 2. Zero variance columns
    nunique = df.nunique()
    zero_var = nunique[nunique <= 1].index.tolist()
    zero_var = [c for c in zero_var if c not in removed]
    removed.extend(zero_var)
    df = df.drop(columns=zero_var, errors="ignore")
    logger.info(f"Removed {len(zero_var)} zero-variance columns")

    # 3. High cardinality object columns
    obj_cols = df.select_dtypes(include=["object", "category"]).columns
    high_card = [c for c in obj_cols if df[c].nunique() > MAX_CARDINALITY]
    high_card = [c for c in high_card if c not in removed]
    removed.extend(high_card)
    df = df.drop(columns=high_card, errors="ignore")
    logger.info(f"Removed {len(high_card)} high-cardinality object columns")

    # 4. Columns with >85% nulls
    null_pct = df.isnull().mean()
    high_null = null_pct[null_pct > NULL_THRESHOLD].index.tolist()
    high_null = [c for c in high_null if c not in removed]
    removed.extend(high_null)
    df = df.drop(columns=high_null, errors="ignore")
    logger.info(f"Removed {len(high_null)} columns with >85% nulls")

    final_cols = set(df.columns)
    actually_removed = sorted(initial_cols - final_cols)
    logger.info(
        f"Total columns removed: {len(actually_removed)}. Remaining: {len(final_cols)}"
    )
    return df, actually_removed


def clean_column_names(columns: list[str]) -> list[str]:
    """Sanitize column names for LightGBM compatibility.

    Args:
        columns: Original column names.

    Returns:
        Sanitized column names.
    """
    return [re.sub(r"[^a-zA-Z0-9_]", "_", col) for col in columns]


def split_features_target(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """Separate features and target, clean column names.

    Args:
        df: DataFrame with TARGET column.

    Returns:
        Tuple of (features DataFrame, target Series).

    Raises:
        ValueError: If TARGET column is missing.
    """
    if "TARGET" not in df.columns:
        raise ValueError("Column 'TARGET' not found in DataFrame")

    y = df["TARGET"].astype(int)
    X = df.drop(columns=["TARGET"])
    X.columns = clean_column_names(X.columns.tolist())
    dist = y.value_counts().to_dict()
    logger.info(f"Features: {X.shape[1]} columns. Target distribution: {dist}")
    return X, y


def train_lgbm(X_train: pd.DataFrame, y_train: pd.Series) -> Pipeline:
    """Train a LightGBM classifier in a sklearn Pipeline.

    Args:
        X_train: Training features.
        y_train: Training target.

    Returns:
        Pipeline: Fitted pipeline.
    """
    classifier = lgb.LGBMClassifier(
        objective="binary",
        is_unbalance=True,
        random_state=42,
        n_estimators=100,
        learning_rate=0.1,
        num_leaves=31,
        verbosity=-1,
    )
    pipeline = Pipeline([("classifier", classifier)])
    logger.info("Training LightGBM model...")
    pipeline.fit(X_train, y_train)
    logger.info("Training completed")
    return pipeline


def evaluate(pipeline: Pipeline, X_test: pd.DataFrame, y_test: pd.Series) -> float:
    """Evaluate the model and return ROC-AUC.

    Args:
        pipeline: Fitted pipeline.
        X_test: Test features.
        y_test: Test target.

    Returns:
        float: ROC-AUC score.
    """
    y_pred_proba = pipeline.predict_proba(X_test)[:, 1]
    roc_auc = roc_auc_score(y_test, y_pred_proba)
    logger.info(f"ROC-AUC: {roc_auc:.4f}")
    return roc_auc


def save_model(pipeline: Pipeline, filepath: str) -> None:
    """Persist the trained pipeline to disk.

    Args:
        pipeline: Fitted pipeline.
        filepath: Output path for the pickle file.
    """
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    joblib.dump(pipeline, filepath)
    logger.info(f"Model saved to {filepath}")


def save_feature_medians(X: pd.DataFrame, filepath: str) -> None:
    """Compute and save feature medians as JSON.

    Args:
        X: Feature DataFrame (post-purge, no TARGET).
        filepath: Output path for the JSON file.
    """
    medians = X.median(numeric_only=True)
    medians_dict = medians.to_dict() if isinstance(medians, pd.Series) else {}
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    with open(filepath, "w") as f:
        json.dump(medians_dict, f, indent=2)
    logger.info(f"Feature medians saved to {filepath} ({len(medians_dict)} features)")


def main() -> None:
    """Orchestrate the full training pipeline."""
    engine = None
    try:
        logger.info("=== STARTING MODEL TRAINING ===")

        database_url = load_environment()
        engine = create_db_engine(database_url)
        df = load_data(engine)

        # Purge low-value columns
        df_purged, removed_cols = purge_columns(df)

        # Split features and target
        X, y = split_features_target(df_purged)

        # Train/test split
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        logger.info(f"Train: {X_train.shape}, Test: {X_test.shape}")

        # Train
        pipeline = train_lgbm(X_train, y_train)

        # Evaluate
        roc_auc = evaluate(pipeline, X_test, y_test)

        # Save model
        save_model(pipeline, os.path.join("models", "modelo_riesgo.pkl"))

        # Recalculate feature medians
        save_feature_medians(X, os.path.join("models", "feature_medians.json"))

        # Summary
        print("\n=== TRAINING SUMMARY ===")
        print(f"Columns removed: {len(removed_cols)}")
        print(f"  {removed_cols}")
        print(f"Final feature count: {X.shape[1]}")
        print(f"ROC-AUC: {roc_auc:.4f}")
        print("========================")

        logger.info("=== TRAINING COMPLETED ===")

    except Exception as e:
        logger.error(f"Critical error during training: {e}")
        raise
    finally:
        if engine is not None:
            engine.dispose()
            logger.info("Database connection closed")


if __name__ == "__main__":
    main()
