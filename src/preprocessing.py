import logging
import os

import numpy as np
import pandas as pd
from dotenv import load_dotenv
from scipy.sparse import issparse
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sqlalchemy import create_engine, text

from utils.config import DB_CHUNK_SIZE, MISSING_VALUE_THRESHOLD

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def main():
    """
    Main function to perform data cleaning and preprocessing on raw_application_train table.
    Note: This script fits preprocessing steps on the entire dataset. To prevent data leakage
    in downstream modeling, the modeling script should split the data before applying
    any fitting of preprocessing steps. However, for consistency with ML pipelines, we
    encapsulate preprocessing steps in a Pipeline and ColumnTransformer here.
    """
    engine = None
    try:
        # Load environment variables
        load_dotenv()
        database_url = os.getenv("DATABASE_URL")
        if not database_url:
            raise ValueError("DATABASE_URL not found in environment variables")

        logger.info("Creating SQLAlchemy engine")
        engine = create_engine(database_url)

        # Load raw_application_train table
        logger.info("Loading raw_application_train table")
        df = pd.read_sql_table("raw_application_train", con=engine)

        # CORRECCIÓN 1: Forzar que los nombres sean texto plano para Scikit-Learn
        df.columns = [str(col) for col in df.columns]
        logger.info(f"Loaded data with shape: {df.shape}")

        # Identify columns with more than 40% missing values
        missing_percent = df.isnull().sum() / len(df) * 100
        cols_to_drop = missing_percent[
            missing_percent > MISSING_VALUE_THRESHOLD
        ].index.tolist()
        logger.info(f"Dropping columns with >40% missing values: {cols_to_drop}")
        df = df.drop(columns=cols_to_drop)

        # Separate numeric and categorical columns
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        categorical_cols = df.select_dtypes(
            include=["object", "category"]
        ).columns.tolist()

        logger.info(f"Numeric columns: {numeric_cols}")
        logger.info(f"Categorical columns: {categorical_cols}")

        # Define preprocessing steps for numeric and categorical columns
        numeric_transformer = Pipeline(
            steps=[("imputer", SimpleImputer(strategy="median"))]
        )

        categorical_transformer = Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="most_frequent")),
                ("onehot", OneHotEncoder(handle_unknown="ignore")),
            ]
        )

        # Combine transformers in a ColumnTransformer
        preprocessor = ColumnTransformer(
            transformers=[
                ("num", numeric_transformer, numeric_cols),
                ("cat", categorical_transformer, categorical_cols),
            ]
        )

        # Encapsulate the preprocessing in a Pipeline
        preprocessing_pipeline = Pipeline(steps=[("preprocessor", preprocessor)])

        # Fit and transform the data
        logger.info("Fitting and transforming data")
        transformed_data = preprocessing_pipeline.fit_transform(df)

        # Get feature names after one-hot encoding
        cat_encoder = (
            preprocessing_pipeline.named_steps["preprocessor"]
            .named_transformers_["cat"]
            .named_steps["onehot"]
        )
        cat_feature_names = cat_encoder.get_feature_names_out(categorical_cols)

        # Combine feature names
        feature_names = np.concatenate([numeric_cols, cat_feature_names])

        # Convert transformed data to DataFrame
        if issparse(transformed_data):
            transformed_data = transformed_data.toarray()

        df_cleaned = pd.DataFrame(transformed_data, columns=feature_names)
        logger.info(f"Cleaned data shape: {df_cleaned.shape}")

        # Save cleaned data to PostgreSQL
        logger.info("Saving cleaned data to cleaned_application_train table")
        try:
            logger.info(f"DataFrame shape before saving: {df_cleaned.shape}")

            # Drop table if exists to avoid conflicts
            with engine.connect() as conn:
                trans = conn.begin()
                try:
                    conn.execute(text("DROP TABLE IF EXISTS cleaned_application_train"))
                    trans.commit()
                    logger.info("Dropped existing table if it existed")
                except Exception as e:
                    trans.rollback()
                    logger.warning(f"Could not drop table (might not exist): {e}")

            # Save the DataFrame to SQL in chunks
            logger.info("Attempting to save DataFrame to SQL in chunks...")

            # CORRECCIÓN 2: Bajamos el chunksize y quitamos method="multi"
            # para evitar el error de "máximo de 32767 parámetros" de PostgreSQL
            chunksize = DB_CHUNK_SIZE
            for start in range(0, len(df_cleaned), chunksize):
                end = start + chunksize
                chunk = df_cleaned.iloc[start:end]
                if_exists = "fail" if start == 0 else "append"
                chunk.to_sql(
                    "cleaned_application_train",
                    con=engine,
                    if_exists=if_exists,
                    index=False,
                    # method="multi" <- ESTO FUE ELIMINADO PARA SALVAR TU RAM
                )
                logger.info(
                    f"Saved rows {start} to {min(end, len(df_cleaned))} of {len(df_cleaned)}"
                )

            logger.info("Data cleaning and preprocessing completed successfully")

            # Verify the table was created
            with engine.connect() as conn:
                result = conn.execute(
                    text("SELECT COUNT(*) FROM cleaned_application_train")
                )
                count = result.scalar()
                logger.info(
                    f"Verified table creation: {count} rows in cleaned_application_train"
                )

        except Exception as e:
            logger.error(f"Error saving data to database: {e!s}")
            raise

    except Exception as e:
        logger.error(f"An unexpected error occurred: {e!s}", exc_info=True)
        raise
    finally:
        if engine is not None:
            engine.dispose()
            logger.info("Database connection closed.")


if __name__ == "__main__":
    main()
