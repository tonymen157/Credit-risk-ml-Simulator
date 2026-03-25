#!/usr/bin/env python3
"""
Script to compute median values for features in cleaned_application_train table
and save them as a JSON file.
"""

import json
import logging
import os
import sys

import pandas as pd
from dotenv import load_dotenv
from sqlalchemy import create_engine
from sqlalchemy.exc import SQLAlchemyError

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def main() -> None:
    """Main function to compute feature medians and save to JSON."""
    try:
        # Load environment variables from .env file
        load_dotenv()
        database_url = os.getenv("DATABASE_URL")

        if not database_url:
            raise ValueError("DATABASE_URL environment variable not set")

        logger.info("Creating SQLAlchemy engine...")
        engine = create_engine(database_url)

        logger.info("Reading cleaned_application_train table from PostgreSQL...")
        # Read the entire table into a DataFrame
        # Note: For very large tables, consider using chunking with read_sql_query
        df = pd.read_sql_table("cleaned_application_train", con=engine)

        # Exclude the TARGET column
        if "TARGET" in df.columns:
            df_features = df.drop(columns=["TARGET"])
            logger.info("Excluded TARGET column")
        else:
            logger.warning("TARGET column not found, using all columns")
            df_features = df

        # Compute median for each column (skip NaNs by default)
        logger.info("Computing median for each feature...")
        medians = df_features.median(numeric_only=True)

        # Convert to dictionary (handle both Series and scalar cases)
        if isinstance(medians, pd.Series):
            medians_dict = medians.to_dict()
        # Single column case: medians is a scalar
        elif len(df_features.columns) == 1:
            # Get the column name and ensure it's hashable (string)
            column_name = str(df_features.columns[0])
            medians_dict = {column_name: medians}
        else:
            # This shouldn't happen with DataFrame.median(), but handle gracefully
            medians_dict = {}

        # Ensure models directory exists
        models_dir = "models"
        os.makedirs(models_dir, exist_ok=True)

        # Save to JSON file
        output_path = os.path.join(models_dir, "feature_medians.json")
        with open(output_path, "w") as f:
            json.dump(medians_dict, f, indent=2)

        logger.info(f"Successfully saved feature medians to {output_path}")
        print(f"Success: Feature medians saved to {output_path}")

    except ValueError as e:
        logger.error(f"Configuration error: {e}")
        sys.exit(1)
    except SQLAlchemyError as e:
        logger.error(f"Database error: {e}")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
