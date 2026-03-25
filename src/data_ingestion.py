#!/usr/bin/env python3
"""
Script to ingest application_train.csv into PostgreSQL in chunks.
"""

import logging
import os

import pandas as pd
from dotenv import load_dotenv
from sqlalchemy import create_engine
from sqlalchemy.exc import SQLAlchemyError

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def main():
    """Main function to orchestrate the data ingestion process."""
    try:
        # Load environment variables from .env file
        load_dotenv()
        database_url = os.getenv("DATABASE_URL")

        if not database_url:
            raise ValueError("DATABASE_URL environment variable not set")

        logger.info("Creating SQLAlchemy engine...")
        engine = create_engine(database_url)

        # Define CSV file path and chunk size
        csv_path = os.path.join("Dataset", "application_train.csv")
        chunk_size = 10000

        logger.info(f"Reading CSV file in chunks of {chunk_size} rows from {csv_path}")

        # Process CSV in chunks
        for i, chunk in enumerate(
            pd.read_csv(csv_path, chunksize=chunk_size, encoding="latin1")
        ):
            try:
                # Determine if_exists parameter: 'replace' for first chunk, 'append' for others
                if_exists = "replace" if i == 0 else "append"

                logger.info(
                    f"Processing chunk {i + 1} ({len(chunk)} rows) with if_exists='{if_exists}'..."
                )

                # Insert chunk into PostgreSQL table
                chunk.to_sql(
                    name="raw_application_train",
                    con=engine,
                    if_exists=if_exists,
                    index=False,
                    method="multi",  # Use multi-row insert for better performance
                )

                logger.info(
                    f"Successfully inserted chunk {i + 1} into raw_application_train"
                )

            except SQLAlchemyError as e:
                logger.error(f"Database error while processing chunk {i + 1}: {e!s}")
                raise  # Re-raise to stop processing on database error
            except Exception as e:
                logger.error(f"Unexpected error while processing chunk {i + 1}: {e!s}")
                raise

        logger.info("Data ingestion completed successfully")

    except FileNotFoundError:
        logger.error(f"CSV file not found at {csv_path}")
        raise
    except ValueError as e:
        logger.error(f"Configuration error: {e!s}")
        raise
    except SQLAlchemyError as e:
        logger.error(f"Database connection error: {e!s}")
        raise
    except Exception as e:
        logger.error(f"Unexpected error in ingestion process: {e!s}")
        raise


if __name__ == "__main__":
    main()
