#!/usr/bin/env python
"""
Script to extract feature importance from a trained LightGBM model pipeline
and generate a horizontal bar chart of the top 20 most important features.
"""

import logging
import os

import joblib
import matplotlib.pyplot as plt
import pandas as pd
from dotenv import load_dotenv
from sqlalchemy import create_engine, text

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def main():
    """Main function to execute the feature importance extraction and plotting."""
    try:
        # Load environment variables
        load_dotenv()
        database_url = os.getenv("DATABASE_URL")
        if not database_url:
            raise ValueError("DATABASE_URL environment variable not set")

        logger.info("Creating SQLAlchemy engine")
        engine = create_engine(database_url)

        # Get column names from cleaned_application_train table (excluding TARGET)
        logger.info("Fetching column names from cleaned_application_train table")
        with engine.connect() as conn:
            # Query to get column names excluding TARGET
            query = text("""
                SELECT column_name 
                FROM information_schema.columns 
                WHERE table_name = 'cleaned_application_train' 
                AND column_name != 'TARGET'
                ORDER BY ordinal_position
            """)
            result = conn.execute(query)
            feature_names = [row[0] for row in result]

        if not feature_names:
            raise ValueError("No feature names retrieved from the table")

        logger.info(f"Retrieved {len(feature_names)} feature names")

        # Load the trained model pipeline
        model_path = "../models/modelo_riesgo.pkl"
        logger.info(f"Loading model from {model_path}")
        model_pipeline = joblib.load(model_path)

        # Extract the final estimator (assuming it's the last step in the pipeline)
        # We assume the pipeline has a step named 'classifier' or we take the last step
        # For safety, we try to get the last step
        if hasattr(model_pipeline, "named_steps"):
            # If it's a Pipeline with named steps, we assume the classifier is the last step
            # We can also look for a step that has feature_importances_ attribute
            classifier = None
            for name, step in model_pipeline.named_steps.items():
                if hasattr(step, "feature_importances_"):
                    classifier = step
                    break
            if classifier is None:
                # Fallback: take the last step
                classifier = model_pipeline.steps[-1][1]
        else:
            # If it's not a Pipeline, assume it's the classifier directly
            classifier = model_pipeline

        if not hasattr(classifier, "feature_importances_"):
            raise AttributeError(
                "The classifier does not have feature_importances_ attribute"
            )

        logger.info("Extracting feature importances from the classifier")
        importances = classifier.feature_importances_

        # Create DataFrame with feature names and importances
        feature_importance_df = pd.DataFrame(
            {"feature": feature_names, "importance": importances}
        )

        # Sort by importance descending and select top 20
        top_features = feature_importance_df.sort_values(
            by="importance", ascending=False
        ).head(20)

        logger.info("Top 20 features:")
        logger.info(top_features.to_string())

        # Generate horizontal bar chart
        plt.figure(figsize=(10, 8))
        plt.barh(
            y=top_features["feature"],
            width=top_features["importance"],
            color="steelblue",
        )
        plt.xlabel("Feature Importance")
        plt.title("Top 20 Feature Importances from LightGBM Model")
        plt.gca().invert_yaxis()  # To have the highest importance at the top
        plt.tight_layout()

        # Save the chart
        output_path = "feature_importance.png"
        logger.info(f"Saving feature importance chart to {output_path}")
        plt.savefig(output_path, dpi=300)
        plt.close()

        logger.info("Feature importance chart generated successfully")

    except Exception as e:
        logger.error(f"An error occurred: {e}", exc_info=True)
        raise


if __name__ == "__main__":
    main()
