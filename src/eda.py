import logging
import os

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from dotenv import load_dotenv
from sqlalchemy import create_engine, text

# 1. Configuración de Logging para ver el progreso en terminal
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def main():
    try:
        # Cargamos variables de entorno (.env)
        load_dotenv()
        db_url = os.getenv("DATABASE_URL")
        if not db_url:
            raise ValueError("No se encontró DATABASE_URL en el archivo .env")

        engine = create_engine(db_url)

        # --- PASO 1: REPORTE DE VALORES NULOS (Tu brújula de limpieza) ---
        logger.info("Calculando valores nulos por columna (procesando por bloques)...")

        # Obtenemos los nombres de todas las columnas de la tabla
        with engine.connect() as conn:
            query_cols = text(
                "SELECT column_name FROM information_schema.columns WHERE table_name = 'raw_application_train'"
            )
            result = conn.execute(query_cols)
            columns = [row[0] for row in result]

        null_counts = {col: 0 for col in columns}
        total_rows = 0

        # Leemos la tabla en trozos de 20,000 para no saturar la RAM
        for chunk in pd.read_sql_table(
            "raw_application_train", con=engine, chunksize=20000
        ):
            total_rows += len(chunk)
            for col in columns:
                null_counts[col] += chunk[col].isnull().sum()

        # Creamos el ranking de nulos
        null_df = pd.DataFrame(
            {
                "columna": list(null_counts.keys()),
                "porcentaje_nulos": [
                    (v / total_rows) * 100 for v in null_counts.values()
                ],
            }
        ).sort_values("porcentaje_nulos", ascending=False)

        # Imprimimos el reporte en la terminal
        print("\n" + "=" * 40)
        print(" TOP 10 COLUMNAS CON MÁS NULOS ")
        print("=" * 40)
        print(null_df.head(10).to_string(index=False))
        print("=" * 40 + "\n")

        # --- PASO 2: VISUALIZACIONES (Entendiendo el riesgo) ---
        logger.info("Cargando datos optimizados para gráficas...")

        # Solo traemos las columnas necesarias para ahorrar memoria
        df_viz = pd.read_sql_query(
            'SELECT "TARGET", "NAME_EDUCATION_TYPE" FROM raw_application_train', engine
        )

        # CORRECCIÓN TÉCNICA: Aseguramos que TARGET sea un número real (0 y 1)
        df_viz["TARGET"] = pd.to_numeric(df_viz["TARGET"])

        # Gráfico 1: Distribución de la variable objetivo
        logger.info("Generando target_distribution.png...")
        plt.figure(figsize=(8, 5))
        ax = sns.countplot(x="TARGET", data=df_viz, palette="viridis")
        plt.title("Distribución de Clientes: Pagadores (0) vs Deudores (1)")

        # Añadir etiquetas de conteo sobre las barras
        for p in ax.patches:
            ax.annotate(
                f"{int(p.get_height())}",
                (p.get_x() + p.get_width() / 2.0, p.get_height()),
                ha="center",
                va="center",
                xytext=(0, 10),
                textcoords="offset points",
            )

        plt.savefig("target_distribution.png")
        plt.close()

        # Gráfico 2: Tasa de Impago por Educación (CORREGIDO)
        logger.info("Generando education_target_rate.png...")
        plt.figure(figsize=(12, 7))

        # El promedio de una columna 0/1 multiplicado por 100 nos da el % de deudores
        edu_stats = df_viz.groupby("NAME_EDUCATION_TYPE")["TARGET"].mean() * 100
        edu_stats = edu_stats.sort_values(ascending=False)

        sns.barplot(x=edu_stats.index, y=edu_stats.values, palette="magma")
        plt.title("Tasa de Impago (%) según Nivel Educativo", fontsize=14)
        plt.ylabel("Porcentaje de Clientes en Default (%)")
        plt.xlabel("Nivel de Educación")
        plt.xticks(rotation=45, ha="right")

        # Añadir etiquetas de porcentaje sobre las barras
        for i, v in enumerate(edu_stats.values):
            plt.text(i, v + 0.2, f"{v:.2f}%", ha="center", fontweight="bold")

        plt.tight_layout()
        plt.savefig("education_target_rate.png")
        plt.close()

        logger.info("¡PROCESO COMPLETADO! Revisa la terminal y las imágenes generadas.")

    except Exception as e:
        logger.error(f"Error durante el análisis: {e}")
    finally:
        if "engine" in locals():
            engine.dispose()


if __name__ == "__main__":
    main()
