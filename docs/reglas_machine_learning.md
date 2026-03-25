# Reglas Estrictas para Machine Learning e Ingeniería de Datos

## 1. Modelado y Scikit-Learn
1. **Prevenir Fugas de Datos (Data Leakage):** Toda imputación de nulos, codificación o escalado debe encapsularse. El método `fit` DEBE aplicarse estrictamente solo sobre `X_train`, y `transform` sobre `X_train` y `X_test`.
2. **Pipeline Obligatorio:** Usa siempre `sklearn.pipeline.Pipeline` (o `imblearn.pipeline.Pipeline` si usas SMOTE) para unificar el preprocesamiento y el estimador final. No se aceptan transformaciones sueltas.
3. **Evaluación Completa:** La métrica `accuracy` es engañosa y está prohibida como métrica única. Imprime siempre el Reporte de Clasificación (`classification_report`), la Matriz de Confusión y el área bajo la curva ROC (`roc_auc_score`).
4. **Manejo de Clases Desbalanceadas:** La variable `TARGET` tiene un desbalanceo severo (aprox. 8% positivos). Implementa técnicas como SMOTE, undersampling, o utiliza obligatoriamente hiperparámetros nativos como `class_weight="balanced"` o `scale_pos_weight`.

## 2. Rendimiento y Procesamiento Masivo (Data Engineering)
5. **Cero Bucles Manuales:** Prohibido usar ciclos `for` para iterar sobre filas de DataFrames. Utiliza exclusivamente operaciones vectorizadas de Pandas o NumPy.
6. **Eficiencia de Memoria (Chunking):** Si el dataset supera las 100,000 filas, asume restricciones críticas de RAM. Al leer o escribir en bases de datos, usa siempre el parámetro `chunksize` adaptado al ancho de la tabla (ej. `chunksize=5000` para tablas con >150 columnas).
7. **Límites de Base de Datos (PostgreSQL):** Al usar `to_sql`, NUNCA uses `method="multi"` si el número de columnas multiplicadas por el chunksize supera los 32,767 marcadores de posición, para evitar colapsar la base de datos.
8. **Código "Pythonic" y Limpio:** Escribe código elegante, conciso y de nivel Senior. Evita la verbosidad excesiva, no hagas copias innecesarias de DataFrames (`inplace=True` cuando aplique) y elimina variables pesadas de la memoria usando `del` o `gc.collect()` si es estrictamente necesario.