# Reglas Estrictas para Análisis de Datos (EDA)
1. **Eficiencia de Memoria:** Al cargar DataFrames grandes con pandas, usar siempre `df.info(memory_usage='deep')` y optimizar los tipos de datos (ej. pasar de `float64` a `float32` o usar `category` para textos repetitivos).
2. **Revisión de Nulos Exhaustiva:** Generar siempre una tabla o gráfico que muestre el porcentaje absoluto y relativo de valores nulos por columna.
3. **Distribución del Target:** Si es un problema de clasificación, el primer paso visual debe ser siempre mostrar el balance de la variable objetivo (Target).
4. **Visualizaciones Limpias:** Usar `seaborn` o `matplotlib`. Los gráficos deben tener título, etiquetas claras en los ejes X e Y, y no estar superpuestos.