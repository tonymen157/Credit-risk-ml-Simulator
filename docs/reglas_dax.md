# Reglas Estrictas para Power BI y DAX
1. Modelado en Estrella: Asume siempre un diseño con tablas de Hechos (Fact) y Dimensiones (Dim).
2. Medidas vs Columnas: Crea SIEMPRE Medidas (Measures). Las columnas calculadas están estrictamente prohibidas a menos que sea para ejes de gráficos.
3. Optimización VertiPaq: Usa bloques `VAR` y `RETURN` en todas las fórmulas de múltiples pasos para no recalcular variables.
4. Formato: Devuelve todo código DAX con saltos de línea e indentación correcta, omitiendo cualquier código de Python o SQL en la respuesta.