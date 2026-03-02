import pandas as pd

print("=== Iniciando Auditoría del Corpus FINAL ===")

try:
    # 1. Cargar el corpus limpio
    df = pd.read_csv('corpus_youtube_asl_FINAL.csv')
    total_fragmentos = len(df)
    print(f"Corpus cargado exitosamente. Total de fragmentos a analizar: {total_fragmentos}\n")

    # Contadores de errores
    errores_nulos = 0
    anomalias_longitud = 0

    # 2. Análisis de datos nulos (El T5 o los subtítulos fallaron)
    filas_nulas = df[df['text_english'].isnull() | df['gloss_asl'].isnull()]
    errores_nulos = len(filas_nulas)

    # 3. Análisis de anomalías de longitud (Alucinaciones)
    if total_fragmentos > 0:
        # Calcular palabras
        df_temp = df.copy()
        df_temp['palabras_en'] = df_temp['text_english'].astype(str).apply(lambda x: len(x.split()))
        df_temp['palabras_asl'] = df_temp['gloss_asl'].astype(str).apply(lambda x: len(x.split()))
        
        # Calcular ratio (Inglés vs Glosa)
        df_temp['ratio_longitud'] = df_temp['palabras_asl'] / (df_temp['palabras_en'] + 0.0001)

        # Buscar lo que está fuera del rango normal (menos de la mitad o más del doble)
        anomalias = df_temp[(df_temp['ratio_longitud'] > 2.0) | (df_temp['ratio_longitud'] < 0.5)]
        anomalias_longitud = len(anomalias)

    # 4. Cálculo de calidad final
    total_errores = errores_nulos + anomalias_longitud
    porcentaje_limpio = ((total_fragmentos - total_errores) / total_fragmentos) * 100 if total_fragmentos > 0 else 0

    # 5. Reporte en consola
    print("--- MÉTRICAS DE CALIDAD POST-LIMPIEZA ---")
    print(f"1. Fragmentos con datos incompletos/nulos: {errores_nulos} (Debería ser 0)")
    print(f"2. Fragmentos con anomalías de longitud: {anomalias_longitud} (Debería ser 0)")
    print("\n--- RESUMEN GENERAL ---")
    print(f"Porcentaje de pureza del corpus: {porcentaje_limpio:.2f}%")
    
    if porcentaje_limpio == 100.0:
        print("✅ ¡AUDITORÍA SUPERADA! El corpus está 100% libre de los errores conocidos. Listo para SlowFast.")
    else:
        print("⚠️ Advertencia: Aún quedan errores residuales.")

except FileNotFoundError:
    print("Error: No se encontró el archivo 'corpus_youtube_asl_FINAL.csv'. Asegúrate de haber corrido el script de limpieza antes.")
except Exception as e:
    print(f"Ocurrió un error inesperado: {e}")