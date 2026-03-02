import pandas as pd

print("=== Iniciando Limpieza Definitiva del Corpus ===")

# 1. Cargar el corpus bruto
df = pd.read_csv('corpus_youtube_asl.csv')
total_original = len(df)

# 2. Descartar errores de formato (Valores nulos o vacíos)
df_limpio = df.dropna(subset=['text_english', 'gloss_asl']).copy()

# 3. Descartar alucinaciones del modelo (Anomalías de longitud)
df_limpio['palabras_en'] = df_limpio['text_english'].astype(str).apply(lambda x: len(x.split()))
df_limpio['palabras_asl'] = df_limpio['gloss_asl'].astype(str).apply(lambda x: len(x.split()))
df_limpio['ratio_longitud'] = df_limpio['palabras_asl'] / (df_limpio['palabras_en'] + 0.0001)

# Nos quedamos solo con los datos sanos (ratio normal)
df_limpio = df_limpio[(df_limpio['ratio_longitud'] <= 2.0) & (df_limpio['ratio_longitud'] >= 0.5)]

# Limpiar columnas matemáticas temporales
df_limpio = df_limpio.drop(columns=['palabras_en', 'palabras_asl', 'ratio_longitud'])

# 4. Guardar el corpus definitivo
nombre_salida = 'corpus_youtube_asl_FINAL.csv'
df_limpio.to_csv(nombre_salida, index=False, encoding='utf-8')

# Reporte final
total_limpio = len(df_limpio)
descartados = total_original - total_limpio

print(f"-> Se procesaron {total_original} fragmentos.")
print(f"-> Se descartaron {descartados} errores (Nulos + Alucinaciones del T5).")
print(f"✅ Tu corpus definitivo y puro tiene {total_limpio} fragmentos.")
print(f"-> Archivo listo para entrenar guardado como: '{nombre_salida}'")