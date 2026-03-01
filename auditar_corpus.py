import pandas as pd
import re

print("=== Iniciando Auditoría del Corpus ASL ===")

# 1. Cargar el dataset
try:
    df = pd.read_csv('corpus_youtube_asl.csv')
    total_filas = len(df)
    print(f"Corpus cargado exitosamente. Total de fragmentos: {total_filas}\n")
except Exception as e:
    print(f"Error al cargar el archivo: {e}")
    exit()

# 2. Detección de errores de codificación (Patrones comunes de UTF-8 mal interpretado)
errores_codificacion = df[
    df['text_english'].str.contains('â€|Ã|ï', na=False, regex=True) | 
    df['gloss_asl'].str.contains('â€|Ã|ï', na=False, regex=True)
]

# 3. Detección de valores nulos o celdas vacías
nulos = df[df.isnull().any(axis=1)]

# 4. Detección de anomalías por longitud (Posibles alucinaciones del modelo T5)
# Contamos palabras en inglés y en glosa
df['palabras_en'] = df['text_english'].astype(str).apply(lambda x: len(x.split()))
df['palabras_asl'] = df['gloss_asl'].astype(str).apply(lambda x: len(x.split()))

# Calculamos el ratio. Si la glosa es el doble de larga o la mitad de corta, es sospechosa.
df['ratio_longitud'] = df['palabras_asl'] / (df['palabras_en'] + 0.0001) # Evitamos división por cero
anomalias_longitud = df[(df['ratio_longitud'] > 2.0) | (df['ratio_longitud'] < 0.5)]

# 5. Reporte de Métricas
print("--- MÉTRICAS DE CALIDAD ---")
print(f"1. Fragmentos con errores de codificación: {len(errores_codificacion)} ({(len(errores_codificacion)/total_filas)*100:.2f}%)")
print(f"2. Fragmentos con datos incompletos/nulos: {len(nulos)} ({(len(nulos)/total_filas)*100:.2f}%)")
print(f"3. Fragmentos con anomalías de longitud: {len(anomalias_longitud)} ({(len(anomalias_longitud)/total_filas)*100:.2f}%)")

total_errores = len(set(errores_codificacion.index.tolist() + nulos.index.tolist() + anomalias_longitud.index.tolist()))
tasa_exito = ((total_filas - total_errores) / total_filas) * 100

print("\n--- RESUMEN GENERAL ---")
print(f"Porcentaje estimado de datos LIMPIOS: {tasa_exito:.2f}%")

# Opcional: Guardar los errores en un CSV nuevo para revisarlos a mano
if total_errores > 0:
    df_errores = df.loc[list(set(errores_codificacion.index.tolist() + nulos.index.tolist() + anomalias_longitud.index.tolist()))]
    df_errores.to_csv('fragmentos_sospechosos.csv', index=False)
    print("-> Se ha guardado un archivo 'fragmentos_sospechosos.csv' con las filas problemáticas para que las revises.")