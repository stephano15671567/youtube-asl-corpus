import os
import glob
import re
import pysrt
import pandas as pd
import torch
from transformers import T5Tokenizer, T5ForConditionalGeneration
from tqdm import tqdm

# --- 1. CONFIGURACIÓN DE RUTAS ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
RUTA_DATA = os.path.join(BASE_DIR, "data")
MODEL_PATH = os.path.join(BASE_DIR, "mi_modelo_asl_MAESTRO")
ARCHIVO_SALIDA = os.path.join(BASE_DIR, "corpus_youtube_asl.csv")

BATCH_SIZE = 32 # Tamaño de lote para la GPU

print("=== Generador de Corpus Paralelo ASL ===")

# --- 2. CARGAR MODELO T5 ---
print("[1/4] Cargando modelo de traducción T5...")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

tokenizer = T5Tokenizer.from_pretrained("google-t5/t5-small", legacy=False)
model = T5ForConditionalGeneration.from_pretrained(MODEL_PATH).to(device)
model.eval() 

# --- 3. FUNCIONES AUXILIARES ---
def limpiar_texto(texto):
    """Elimina ruido de los subtítulos de YouTube."""
    texto = re.sub(r'<[^>]+>', '', texto) 
    texto = texto.replace('\n', ' ') 
    texto = re.sub(r'\[.*?\]|\(.*?\)|\♪.*?\♪', '', texto) 
    texto = re.sub(r'\s+', ' ', texto) 
    return texto.strip()

def traducir_batch(lista_textos):
    """Pasa un lote de textos por la GPU para mayor velocidad."""
    inputs_str = ["translate English to ASL: " + t for t in lista_textos]
    inputs = tokenizer(inputs_str, return_tensors="pt", padding=True, truncation=True, max_length=128).to(device)
    
    with torch.no_grad():
        outputs = model.generate(**inputs, max_length=128)
        
    return [tokenizer.decode(out, skip_special_tokens=True).upper() for out in outputs]

# --- 4. EXTRACCIÓN DE DATOS ---
print("[2/4] Escaneando subtítulos en 'data/'...")
datos_crudos = []

if not os.path.exists(RUTA_DATA):
    print(f"❌ Error: No se encontró la carpeta {RUTA_DATA}")
    exit()

for folder_name in os.listdir(RUTA_DATA):
    folder_path = os.path.join(RUTA_DATA, folder_name)
    if not os.path.isdir(folder_path):
        continue
    
    archivos_srt = glob.glob(os.path.join(folder_path, "*.srt"))
    
    # NUEVA REGLA: Solo nos importa que exista el SRT
    if not archivos_srt:
        continue
        
    ruta_srt = archivos_srt[0]
    # Asumimos la ruta del video (aunque no esté descargado aún)
    ruta_video_relativa = f"data/{folder_name}/{folder_name}.mp4"
    
    try:
        subs = pysrt.open(ruta_srt, encoding='utf-8')
    except Exception:
        try:
            subs = pysrt.open(ruta_srt, encoding='iso-8859-1') 
        except Exception as e:
            print(f"Saltando {ruta_srt} por error de lectura: {e}")
            continue
            
    for sub in subs:
        texto_limpio = limpiar_texto(sub.text)
        if not texto_limpio or len(texto_limpio) < 2:
            continue 
            
        t_start = f"{sub.start.hours:02d}:{sub.start.minutes:02d}:{sub.start.seconds:02d}.{sub.start.milliseconds:03d}"
        t_end = f"{sub.end.hours:02d}:{sub.end.minutes:02d}:{sub.end.seconds:02d}.{sub.end.milliseconds:03d}"
        
        datos_crudos.append({
            "video_id": folder_name,
            "video_path": ruta_video_relativa,
            "start_time": t_start,
            "end_time": t_end,
            "text_english": texto_limpio
        })

df = pd.DataFrame(datos_crudos)
print(f" -> Se encontraron {len(df)} fragmentos de texto válidos.")

if len(df) == 0:
    print("No hay datos para procesar. Verifica que los .srt tengan contenido.")
    exit()

# --- 5. TRADUCCIÓN MASIVA ---
print(f"[3/4] Generando glosas ASL usando la GPU (Lotes de {BATCH_SIZE})...")
glosas_finales = []
textos = df["text_english"].tolist()

for i in tqdm(range(0, len(textos), BATCH_SIZE), desc="Progreso"):
    batch = textos[i : i + BATCH_SIZE]
    glosas_finales.extend(traducir_batch(batch))

df["gloss_asl"] = glosas_finales

# --- 6. GUARDAR RESULTADOS ---
print(f"[4/4] Guardando corpus...")
df.to_csv(ARCHIVO_SALIDA, index=False, encoding='utf-8')
print(f"\n✅ ¡Trabajo terminado! Corpus guardado en: {ARCHIVO_SALIDA}")