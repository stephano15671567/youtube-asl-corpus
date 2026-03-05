import cv2
import mediapipe as mp
import numpy as np
import pandas as pd
import os
from tqdm import tqdm

print("=== Iniciando Extracción Masiva de Esqueletos ASL ===")

# Carpeta definitiva para tu dataset de matrices
CARPETA_SALIDA = 'dataset_matrices'
os.makedirs(CARPETA_SALIDA, exist_ok=True)

mp_holistic = mp.solutions.holistic

def extract_keypoints(results):
    pose = np.array([[res.x, res.y, res.z] for res in results.pose_landmarks.landmark]).flatten() if results.pose_landmarks else np.zeros(33*3)
    lh = np.array([[res.x, res.y, res.z] for res in results.left_hand_landmarks.landmark]).flatten() if results.left_hand_landmarks else np.zeros(21*3)
    rh = np.array([[res.x, res.y, res.z] for res in results.right_hand_landmarks.landmark]).flatten() if results.right_hand_landmarks else np.zeros(21*3)
    return np.concatenate([pose, lh, rh])

def time_to_seconds(time_str):
    if isinstance(time_str, (float, int)): return float(time_str)
    h, m, s = str(time_str).split(':')
    return int(h) * 3600 + int(m) * 60 + float(s)

# 1. Cargar el corpus completo
print("-> Cargando corpus completo...")
df = pd.read_csv('corpus_youtube_asl_FINAL.csv')
total_clips = len(df)

# Contadores de métricas
clips_procesados = 0
clips_descartados_presencia = 0
clips_saltados = 0
videos_no_encontrados = 0

# Iniciar MediaPipe
with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
    # Envolver el iterador del DataFrame con tqdm para la barra de progreso
    for index, row in tqdm(df.iterrows(), total=total_clips, desc="Procesando Dataset"):
        video_id = str(row['video_id'])
        start_time = time_to_seconds(row['start_time'])
        end_time = time_to_seconds(row['end_time'])
        
        # Generar nombre del archivo de salida
        nombre_archivo = f"{video_id}_{start_time}.npy"
        ruta_salida = os.path.join(CARPETA_SALIDA, nombre_archivo)
        
        # SISTEMA DE AUTO-REANUDACIÓN: Si ya existe, saltarlo
        if os.path.exists(ruta_salida):
            clips_saltados += 1
            continue
            
        video_path = os.path.join('data', video_id, f"{video_id}.mp4")
        
        if not os.path.exists(video_path):
            videos_no_encontrados += 1
            continue
            
        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        
        start_frame = int(start_time * fps)
        end_frame = int(end_time * fps)
        
        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
        
        frames_data = []
        current_frame = start_frame
        
        while cap.isOpened() and current_frame <= end_frame:
            ret, frame = cap.read()
            if not ret:
                break
            
            image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image.flags.writeable = False                  
            
            results = holistic.process(image)
            keypoints = extract_keypoints(results)
            frames_data.append(keypoints)
            
            current_frame += 1
            
        cap.release()
        
        # FILTRO DE PRESENCIA DE MIGUEL
        if len(frames_data) > 0:
            matriz = np.array(frames_data)
            
            # Revisar las columnas correspondientes a las manos (índices 99 al 224)
            # Si todas esas columnas son cero, no hay manos detectadas en ese frame
            frames_sin_manos = 0
            for frame in matriz:
                manos = frame[99:]
                if np.all(manos == 0):
                    frames_sin_manos += 1
            
            # Si más del 50% de los frames no tienen manos, es basura. Se descarta.
            if frames_sin_manos / len(matriz) > 0.5:
                clips_descartados_presencia += 1
            else:
                # El clip es de calidad, se guarda.
                np.save(ruta_salida, matriz)
                clips_procesados += 1

print("\n=== RESUMEN DE EXTRACCIÓN ===")
print(f"✅ Clips extraídos y guardados: {clips_procesados}")
print(f"⏭️ Clips saltados (ya existían): {clips_saltados}")
print(f"🗑️ Clips descartados (sin manos > 50%): {clips_descartados_presencia}")
print(f"⚠️ Videos no encontrados en 'data/': {videos_no_encontrados}")
print("El dataset está listo en la carpeta 'dataset_matrices'.")