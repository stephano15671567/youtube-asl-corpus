import cv2
import numpy as np
import time

print("=== Reproductor de Esqueletos ASL ===")

# Cargar la primera matriz que generaste
# Asegúrate de que el nombre del archivo coincida exactamente con el de tu consola
ruta_matriz = r'matrices_piloto\--6bmFM9wT4_6.32.npy'
data = np.load(ruta_matriz)

# Crear una ventana de visualización
cv2.namedWindow('Sanity Check - Esqueleto ASL', cv2.WINDOW_NORMAL)
cv2.resizeWindow('Sanity Check - Esqueleto ASL', 800, 800)

print(f"-> Reproduciendo matriz de forma: {data.shape}")
print("-> Presiona la tecla 'q' en la ventana para salir.")

# Bucle infinito para que el video se repita hasta que lo cierres
while True:
    for frame in data:
        # Crear un lienzo negro de 800x800 pixeles
        img = np.zeros((800, 800, 3), dtype=np.uint8)

        # Reagrupar los 225 valores en 75 puntos de 3 coordenadas (X, Y, Z)
        puntos = frame.reshape(75, 3)

        for i, punto in enumerate(puntos):
            x, y, z = punto
            # Si x e y son exactamente 0, significa que la mano salió de la cámara
            if x == 0 and y == 0:
                continue
            
            # Convertir coordenadas normalizadas (0.0 a 1.0) a pixeles de la pantalla (0 a 800)
            px = int(x * 800)
            py = int(y * 800)
            
            # Pintar los puntos: Cuerpo (Blanco), Mano Izq (Verde), Mano Der (Rojo)
            color = (255, 255, 255) 
            if 33 <= i < 54:
                color = (0, 255, 0) 
            elif i >= 54:
                color = (0, 0, 255) 

            # Dibujar el punto en la pantalla
            cv2.circle(img, (px, py), 4, color, -1)

        cv2.imshow('Sanity Check - Esqueleto ASL', img)
        
        # Pausa de 30 milisegundos para simular la velocidad real del video
        if cv2.waitKey(30) & 0xFF == ord('q'):
            cv2.destroyAllWindows()
            exit()
            
    time.sleep(0.5) # Pequeña pausa antes de repetir el clip