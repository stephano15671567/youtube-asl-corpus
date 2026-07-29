# README Técnico y Manual de Adaptación
## Pipeline de Procesamiento CSLR y Migración a Lengua de Señas Chilena (LSCh)

---

## 1. Descripción General del Sistema

Este repositorio contiene el ecosistema completo de scripts diseñado para automatizar la ingeniería de datos en sistemas de Reconocimiento Continuo de Lenguaje de Señas (CSLR). El pipeline está construido bajo un enfoque desacoplado y modular, lo que permite ejecutar cada fase de manera independiente y persistir la información mediante archivos intermedios estandarizados y bases de datos relacionales compactas.

El sistema original fue desarrollado y calibrado para procesar corpus masivos en inglés hacia el Lenguaje de Señas Americano (ASL), pero su arquitectura paramétrica permite una migración directa hacia cualquier otro par lingüístico visogestual, tal como la **Lengua de Señas Chilena (LSCh)**.

---

## 2. Guía de Despliegue y Configuración del Entorno

Para que un nuevo desarrollador o investigador pueda clonar el repositorio y levantar el entorno de ejecución técnico, debe seguir los pasos que se detallan a continuación:

### Paso 2.1: Clonación del Repositorio e Inicialización
Abra una terminal en su sistema operativo y ejecute los siguientes comandos para descargar el código fuente y crear el entorno virtual aislado:

```bash
# Clonar el repositorio oficial
git clone [https://github.com/stephano15671567/youtube-asl-corpus.git](https://github.com/stephano15671567/youtube-asl-corpus.git)
cd youtube-asl-corpus

# Crear un entorno virtual de Python (se recomienda Python 3.10 o superior)
python -m venv venv

# Activar el entorno virtual en sistemas Linux/macOS
source venv/bin/activate

# Activar el entorno virtual en sistemas Windows (PowerShell)
.\venv\Scripts\Activate.ps1

Paso 2.2: Instalación de Dependencias
Instale todas las librerías numéricas, de visión computacional y procesamiento del lenguaje natural requeridas por el software:

pip install -r requirements.txt

Dependencias de Sistema Externas (Nota crítica de software):
El sistema invoca binarios de FFmpeg de forma nativa para realizar recortes de video de alta velocidad con precisión de milisegundos. Asegúrese de que ffmpeg esté instalado en el sistema operativo y debidamente registrado en las variables de entorno (PATH).

Para acelerar la extracción de keypoints anatómicos y la inferencia de los modelos Transformers, se recomienda disponer de una tarjeta gráfica NVIDIA con los controladores y el toolkit CUDA correctamente configurados.

3. Arquitectura del Flujo de Datos
El procesamiento de datos avanza de forma lineal a través de etapas secuenciales estrictas:

Descarga Cruda: La adquisición descarga de forma masiva los contenidos multimedia basándose en una lista de identificadores únicos.

Glosado Sintáctico: Los subtítulos temporizados son limpiados y alimentados a una red neuronal profunda para mapear palabras a conceptos glosados.

Control de Calidad Textual: Se aplican filtros estadísticos sobre las longitudes de los textos para eliminar alucinaciones del modelo de traducción.

Extracción Visuoperceptual: Los fragmentos de video válidos se segmentan y procesan con visión artificial para extraer coordenadas cartesianas tridimensionales de las manos y articulaciones, descartando clips con baja presencia gestual.

4. Inventario Técnico de Componentes y Análisis de Código
A continuación, se detalla el propósito operativo y la lógica interna de cada uno de los archivos que componen el sistema:

4.1. descargar_youtube_asl.py (Adquisición Multihilo)
Este script es el encargado de descargar los videos y sus respectivos archivos de subtítulos cronometrados (.srt).

Mecanismo de Descarga: Utiliza la librería yt-dlp para conectarse de manera eficiente a los servidores de video.

Evitación de Bloqueos: Implementa una rotación dinámica de firmas de cliente (web, ios, android, tv, mweb) para mitigar bloqueos por peticiones excesivas (throttling).

Persistencia y Tolerancia a Fallos: Persiste el progreso en una base de datos relacional compacta estado_descargas.sqlite. Cada ID pasa por estados estrictos (pending, downloading, done, failed). Si el proceso se interrumpe, se reanuda de manera automática al reiniciar el script sin duplicar descargas.

Filtro Temático: Utiliza una expresión regular estricta (RE_srt_EN) que valida y extrae únicamente subtítulos oficiales o automáticos en formato SRT inglés, descartando metadatos corruptos como los archivos de Live Chat.

4.2. generar_corpus_asl.py (Módulo de Glosado Automático)
Transforma las líneas de texto natural de los subtítulos en secuencias conceptuales de glosas indexables.

Carga del Modelo: Carga en memoria de video (CUDA) una arquitectura Transformer ajustada localmente desde la carpeta mi_modelo_asl_MAESTRO.

Limpieza de Subtítulos: Emplea expresiones regulares para limpiar ruido textual propio de YouTube, removiendo etiquetas HTML, saltos de línea \n, indicaciones musicales (♪) y marcas ambientales como [Risas].

Procesamiento por Lotes: Agrupa las oraciones depuradas en lotes vectorizados (BATCH_SIZE = 32) para maximizar la concurrencia en los núcleos de la GPU, reduciendo drásticamente el tiempo de traducción masiva.

Salida: Escribe un archivo estructurado llamado corpus_youtube_asl.csv que vincula permanentemente el ID del video, sus tiempos de inicio y fin, el texto original y la glosa generada en mayúsculas.

4.3. limpiar_corpus.py (Filtro Estadístico de Alucinación)
Este componente realiza una purga matemática sobre el archivo CSV bruto generado en el paso anterior.

El Problema: Los modelos abstractos como T5 pueden sufrir de "alucinaciones", generando bucles repetitivos de texto o reducciones excesivas cuando se enfrentan a estructuras gramaticales complejas.

Métrica de Control: Calcula la longitud léxica de cada par y establece una métrica paramétrica: ratio_longitud = palabras_glosa / palabras_texto_natural.

Criterio de Exclusión: Aplica una máscara booleana rígida donde solo se aceptan filas cuyo ratio esté en el rango inclusivo de 0.5 a 2.0. Si una glosa tiene menos de la mitad de palabras que el texto original, o más del doble, es descartada.

Productos: Genera el archivo definitivo corpus_youtube_asl_FINAL.csv y desvía las filas corruptas hacia fragmentos_sospechosos.csv para auditorías de depuración.

4.4. auditar_corpus_final.py (Control de Calidad Léxico)
Actúa como un script de verificación post-limpieza (Quality Assurance). Lee el archivo final de manera exhaustiva buscando campos nulos dispersos o registros anómalos que hayan evadido las expresiones regulares. Imprime en consola un reporte con el porcentaje exacto de pureza del dataset. Un resultado del 100.00% certifica que el archivo está listo para entrenar el modelo clasificador.

4.5. extraccion_masiva.py (Procesamiento por Visión Computacional)
Es el núcleo de procesamiento espacial y extracción de rasgos biomecánicos.

Algoritmo de Extracción: Lee el archivo corpus_youtube_asl_FINAL.csv, realiza cortes temporales precisos sobre los videos y alimenta los fotogramas en formato RGB a la solución mediapipe.solutions.holistic.

Representación Tensorial: La función extract_keypoints(results) extrae y aplana las coordenadas (X, Y, Z) de tres estructuras anatómicas críticas: Pose (33 puntos / 99 valores), Mano Izquierda (21 puntos / 63 valores) y Mano Derecha (21 puntos / 63 valores), sumando un vector plano de exactamente 225 características por fotograma.

Filtro de Descarte Anatómico: Evalúa matemáticamente la presencia de las manos. Si en un frame las sub-matrices correspondientes a la mano izquierda y derecha contienen únicamente valores en cero, significa que el tracking se perdió (manos fuera de cuadro u ocluidas).

Umbral Crítico: Si la tasa de pérdida supera el 50% de la duración total del fragmento (frames_sin_manos / total_frames > 0.5), el clip se descarta por completo de la base de entrenamiento para evitar inestabilidad en las redes neuronales temporales.

4.6. reproducir_esqueletos.py (Sanity Check Gráfico)
Es una herramienta visual indispensable para el desarrollador. Carga una matriz binaria (.npy) generada por el extractor y, mediante un bucle gráfico sobre OpenCV, reconstruye el esqueleto tridimensional en un lienzo negro de 800x800 píxeles. Mapea colores diferenciados (Cuerpo en blanco, Mano Izquierda en verde y Mano Derecha en rojo) lo que permite detectar visualmente desajustes de tracking, pérdida de puntos o inversiones de extremidades.

5. Manual Técnico de Adaptación para la Lengua de Señas Chilena (LSCh)
Para reutilizar este potente ecosistema de software y adaptarlo al procesamiento de la Lengua de Señas Chilena (LSCh), el desarrollador sucesor debe ejecutar de forma obligatoria las siguientes modificaciones estructurales en el código:

Fase I: Modificación de la Captura Multimedia (descargar_youtube_asl.py)
El script actual busca de manera exclusiva subtítulos en inglés mediante filtros de nombres. Para capturar el contexto nacional:

Modificar Filtro de Idioma: Localice la expresión regular RE_srt_EN en las líneas superiores del archivo.

Adaptar Expresión Regular: Cambie el patrón para que acepte subtítulos codificados para el idioma español y la región chilena. Reemplace la lógica por expresiones que capturen sufijos como .es.srt o .es-CL.srt.

Actualizar ids.txt: Vacíe el contenido actual de ids.txt e inserte identificadores de canales de YouTube que transmitan contenido oficial con intérpretes de señas chilenos en recuadro fijo. Fuentes recomendadas: Canal oficial de la Cámara de Diputadas y Diputados de Chile, transmisiones del Poder Judicial de Chile, noticieros nacionales con recuadro oficial de accesibilidad o material pedagógico inclusivo de la Biblioteca del Congreso Nacional (BCN).

Fase II: Substitución del Modelo de Glosado (generar_corpus_asl.py)
El modelo T5 configurado originalmente está entrenado para mapear sintaxis inglesa a ASL y no procesará español correctamente. El nuevo desarrollador debe:

Reemplazar MODEL_PATH: Modificar la variable para apuntar a un modelo Transformer entrenado específicamente en el par lingüístico Español -> Glosa LSCh.

Alternativa por API de LLM: En caso de no contar con un modelo T5 ajustado para LSCh en el disco local, debe reescribir el cuerpo de la función traducir_batch. Se puede implementar una conexión directa con la API de un modelo fundacional avanzado (como Gemini) mediante el SDK de Google AI Studio, enviando los subtítulos en lotes acompañados de un prompt estructurado de tipo Few-Shot Learning (ejemplo: "Traduce las siguientes frases en español a su estructura conceptual de glosas en Lengua de Señas Chilena, manteniendo el orden Sujeto-Objeto-Verbo, removiendo artículos y nexos, y entregando la salida estrictamente en mayúsculas").

Fase III: Recalibración Estricta de Ratios Léxicos (limpiar_corpus.py)
La densidad de palabras y las pausas gramaticales entre el español hablado y la estructura visogestual de la LSCh difieren notablemente de los patrones del inglés y la ASL.

Riesgo de Descarte: Si mantiene el filtro original (0.5 a 2.0), el script podría descartar masivamente glosas válidas chilenas debido a la naturaleza sintáctica del español (uso intensivo de preposiciones que no se traducen a señas individuales).

Procedimiento de Ajuste: Ejecute un lote piloto de prueba con 50 videos y abra el archivo fragmentos_sospechosos.csv. Si observa que frases lingüísticamente correctas están siendo rechazadas, altere los umbrales matemáticos en la sección de filtrado del script:

Python
# Código original:
# df_limpio = df_limpio[(df_limpio['ratio_longitud'] <= 2.0) & (df_limpio['ratio_longitud'] >= 0.5)]

# Modificación sugerida para la transición a Español/LSCh:
df_limpio = df_limpio[(df_limpio['ratio_longitud'] <= 2.5) & (df_limpio['ratio_longitud'] >= 0.4)]
Fase IV: Calibración del Filtro de Visión y Encuadres (extraccion_masiva.py)
Los videos institucionales o televisivos en Chile suelen utilizar planos de cámara específicos (planos medios, bustos o recuadros pequeños en la esquina inferior de la pantalla). Esto puede generar pérdidas parciales de oclusión en MediaPipe Holistic cuando el intérprete realiza señas pegadas al torso o fuera del eje central.

Ajustar Tolerancia Geométrica: Si el extractor masivo empieza a descartar un porcentaje muy alto de clips (reportados bajo la etiqueta Clips descartados (sin manos > 50%)), el desarrollador debe flexibilizar el umbral de aceptación.

Modificación del Código: Localice la línea condicional del filtro en extraccion_masiva.py y modifique el ratio permisivo de pérdida de frames de manos de la siguiente manera:

Python
# Código original:
# if frames_sin_manos / len(matriz) > 0.5:

# Modificación para encuadres chilenos de TV o recuadros reducidos:
if frames_sin_manos / len(matriz) > 0.65:
    clips_descartados_presencia += 1
Con esto, se evitará la pérdida innecesaria de material valioso de LSCh que contenga oclusiones momentáneas causadas por la compresión del video de origen o por la velocidad del señado tradicional chileno.
