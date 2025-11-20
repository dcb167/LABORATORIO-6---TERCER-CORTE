# LABORATORIO 6

## Elaborado por: Laura Rodriguez y Diana Bernal 

### 1. Primer Punto : Ejercicio de Análisis de Sentimientos

+ Para poder cumplir realizar el primer punto, se emplearon dos librerías importantes las cuales fueron: 

<img width="251" height="37" alt="image" src="https://github.com/user-attachments/assets/b1acd49a-5658-4772-985c-5ef146cc96cd" /></br>

<strong> Figura 1. </strong> Librerías empleadas en Python.

+ Esto con el objetivo de poder implementar el uso de threading y la sincronización con Lock para evitar condiciones de carrera. Para la implementación del código se tuvo en cuenta únicamente dos productos electrónicos, pero el código se diseñó con el objetivo de que si se desea ampliar dichos productos se realice modificando la variable "Threads" y también anadiendo un nuevo arreglo al código con el nuevo producto electrónico que se desee añadir.
  
+ Es importante mencionar, que en cada arreglo se tiene la clasificación del sentimiento. En este caso: Positivo, Neutro y Negativo.
  
+ De tal forma que, se generó un conteo con cada clasificación por producto electrónico llegando al siguiente código resultante:


      import threading
      from collections import Counter

      Reseña_Tablet = ["Positiva","Negativa","Positiva","Neutra","Negativa","Neutra"]
      Reseña_Reloj  = ["Positiva","Positiva","Neutra","Neutra","Neutra","Neutra"]

      resultados = {}
      lock = threading.Lock()

      def Reseña_Productos_Electronicos(nombre, lista):
          conteo = Counter(lista)
          with lock:
              resultados[nombre] = conteo

      threads = [
          threading.Thread(target=Reseña_Productos_Electronicos, args=("Tablet", Reseña_Tablet)),
          threading.Thread(target=Reseña_Productos_Electronicos, args=("Reloj", Reseña_Reloj))
      ]

      for t in threads: t.start()
      for t in threads: t.join()

      print(resultados)

+ Seguido de ello, se realizo una pequeña interfaz de visualización en Streamlit. A continuación, se podra evidenciar el código realizado y el respectivo resultado:

      import streamlit as st
      import pandas as pd

      st.title(":blue[Reseña de Productos de Electrónica]")

      st.sidebar.title("Visualización de Productos de Electrónica:")
      st.sidebar.radio("Selecciona el producto",["Tablet","Reloj"])


      st.header(":blue[1. Reseña Tablet]")

      st.image("Imagen1.jpg")

      data_Tablet = pd.DataFrame({
        'Clasificación Sentimiento': ['Positiva', 'Negativa', 'Neutra'],
        'Cantidad': [2, 2, 2]
      })

      st.bar_chart(data_Tablet)

      st.header(":blue[2. Reseña Reloj]")

      st.image("Imagen2.jpg")

      data_Reloj = pd.DataFrame({
        'Clasificación': ['Neutra', 'Positiva'],
        'Cantidad': [4, 2]
      })

      st.bar_chart(data_Reloj)

+ De tal forma, que el resultado obtenido fue el siguiente:
  
<img width="1254" height="615" alt="image" src="https://github.com/user-attachments/assets/c5ba45ba-a244-4e37-bbc3-7268ef69f8b1" /></br>

<strong> Figura 2. </strong> Visualización Aplicactivo en Streamlit 1 (Interfaz Visual).

<img width="1353" height="615" alt="image" src="https://github.com/user-attachments/assets/c986c343-beb4-4a06-8f82-7e077a11925c" /></br>

<strong> Figura 3. </strong> Visualización Aplicactivo en Streamlit 2 (Interfaz Visual).

<img width="1238" height="610" alt="image" src="https://github.com/user-attachments/assets/1b6445fb-2518-4ccb-9b4d-ace514ceba05" /></br>

<strong> Figura 4. </strong> Visualización Aplicactivo en Streamlit 3 (Interfaz Visual).

+ Seguido de ello, se procedió a realizar el archivo Dockerfile y requirements.txt para poder crear la imagen como se puede visualizar a continuación:

<img width="886" height="264" alt="image" src="https://github.com/user-attachments/assets/5f32db6e-23d2-4f21-ae84-190619764ce0" /></br>

<strong> Figura 5. </strong> Crear una imagen en Docker en el S.O de Windows.

+ Por último, se ejecutó el contenedor:

<img width="622" height="123" alt="image" src="https://github.com/user-attachments/assets/60a2f8fc-2a45-41fe-96b1-ad5b8d25c27d" /></br>

<strong> Figura 6. </strong> Ejecución del contenedor en el S.O de Windows.


### 2. Segundo Punto: Desarrollar Juego 2D Tipo Mario Bros Implementando Hilos

+ Primero creamos el archivo del proyecto:
<img width="1011" height="497" alt="imagen" src="https://github.com/user-attachments/assets/87bfcacb-cb6c-42a5-bf56-9e11d44e6ca0" />

+ Luego, se inicio realizando la creación del archivio para el codigo del juego:
<img width="882" height="45" alt="imagen" src="https://github.com/user-attachments/assets/ee664d3e-1fef-4ba9-b634-6338db6fae98" />

+ Aqui se puede visualizar la inicialización del codigo del juego de Mario Bros:
<img width="1509" height="931" alt="imagen" src="https://github.com/user-attachments/assets/be2bc986-f5f4-4115-9242-1b6b1650d92f" />

+ Creación de requerimientos:
<img width="937" height="34" alt="imagen" src="https://github.com/user-attachments/assets/126f1514-5c82-4812-aedb-b6cc523bc67a" />

+ Creación del archivo del Dockerfile:
<img width="937" height="34" alt="imagen" src="https://github.com/user-attachments/assets/92928ea5-2c7e-42fc-a51f-b16d1c9887d7" />

+ Ahora seguimos con la creación del Docker-compose:
<img width="937" height="34" alt="imagen" src="https://github.com/user-attachments/assets/dc733380-ba90-46b0-8ef2-a1c3e4275f07" />

###### Una vez ya creados todos los archivos necesarios, y haber incluido hilos, mutex, semaforos, etc, iniciamos con la construcción y ejecución del juego:
+ Permitimos la conexión para la ventana del juego e iniciamos con la construcción de la imagen de Docker:
<img width="1379" height="419" alt="imagen" src="https://github.com/user-attachments/assets/9d41571e-1f61-48f3-b024-8b73b60f1584" />
<img width="1379" height="419" alt="imagen" src="https://github.com/user-attachments/assets/1f0fbd76-d7e2-4b87-8675-d915cfcf37f9" />


### 3. Tercer Punto: Desarrollar Detección de Diferentes Gestos de Mano Usando Hilos

+ Para este iteral se realizó la modificación al código que fue adjuntando en la tarea con el objetivo de adecuarlo al uso de librerías como hilos, mutex y sección crítica y se realizó el respectivo despliegue en streamlit llegando al siguiente resultado:

      import math
      import threading
      from matplotlib import pyplot as plt
      import mediapipe as mp
      from mediapipe.framework.formats import landmark_pb2
      import streamlit as st
      import numpy as np
      from PIL import Image
      import io


      image_lock = threading.Lock()
      result_lock = threading.Lock()
      processing_semaphore = threading.Semaphore(4)  

      plt.rcParams.update({
        'axes.spines.top': False,
        'axes.spines.right': False,
        'axes.spines.left': False,
        'axes.spines.bottom': False,
        'xtick.labelbottom': False,
        'xtick.bottom': False,
        'ytick.labelleft': False,
        'ytick.left': False,
        'xtick.labeltop': False,
        'xtick.top': False,
        'ytick.labelright': False,
        'ytick.right': False
      })

      mp_hands = mp.solutions.hands
      mp_drawing = mp.solutions.drawing_utils
      mp_drawing_styles = mp.solutions.drawing_styles

      class GestureProcessor:
        def __init__(self):
            self.hands = mp_hands.Hands(
                static_image_mode=True,
                max_num_hands=2,
                min_detection_confidence=0.7)
            self.results_cache = {}

        def process_image(self, image, image_id):
            """Process an image to detect hand landmarks (thread-safe)"""
            try:
                processing_semaphore.acquire()
            
            
                with result_lock:
                    if image_id in self.results_cache:
                        return self.results_cache[image_id]
            
            
                with image_lock:
                    image_np = np.array(image)
                    results = self.hands.process(image_np)
            
           
                multi_hand_landmarks = results.multi_hand_landmarks or []
                with result_lock:
                    self.results_cache[image_id] = multi_hand_landmarks
            
                return multi_hand_landmarks
        finally:
            processing_semaphore.release()

      def display_one_image(image, title, subplot, titlesize=16):
        """Displays one image along with the predicted category name and score."""
          plt.subplot(*subplot)
          plt.imshow(image)
          if len(title) > 0:
              plt.title(title, fontsize=int(titlesize), color='black', 
                 fontdict={'verticalalignment':'center'}, pad=int(titlesize/1.5))
          return (subplot[0], subplot[1], subplot[2]+1)

        def process_and_annotate_image(image, gesture, processor):
          """Process and annotate a single image with landmarks"""
            image_np = np.array(image)
            multi_hand_landmarks = processor.process_image(image, id(image))
    
            annotated_image = image_np.copy()
            if multi_hand_landmarks:
                for hand_landmarks in multi_hand_landmarks:
                    hand_landmarks_proto = landmark_pb2.NormalizedLandmarkList()
                    hand_landmarks_proto.landmark.extend([
                    landmark_pb2.NormalizedLandmark(x=landmark.x, y=landmark.y, z=landmark.z) 
                    for landmark in hand_landmarks
            ])
            
            mp_drawing.draw_landmarks(
                annotated_image,
                hand_landmarks_proto,
                mp_hands.HAND_CONNECTIONS,
                mp_drawing_styles.get_default_hand_landmarks_style(),
                mp_drawing_styles.get_default_hand_connections_style())
    
                return annotated_image, f"{gesture.category_name} ({gesture.score:.2f})"

      def display_batch_of_images_with_gestures_and_hand_landmarks(images, results):
        """Displays a batch of images with the gesture category and its score along with the hand landmarks."""
    
          processor = GestureProcessor()
    
    
          images = [image.numpy_view() for image in images]
          gestures = [top_gesture for (top_gesture, _) in results]
    
    
          rows = int(math.sqrt(len(images)))
          cols = len(images) // rows
    
    
          FIGSIZE = 13.0
          SPACING = 0.1
          subplot = (rows, cols, 1)
    
          if rows < cols:
              plt.figure(figsize=(FIGSIZE, FIGSIZE/cols*rows))
          else:
              plt.figure(figsize=(FIGSIZE/rows*cols, FIGSIZE))
    
    
          threads = []
          processed_data = []
    
          for i, (image, gesture) in enumerate(zip(images[:rows*cols], gestures[:rows*cols])):
              thread = threading.Thread(
                  target=lambda i, img, gest: processed_data.append(
                      (i, *process_and_annotate_image(img, gest, processor))),
                  args=(i, image, gesture)
              )  
              threads.append(thread)
              thread.start()
    
    
          for thread in threads:
              thread.join()
    
    
          processed_data.sort(key=lambda x: x[0])
    
    
          for _, annotated_image, title in processed_data:
              dynamic_titlesize = FIGSIZE*SPACING/max(rows,cols) * 40 + 3
              subplot = display_one_image(annotated_image, title, subplot, titlesize=dynamic_titlesize)
    
    
          plt.tight_layout()
          plt.subplots_adjust(wspace=SPACING, hspace=SPACING)
    
    
          buf = io.BytesIO()
          plt.savefig(buf, format='png', bbox_inches='tight', pad_inches=0.1)
          buf.seek(0)
          plt.close()
    
          return buf


    def main():
        st.title("Hand Gesture Recognition with MediaPipe")
        st.write("Upload images to detect hand gestures and landmarks")
    
        uploaded_files = st.file_uploader("Choose images", type=["jpg", "jpeg", "png"], accept_multiple_files=True)
    
        if uploaded_files:
            images = []
            results = []  
        
        for uploaded_file in uploaded_files:
            image = Image.open(uploaded_file)
            images.append(image)
            
            from collections import namedtuple
            Gesture = namedtuple('Gesture', ['category_name', 'score'])
            results.append((Gesture(category_name="Sample Gesture", score=0.95), None))
        
        if st.button("Process Images"):
            
            mp_images = [mp.Image(image_format=mp.ImageFormat.SRGB, data=np.array(img)) for img in images]
            
            
            image_buffer = display_batch_of_images_with_gestures_and_hand_landmarks(mp_images, results)
            st.image(image_buffer, use_container_width=True)

      if __name__ == "__main__":
      main()

+ Seguido de ello, se creo el archivo Dockerfile dando como resultado lo siguiente:

      FROM python:3.9-slim


      WORKDIR /app

      RUN apt-get update && apt-get install -y \
          libgl1 \
          libglib2.0-0 \
          libsm6 \
          libxrender1 \
          libfontconfig1 \
          && rm -rf /var/lib/apt/lists/*

        COPY requirements.txt .


        RUN pip install --no-cache-dir -r requirements.txt


        COPY . .


        EXPOSE 8501


        CMD ["python", "-m", "streamlit", "run", "punto3.py", "--server.port=8501", "--server.address=0.0.0.0"]

+ Luego, se realizó el archivo requirements.txt de la siguiente forma:

        streamlit
        mediapipe
        matplotlib
        numpy
        Pillow
+ Después, se ejecutó la imagen y el contenedor para poder visualizar el streamlit con el resultado final de la detección de la mano:

<img width="713" height="376" alt="image" src="https://github.com/user-attachments/assets/880df93d-b3ac-430a-b18c-8e78f9f73df4" /></br>

<strong> Figura 7. </strong> Visualización del Streamlit.

<img width="517" height="387" alt="image" src="https://github.com/user-attachments/assets/e3e07bf4-7cc1-4a41-941e-8f380afc1a93" /></br>

<strong> Figura 7. </strong> Visualización de la detección de la mano (Resultado Final).






