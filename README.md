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

### 2. Segundo Punto: Desarrollar Juego 2D Tipo Mario Bros Implementando Hilos

### 3. Tercer Punto: Desarrollar Detección de Diferentes Gestos de Mano Usando Hilos






