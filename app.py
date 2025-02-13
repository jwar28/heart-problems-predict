import streamlit as st
import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.neighbors import KNeighborsClassifier
from PIL import Image
import requests
from io import BytesIO

# Cargar el modelo y el escalador
modelo_knn = joblib.load('modelo_knn.bin')
escalador = joblib.load('escalador.bin')

# Título y introducción
st.title("Asistente IA para cardiólogos")
st.write("""
Este asistente de inteligencia artificial ha sido diseñado para predecir si una persona podría tener problemas cardíacos o no.
La predicción se basa en datos de edad y colesterol. Introduzca los datos en la sección de capturar datos, luego seleccione la pestaña para hacer la predicción.
""")

# Pestaña de instrucciones
with st.expander("Instrucciones de uso"):
    st.write("""
    - **Edad**: Ingrese la edad de la persona entre 18 y 80 años.
    - **Colesterol**: Ingrese el valor del colesterol entre 50 y 600.
    - **Predicción**: Después de ingresar los valores, seleccione la pestaña de 'Predicción' para obtener el diagnóstico.
    - El modelo ha sido entrenado utilizando un clasificador KNN (K-Nearest Neighbors) y los datos han sido normalizados previamente utilizando MinMaxScaler de scikit-learn.
    """)

# Crear los tabs para capturar datos y hacer la predicción
tabs = st.radio("Selecciona una pestaña", ["Capturar Datos", "Predicción"])

# Pestaña de Capturar Datos
if tabs == "Capturar Datos":
    st.header("Capturar datos de entrada")

    # Entradas del usuario
    edad = st.slider("Edad", 18, 80, 30)
    colesterol = st.slider("Colesterol", 50, 600, 150)

    # Guardar los datos en el session state
    st.session_state.edad = edad
    st.session_state.colesterol = colesterol

    st.write("Haz clic en la pestaña 'Predicción' para obtener el diagnóstico.")

# Pestaña de Predicción
elif tabs == "Predicción":
    if 'edad' in st.session_state and 'colesterol' in st.session_state:
        st.header("Resultado de la Predicción")

        # Crear un DataFrame con los mismos nombres de las columnas que el modelo espera
        datos_entrada = pd.DataFrame({
            'edad': [st.session_state.edad],
            'colesterol': [st.session_state.colesterol]
        })

        # Normalizamos los datos de entrada
        datos_normalizados = escalador.transform(datos_entrada)

        # Realizamos la predicción
        prediccion = modelo_knn.predict(datos_normalizados)

        # Mostrar el resultado de la predicción
        if prediccion == 1:
            st.write("**Diagnóstico: Tiene problema cardíaco.**")
            # Descargar y mostrar la imagen de problema cardíaco
            url_imagen = "https://www.clikisalud.net/wp-content/uploads/2018/09/problemas-cardiacos-jovenes.jpg"
            respuesta = requests.get(url_imagen)
            img = Image.open(BytesIO(respuesta.content))
            st.image(img, caption="Problema cardíaco", use_column_width=True)
        else:
            st.write("**Diagnóstico: No tiene problema cardíaco.**")
            # Descargar y mostrar la imagen de salud
            url_imagen = "https://previews.123rf.com/images/yatate10/yatate101901/yatate10190100019/125884416-el-ni%C3%B1o-sano-refleja-el-ataque-de-bacterias-estilo-de-vida-saludable.jpg"
            respuesta = requests.get(url_imagen)
            img = Image.open(BytesIO(respuesta.content))
            st.image(img, caption="Salud excelente", use_column_width=True)
    else:
        st.warning("Por favor, captura los datos de entrada en la pestaña 'Capturar Datos' antes de realizar la predicción.")

# Footer con autor
st.write("""
- **Realizado por**: Jorge Vergel
""")