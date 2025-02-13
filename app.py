import streamlit as st
import joblib
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.neighbors import KNeighborsClassifier
from PIL import Image

# Cargar el modelo y el escalador desde los archivos guardados
modelo_knn = joblib.load('modelo_knn.bin')
escalador = joblib.load('escalador.bin')

# Nombres de los features que el modelo espera
columnas = ['edad', 'colesterol']

# Título de la aplicación
st.title('Asistente IA para Cardiólogos')

# Introducción
st.write("""
Este asistente utiliza un modelo de inteligencia artificial basado en KNN para predecir si una persona tiene o no problemas cardíacos. 
Puedes ingresar la edad y el nivel de colesterol para obtener la predicción. 
Además, te proporcionamos imágenes informativas dependiendo del resultado de la predicción.
""")

# Crear dos tabs: uno para captura y otro para la predicción
tab_captura, tab_prediccion = st.tabs(['Captura de Datos', 'Predicción'])

# Variables globales para almacenar los datos de entrada
datos_usuario = None

with tab_captura:
    st.header("Captura de Datos")
    
    # Solicitar al usuario que ingrese los datos
    edad = st.slider('Edad', 18, 80, 25)
    colesterol = st.slider('Colesterol', 50, 600, 200)
    
    # Botón para guardar los datos de entrada
    if st.button("Guardar Datos"):
        # Crear un DataFrame con los datos ingresados
        datos_usuario = pd.DataFrame([[edad, colesterol]], columns=columnas)
        st.success("Datos guardados correctamente. Ahora puedes ir a la predicción.")

with tab_prediccion:
    st.header("Predicción")
    
    # Verificar si el usuario ha ingresado los datos antes
    if datos_usuario is not None:
        # Normalizar los datos de entrada
        datos_normalizados = escalador.transform(datos_usuario)
        
        # Hacer la predicción con el modelo KNN
        prediccion = modelo_knn.predict(datos_normalizados)
        
        if prediccion == 1:
            # Si tiene problema cardíaco
            st.write("**¡Alerta!** La predicción indica que la persona tiene problemas cardíacos.")
            imagen = Image.open('https://blog.clinicainternacional.com.pe/wp-content/uploads/2017/11/clinica-internacional-ataque-corazon-sintomas-causas.jpg')
            st.image(imagen, caption="Posibles síntomas de un ataque cardíaco", use_column_width=True)
        else:
            # Si no tiene problema cardíaco
            st.write("**¡Todo está bien!** La predicción indica que la persona no tiene problemas cardíacos.")
            imagen = Image.open('https://e7.pngegg.com/pngimages/270/467/png-clipart-smiley-heart-emoticon-sticker-tire-love-miscellaneous.png')
            st.image(imagen, caption="Estado saludable", use_column_width=True)
    else:
        st.write("Por favor, ingresa los datos en el tab de 'Captura de Datos' para obtener la predicción.")
