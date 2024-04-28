import numpy as np
import streamlit as st
import pandas as pd

#Chemins
chemin_database = '../Databases/data_missFrance.csv'
chemin_logo = '../Sources/Logo_MissFrance.png'
chemin_audio = "../Sources/Generique_Miss_France.mp3"
chemin_video = "../Sources/Couronnement.mp4"

#Charger les données
data = pd.read_csv(chemin_database,delimiter=';')

def page_prediction():
    st.image(chemin_logo, use_column_width=True,width=100)
    st.title("Page 2")
    st.write("C'est la page 2.")

