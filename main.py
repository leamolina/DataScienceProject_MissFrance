import pandas as pd
import streamlit as st

from Interface.data import page_data
from Interface.prediction import page_prediction
from Interface.presentation import page_presentation

#Chemins
chemin_database = './Databases/data_missFrance.csv'
chemin_logo = './Sources/Logo_MissFrance.png'
chemin_audio = "./Sources/Generique_Miss_France.mp3"
chemin_video = "./Sources/Couronnement.mp4"

#Charger les données
data = pd.read_csv(chemin_database,delimiter=';')

st.sidebar.title("Projet")
pages = {
    "Présentation": page_presentation,
    "Nos données": page_data,
    "Prédiction": page_prediction,
}
choix_page = st.sidebar.radio("Aller à", list(pages.keys()))
pages[choix_page]()