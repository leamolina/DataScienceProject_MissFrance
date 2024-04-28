import streamlit as st
import pandas as pd


#Chemins
chemin_database = '../Databases/data_missFrance.csv'
chemin_logo = '../Sources/Logo_MissFrance.png'
chemin_audio = "../Sources/Generique_Miss_France.mp3"
chemin_video = "../Sources/Couronnement.mp4"

#Charger les données
data = pd.read_csv(chemin_database,delimiter=';')

def page_presentation():
    st.title("Qui sera Miss France 2025 ?")
    st.audio(chemin_audio, format='audio/mp3')
    st.video(chemin_video, format="video/mp4", start_time=0)

    st.subheader('Notre projet: ', divider='rainbow')
    st.subheader('En quoi consiste notre modèle ?')

    #st.write("Bienvenue sur la page d'accueil !")
    #st.balloons()