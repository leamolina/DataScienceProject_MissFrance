import streamlit as st
import pandas as pd

#Chemins
chemin_lea = '/Users/LEAMOLINA1/Desktop/M1/S2/Projet ML : DataScience/Projet/COHEN_Anaelle_MOLINA_Lea/data_missFrance.csv'
chemin_ana = '/Users/anaellecohen/Desktop/Cours/M1 I2D/S2/Projet:ML/Projet/DataScienceProject_MissFrance//data_missFrance.csv'
chemin_logo_lea = '/Users/LEAMOLINA1/Desktop/M1/S2/Projet ML : DataScience/Projet/COHEN_Anaelle_MOLINA_Lea/sources/Logo_MissFrance.png'
chemin_logo_ana = '/Users/anaellecohen/Desktop/Cours/M1 I2D/S2/Projet:ML/Projet/DataScienceProject_MissFrance/sources/Logo_MissFrance.png'
chemin_audio_lea = "/Users/LEAMOLINA1/Desktop/M1/S2/Projet ML : DataScience/Projet/COHEN_Anaelle_MOLINA_Lea/sources/Generique_Miss_France.mp3"
chemin_audio_ana = "/Users/anaellecohen/Desktop/Cours/M1 I2D/S2/Projet:ML/Projet/DataScienceProject_MissFrance/sources/Générique Miss France.mp3"
chemin_video_lea = "/Users/LEAMOLINA1/Desktop/M1/S2/Projet ML : DataScience/Projet/COHEN_Anaelle_MOLINA_Lea/sources/Couronnement.mp4"
chemin_video_ana = "/Users/anaellecohen/Desktop/Cours/M1 I2D/S2/Projet:ML/Projet/DataScienceProject_MissFrance/sources/Couronnement.mp4"

# Fonction pour afficher la page d'accueil
def accueil():
    # Récupération des données
    data = pd.read_csv(chemin_lea)
    # data = pd.read_csv(chemin_ana)
    st.title("MISS FRANCE 2025")
    st.image(chemin_logo_lea, use_column_width=False, output_format='auto', width=100, clamp=False, channels='RGB')
    st.write("Bienvenu dans notre projet  \nNous allons essayer de prédire la Miss France 2025")

    st.audio(chemin_audio_lea)
    st.video(chemin_video_lea, format="video/mp4", start_time=0, subtitles=None)


def page_un():
    st.title("Nos données")
    st.write("Ici on va présenter notre Dataset.")


def page_deux():
    st.title("Prédictions")
    st.write("Ici on va prédire notre future Miss France.")

st.sidebar.image(chemin_logo_lea, width=100)
st.sidebar.markdown("<h1 style='text-align: center;'>Navigation</h1>", unsafe_allow_html=True)
page = st.sidebar.radio(
    "",
    ("Accueil", "Présentation des données", "Prédictions")
)

# Affichage de la page sélectionnée
if page == "Accueil":
    accueil()
elif page == "Présentation des données":
    page_un()
elif page == "Prédictions":
    page_deux()
