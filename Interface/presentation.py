import streamlit as st
import pandas as pd


#Chemins
chemin_database = '../Databases/data_missFrance.csv'
chemin_logo = '../Sources/Logo_MissFrance.png'
chemin_audio = "../Sources/Generique_Miss_France.mp3"
chemin_video = "../Sources/Video_avec_audio.mp4"

#Charger les données
data = pd.read_csv(chemin_database,delimiter=';')

def page_presentation():
    st.title("Qui sera Miss France 2025 ? :crown:")
    st.video(chemin_video, format="video/mp4", start_time=0)

    st.subheader('Notre projet: ', divider='violet')
    st.write(" Bienvenue dans notre projet de prédiction de Miss France 2025 ! :crown:\n\n Nous nous sommes lancées dans ce projet passionnant en utilisant des techniques de Web Scraping pour recueillir des données sur les candidates à Miss France de 2009 à 2024, avec leurs caractéristiques détaillées.\n\n Notre but : Déterminer QUI sera Miss France 2025 et  \n\n Quelles candidates feront parties du top 12 ?")

    st.subheader('En quoi consiste notre modèle ?', divider='violet')
    st.write("Notre modèle tente de prédire avec précision le top 12 des candidates. Grâce à notre approche méthodique et à l'analyse minutieuse de vastes ensembles de données, nous essayons de fournir des prédictions aussi fiables que possible.\n\nNos données proviennent d'une variété de sources sur Internet, y compris des sites officiels, et des articles de presse. Nous avons collecté des informations sur les caractéristiques des candidates telles que leur région d'origine, leur âge, leur taille, et bien plus encore. \n\nL'objectif est de créer un modèle robuste et complet qui prend en compte toutes les variables pertinentes pour construire un modèle nous permettant d’obtenir un score de prédiction pour chacune des Miss et déterminer le classement final des candidates. \n\nN'hésitez pas à cliquer sur la rubrique PRÉDICTION pour découvrir qui sont les candidates les plus prometteuses selon notre analyse approfondie des données !")
    st.write("\n\n\n\nPour cette 95ème élection, ce sont 30 candidates âgées de 18 à 28 ans qui sont en lice pour la couronne de Miss France 2025.")
