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
    #Qui sera la Miss France 2025 ? / Quelle Miss figurera dans le Top 15 / Top 5 ?")
    st.subheader('Notre projet: ', divider='violet')
    description_projet = st.write(" Bienvenue dans notre projet ambitieux de prédiction de Miss France 2025 !\n\nNous nous sommes lancés dans cette aventure passionnante en utilisant des techniques de Web Scraping pour recueillir des données exhaustives sur les candidates à Miss France de 2009 à 2024, avec leurs caractéristiques détaillées.")

    st.subheader('En quoi consiste notre modèle ?')
    description_model = st.write("Chaque année, notre équipe se mobilise avec un zèle renouvelé pour affiner nos modèles et tenter de prédire avec précision le top 5 des finalistes. Grâce à notre approche méthodique et à l'analyse minutieuse de vastes ensembles de données, nous nous efforçons de fournir des prédictions aussi fiables que possible.Nos données proviennent d'une variété de sources sur Internet, y compris des sites officiels, des articles de presse et des plateformes de médias sociaux.\n\n Nous collectons des informations sur les caractéristiques des candidates telles que leur région d'origine, leur âge, leur taille, et bien plus encore.Mais notre analyse ne s'arrête pas là. Nous examinons également la composition du jury, les tendances sur les réseaux sociaux, et d'autres facteurs pertinents pour affiner nos prédictions. L'objectif est de créer un modèle robuste et complet qui prend en compte toutes les variables pertinentes pour déterminer le classement final des candidates.Cette année, nous avons redoublé d'efforts pour améliorer notre méthodologie et affiner nos prédictions. Nous avons élargi notre champ d'analyse pour inclure une gamme encore plus large de données, et nous avons investi dans de nouvelles technologies pour améliorer la précision de nos modèles.Nous sommes impatients de partager nos résultats avec vous et de vous présenter nos prédictions pour Miss France 2025. Restez à l'écoute pour découvrir qui sont les candidates les plus prometteuses selon notre analyse approfondie des données !")



    #st.write("Bienvenue sur la page d'accueil !")
    #st.balloons()