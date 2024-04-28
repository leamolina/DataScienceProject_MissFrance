import numpy as np
import streamlit as st
import pandas as pd


# Chemins

chemin_lea = '/Users/LEAMOLINA1/Desktop/M1/S2/Projet ML : DataScience/Projet/COHEN_Anaelle_MOLINA_Lea/data_missFrance.csv'
chemin_ana = '/Users/anaellecohen/Desktop/Cours/M1 I2D/S2/Projet:ML/Projet/DataScienceProject_MissFrance/data_missFrance.csv'
chemin_logo_lea = '/Users/LEAMOLINA1/Desktop/M1/S2/Projet ML : DataScience/Projet/COHEN_Anaelle_MOLINA_Lea/Sources/Logo_MissFrance.png'
chemin_logo_ana = '/Users/anaellecohen/Desktop/Cours/M1 I2D/S2/Projet:ML/Projet/DataScienceProject_MissFrance/Sources/Logo_MissFrance.png'
chemin_audio_lea = "/Users/LEAMOLINA1/Desktop/M1/S2/Projet ML : DataScience/Projet/COHEN_Anaelle_MOLINA_Lea/Sources/Generique_Miss_France.mp3"  # Remplacez par le chemin de votre fichier audio
chemin_audio_ana = "/Users/anaellecohen/Desktop/Cours/M1 I2D/S2/Projet:ML/Projet/DataScienceProject_MissFrance/Sources/Generique_Miss_France.mp3"
chemin_video_lea = "/Users/LEAMOLINA1/Desktop/M1/S2/Projet ML : DataScience/Projet/COHEN_Anaelle_MOLINA_Lea/Sources/Couronnement.mp4"
chemin_video_ana = "/Users/anaellecohen/Desktop/Cours/M1 I2D/S2/Projet:ML/Projet/DataScienceProject_MissFrance/Sources/Couronnement.mp4"

#Charger les données
data = pd.read_csv(chemin_lea,delimiter=';')
# data = pd.read_csv(chemin_lea)


#Esthétique page
couleurPrincipale = "#0000FF"
couleurDeFond = "#FFFFFF"
couleurDeFondSecondaire = "#F0F0F0"
couleurDuTexte = "#000000"


def page_accueil():
    st.title("Qui sera Miss France 2025 ?")
    st.audio(chemin_audio_lea, format='audio/mp3')
    st.video(chemin_video_lea, format="video/mp4", start_time=0, subtitles=None)
    st.write("Bienvenue sur la page d'accueil !")
    st.balloons()
def page_1():
    st.image(chemin_logo_lea, use_column_width=True,width=100)
    st.title("Qu'est-ce-que Miss France ?")
    st.write("C'est la page 1.")
    st.title("Répartition de la couleur de cheveux des Miss par année")
    data_subset = data[['annee', 'couleur_cheveux']]
    #print(data_subset)

    # Regrouper les données par année et par couleur de cheveux, puis compter le nombre d'occurrences
    cheveux_par_annee = data_subset.groupby(['annee', 'couleur_cheveux']).size().reset_index(name='nombre')

    # Afficher l'histogramme
    st.bar_chart(cheveux_par_annee, x='annee', y='nombre', color='couleur_cheveux', use_container_width=True)

    chart_data = pd.DataFrame(cheveux_par_annee, data_subset)

    #st.bar_chart(chart_data)
def page_2():
    st.image(chemin_logo_lea, use_column_width=True,width=100)
    st.title("Page 2")
    st.write("C'est la page 2.")

def contact():
    st.title("Nous contacter")
    st.write("C'est la page de contact.")
    st.image(chemin_logo_lea, use_column_width=True)
    st.write("anaelle.cohen@dauphine.eu")
    st.write("lea.molina@dauphine.eu")


st.sidebar.title("Projet")
pages = {
    "Accueil": page_accueil,
    "Qu\'est-ce-que Miss France ?": page_1,
    "Page 2": page_2,
    "Contact": contact
}
choix_page = st.sidebar.radio("Aller à", list(pages.keys()))

# Charger l'image de paillettes
#chemin_image_paillettes = '/Users/anaellecohen/Desktop/Cours/M1 I2D/S2/Projet:ML/Projet/DataScienceProject_MissFrance/Sources/paillettes2.jpeg'
#st.image(chemin_image_paillettes, use_column_width=True, caption='Image de paillettes')

# Exécuter la fonction correspondant à la page sélectionnée
pages[choix_page]()







# Afficher les données dans un tableau
#st.table(data)

#st.line_chart(data)