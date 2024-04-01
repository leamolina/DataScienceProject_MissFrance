import streamlit as st
import pandas as pd

# Charger les données
#data = pd.read_csv('/Users/anaellecohen/Desktop/Cours/M1 I2D/S2/Projet:ML/Projet/data_missFrance_new.numbers')


couleurPrincipale = "#0000FF"
couleurDeFond = "#FFFFFF"
couleurDeFondSecondaire = "#F0F0F0"
couleurDuTexte = "#000000"


def main():
    # Personnalisation de la couleur de fond de page
    st.markdown(
        """
        <style>
        .reportview-container {
            background: 'blue'; 
        }
        </style>
        """,
        unsafe_allow_html=True
    )
    #chemin_lea = '/Users/LEAMOLINA1/Desktop/M1/S2/Projet ML : DataScience/Projet/COHEN_Anaelle_MOLINA_Lea/data_missFrance.csv'
    chemin_ana = '/Users/anaellecohen/Desktop/Cours/M1 I2D/S2/Projet:ML/Projet/DataScienceProject_MissFrance/data_missFrance.csv'
    #data = pd.read_csv(chemin_lea)
    data = pd.read_csv(chemin_ana)

    #chemin_logo_lea_ = '/Users/LEAMOLINA1/Desktop/M1/S2/Projet ML : DataScience/Projet/COHEN_Anaelle_MOLINA_Lea/Logo_MissFrance.png'
    chemin_logo_ana = '/Users/anaellecohen/Desktop/Cours/M1 I2D/S2/Projet:ML/Projet/DataScienceProject_MissFrance/Logo_MissFrance.png'

    st.title("MISS FRANCE 2025")
    st.text("Bienvenu dans notre projet")
    st.text("Nous allons essayer de prédire la Miss France 2025")
    st.image(chemin_logo_ana, use_column_width=True)

    #st.audio(data)
    #st.video(data)
    #st.video(data, subtitles="./subs.vtt")

    #chemin_audio_lea = "/Users/LEAMOLINA1/Desktop/M1/S2/Projet ML : DataScience/Projet/COHEN_Anaelle_MOLINA_Lea/Generique_Miss_France.mp3"  # Remplacez par le chemin de votre fichier audio
    chemin_audio_ana = "/Users/anaellecohen/Desktop/Cours/M1 I2D/S2/Projet:ML/Projet/DataScienceProject_MissFrance/Générique Miss France.mp3"  # Remplacez par le chemin de votre fichier audio

    st.audio(chemin_audio_ana, format='audio/mp3')
    #chemin_video_lea= "/Users/LEAMOLINA1/Desktop/M1/S2/Projet ML : DataScience/Projet/COHEN_Anaelle_MOLINA_Lea/Couronnement.mp4"
    chemin_video_ana= "/Users/anaellecohen/Desktop/Cours/M1 I2D/S2/Projet:ML/Projet/DataScienceProject_MissFrance/Couronnement.mp4"

    st.video(chemin_video_ana, format="video/mp4", start_time=0, subtitles=None)
    #audio_html = f'<audio src="{chemin_audio}" autoplay controls></audio>'
    #st.markdown(audio_html, unsafe_allow_html=True)
if __name__ == "__main__":
    main()


# Afficher les données dans un tableau
#st.table(data)

#st.line_chart(data)