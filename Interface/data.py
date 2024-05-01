import streamlit as st
import pandas as pd

#Chemins
chemin_database = './Databases/data_missFrance.csv'
chemin_logo = './Sources/Logo_MissFrance.png'
chemin_audio = "./Sources/Generique_Miss_France.mp3"
chemin_video = "./Sources/Couronnement.mp4"

#Charger les données
data = pd.read_csv(chemin_database,delimiter=';')

def page_data():
    st.image(chemin_logo, use_column_width=True,width=100)
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