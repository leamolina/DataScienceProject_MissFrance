import numpy as np
import streamlit as st
import pandas as pd
from joblib import dump, load
import math

import os
import sys
# Chemin absolu du dossier parent (interface/)
current_dir = os.path.dirname(os.path.realpath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)
from Modele.data_split import data_split


#Chemins
chemin_database = '../Databases/data_missFrance.csv'
chemin_logo = '../Sources/Logo_MissFrance.png'
chemin_audio = "../Sources/Generique_Miss_France.mp3"
chemin_video = "../Sources/Couronnement.mp4"

#Charger les données
data = pd.read_csv(chemin_database,delimiter=';')



#Fonctions diverses :
def evaluate_prediction(prediction, real_score):
    sum = 0
    for (key, value) in prediction.items():
        if(value in real_score.keys()):
            diff = key - real_score[value]
        else :
            diff = 20
        sum += math.pow(diff,2)
    return sum

def give_real_rank(df, annee):
    filtered_df = df[df['annee'] == annee]
    rank = {}
    for i in range(1,13):
        miss = filtered_df.loc[df['rang'] == i, 'name'].tolist()[0]
        rank[miss] = i
    return rank

#Option 1 : Prédire (manuellement) la miss France 2025
def page_option1():
    st.write("Nous sommes dans l'option 1")

#Option 2 : Voir la prédiction de la miss France 2022
def page_option2():
    st.write("Nous sommes dans l'option 2")
    # Récupération des données
    data_missFrance = pd.read_csv('../Databases/data_missFrance.csv', delimiter=';')
    data_missFrance_copy = data_missFrance.copy()
    data_missFrance = data_missFrance.drop(["audience", "name", "image"], axis=1)

    annee_test = 2019
    X_train, X_test, y_train, y_test = data_split(data_missFrance, annee_test)
    filtered_df = data_missFrance_copy[data_missFrance_copy['annee'] == annee_test]
    list_candidate = filtered_df['name'].tolist()
    # Chemin relatif pour accéder au dossier "Modele" depuis "Interface"
    modele_path = os.path.join(os.path.dirname(__file__), '..', 'Modele', 'myModelRanking.joblib')

    # Charger le modèle
    #myModel = load(modele_path)current_dir = os.path.dirname(os.path.realpath(__file__))
    parent_dir = os.path.dirname(current_dir)
    sys.path.append(parent_dir)

    myModel = load('../Modele/myModelRanking.joblib')
    """
    ct = load('../Modele/column_transformer.joblib')
    X_test = ct.transform(X_test)

    prediction = myModel.predict(X_test, list_candidate)
    real_rank = give_real_rank(data_missFrance_copy, annee_test)
    st.write("prediction : ", prediction)
    st.write("vrai classement :", real_rank)
    st.write("score de prédiction lea :", evaluate_prediction(prediction, real_rank))"""


def page_prediction():
    st.title("Prediction")
    #Logo en haut à droite
    col1, col2, col3 = st.columns((1, 4, 1))
    with col3:
        st.image(chemin_logo, use_column_width=True, width=10)
    option1 = "Option 1 : Prédire manuellement la miss France 2025"
    option2 = "Option 2 : Voir la prédiction de la miss France 2022"
    st.title("Prédiction Miss France")
    st.write("Nous allons prédire la future Miss France")

    options = [option1, option2]

    selected_option = st.selectbox("Choisissez votre option 👇", options)

    with st.expander("A propos"):
        st.markdown(
            """
        The **#30DaysOfStreamlit** is a coding challenge designed to help you get started in building Streamlit apps.

        Particularly, you'll be able to:
        - Set up a coding environment for building Streamlit apps
        - Build your first Streamlit app
        - Learn about all the awesome input/output widgets to use for your Streamlit app
        """
        )

    if selected_option == option1:
        page_option1()
    elif selected_option == option2:
        page_option2()
    else:
        print("Erreur ?")



