import pickle

import streamlit as st
import math
import pandas as pd
import streamlit as st
import joblib
import json
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import OneHotEncoder, StandardScaler, OrdinalEncoder
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

import Modele.my_model as my_model
import Modele.data_split as ds


#Chemins
chemin_database = './Databases/data_missFrance.csv'
chemin_logo = './Sources/Logo_MissFrance.png'
chemin_audio = "./Sources/Generique_Miss_France.mp3"
chemin_video = "./Sources/Couronnement.mp4"

#Charger les données
data = pd.read_csv(chemin_database,delimiter=';')



#Fonctions diverses :

def tranform_data(X_train, X_test):
    # Order for the ordinal encoding
    hair_length_order = ["Longs", "Mi-longs", "Courts"]
    hair_color_order = ["Noirs", "Bruns", "Chatains", "Roux", "Blonds"]
    skin_color_order = ["Noire", "Métisse", "Blanche"]
    eye_color_order = ["Noirs", "Marrons", "Gris", "Bleus", "Verts"]
    categories = [hair_length_order, hair_color_order, eye_color_order, skin_color_order]

    # Créer un ColumnTransformer avec votre Custom_OneHotEncoder et d'autres transformations
    ct = ColumnTransformer([
        ("preprocessing_one_hot_encoder", OneHotEncoder(handle_unknown="ignore"), ["annee", "region"]),
        ("preprocessing_age", StandardScaler(), ["age", "taille"]),
        ("preprocessing_OrdinalEncoder", OrdinalEncoder(categories=categories),
         ['longueur_cheveux', 'couleur_cheveux', 'couleur_yeux', 'couleur_peau']),
        ("preprocessing_general_knowledge_test", SimpleImputer(strategy="constant", fill_value=0),
         ["laureat_culture_generale"]),
    ],
        remainder="passthrough")

    ct.fit(X_train)
    X_train = ct.transform(X_train)
    X_test = ct.transform(X_test)
    return X_train, X_test


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
    data_missFrance = pd.read_csv('./Databases/data_missFrance.csv', delimiter=';')
    data_missFrance_copy = data_missFrance.copy()
    data_missFrance = data_missFrance.drop(["audience", "name", "image"], axis=1)

    annee_test = 2019
    X_train, X_test, y_train, y_test = ds.data_split(data_missFrance, annee_test)
    filtered_df = data_missFrance_copy[data_missFrance_copy['annee'] == annee_test]
    list_candidate = filtered_df['name'].tolist()
    X_train, X_test = tranform_data(X_train, X_test)

    """
    #Récupération des hyperparamètres du modèle (avec JSON) 
    model_classes = {
        "RandomForestClassifier": RandomForestClassifier,
        "DecisionTreeClassifier": DecisionTreeClassifier,
        "SVC": SVC,
        "LogisticRegression": LogisticRegression
    }

    # Chargement des informations du modèle à partir du fichier JSON
    with open('./Modele/data.json', 'r') as f:
        model_info = json.load(f)

    # Extraire le type de modèle et les paramètres du dictionnaire d'informations
    model_type = model_info["model_type"]
    params = model_info["params"]

    model_classes = {
        "RandomForestClassifier": RandomForestClassifier,
        "DecisionTreeClassifier": DecisionTreeClassifier,
        "SVC": SVC,
        "LogisticRegression": LogisticRegression
    }
    if model_type in model_classes:
        best_model_class = model_classes[model_type]

    # Créer une liste de 12 objets RandomForestClassifier avec des paramètres différents
    list_of_models = []
    for i in range(12):
        model = best_model_class(**params)
        list_of_models.append(model)

    myModel = my_model.MyModel(list_of_models)
    myModel.fit(X_train, y_train)
    
    #myModel = load('./Modele/myModelRanking.joblib')
    print("YOUHOUUUU")
    """
    list_of_models = []
    for i in range(12):
        path = './Modele/train/model_'+str(i)+'.pkl'
        model = pickle.load(open(path, 'rb'))
        list_of_models.append(model)


    st.write("Le modèle a été récupéré avec succès ")
    myModel = my_model.MyModel(list_of_models)
    st.write("La classe a été créée avec succès")


    #Affichage des prédictions
    prediction = myModel.predict(X_test, list_candidate)
    real_rank = give_real_rank(data_missFrance_copy, annee_test)
    st.write("prediction : ", prediction)
    st.write("vrai classement :", real_rank)
    st.write("score de prédiction lea :", evaluate_prediction(prediction, real_rank))

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



