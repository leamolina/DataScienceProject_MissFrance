import math
import pickle
import pandas as pd
import streamlit as st
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler, OrdinalEncoder
import Modele.data_split as ds
import Modele.my_model as my_model

#Chemins
chemin_database = './Databases/data_missFrance.csv'
chemin_logo = './Sources/Logo_MissFrance.png'
chemin_audio = "./Sources/Generique_Miss_France.mp3"
chemin_video = "./Sources/Couronnement.mp4"

#Charger les données
data = pd.read_csv(chemin_database,delimiter=';')



#Fonctions diverses :

def give_real_rank(df, annee):
    filtered_df = df[df['annee'] == annee]
    rank = {}
    for i in range(1,13):
        miss = filtered_df.loc[df['rang'] == i, 'name'].tolist()[0]
        rank[miss] = i
    return rank

#Option 1
def define_tab1(tab1):
    # Récupération des données
    data_missFrance = pd.read_csv('./Databases/data_missFrance.csv', delimiter=';')
    data_missFrance_copy = data_missFrance.copy()
    data_missFrance = data_missFrance.drop(["audience", "name", "image"], axis=1)

    annee_test = 2019
    X_train, X_test, y_train, y_test = ds.data_split(data_missFrance, annee_test)
    filtered_df = data_missFrance_copy[data_missFrance_copy['annee'] == annee_test]
    list_candidate = filtered_df['name'].tolist()
    list_region = filtered_df['region'].tolist()

    # Récupération du column_transformer
    path_ct = './Modele/train/column_transformer.pkl'
    ct = pickle.load(open(path_ct, 'rb'))
    X_train = ct.transform(X_train)
    X_test = ct.transform(X_test)

    # Récupération du modèle
    list_of_models = []
    for i in range(12):
        path = './Modele/train/model_' + str(i) + '.pkl'
        model = pickle.load(open(path, 'rb'))
        list_of_models.append(model)
    myModel = my_model.MyModel(list_of_models)

    # Affichage des prédictions
    prediction = myModel.predict(X_test, list_candidate)
    real_rank = give_real_rank(data_missFrance_copy, annee_test)
    i = 0
    while(i<12):
        columns = tab1.columns([1, 1, 1, 1])
        for j in range(4):
            name = prediction[i+1]
            #url_image = filtered_df[filtered_df['name'] == name]['image'].tolist()[0]
            path_image = 'Sources/Images_Candidates_2019/' + name + '.webp'
            print("URL: ", path_image)
            columns[j].image(path_image)
            columns[j].write(str(i+1)+": "+name)
            if(name in real_rank):
                columns[j].write("Réel rang : "+ str(real_rank[name]))
            else:
                columns[j].write("Réel rang : non classée")
            i+=1
            tab1.write("")
    tab1.write("score de prédiction:")
    tab1.write(myModel.evaluate_prediction(X_test, list_candidate, real_rank))


#Option 2
def define_tab2(tab2):
    data_missFrance = pd.read_csv('./Databases/data_missFrance.csv', delimiter=';')
    list_region = sorted(list(set(data_missFrance['region'].tolist())))
    nb_candidates = tab2.number_input("Choisir le nombre de candidates", 1, 30)
    tab2.write("Vous avez choisi " + str(nb_candidates) + " candidates ")
    tab2.write("")
    tab2.write("")
    tab2.write("")
    for i in range(nb_candidates):
        with tab2.form(key="my_form_"+str(i)):
            columns_infos_generales = tab2.columns([2, 2, 1, 1])
            name = columns_infos_generales[0].text_input("Nom de la candidate")
            region = columns_infos_generales[1].selectbox("Région", list_region)
            age = columns_infos_generales[2].number_input("Âge", 18, 40)
            taille = columns_infos_generales[3].slider("Taille (cm)", 130, 200)

            columns_caracteristiques_physiques = tab2.columns([1, 1, 1, 1])
            couleur_cheveux = columns_caracteristiques_physiques[0].selectbox("Couleur des cheveux", ["Noirs", "Bruns", "Chatains", "Roux", "Blonds"])
            longueur_cheveux = columns_caracteristiques_physiques[1].selectbox("Longueur des cheveux", ["Longs", "Mi-longs", "Courts"])
            couleur_yeux = columns_caracteristiques_physiques[2].selectbox("Couleur des yeux", ["Noirs", "Marrons", "Gris", "Bleus", "Verts"])
            couleur_peau = columns_caracteristiques_physiques[3].selectbox("Couleur de la peau", ["Noire", "Métisse", "Blanche"])

            column_autre = tab2.columns([1, 1])
            laureate_culture_generale = column_autre[0].radio("La candidate a-t-elle eu le prix de culture générale ? ", ["Oui", "Non"])
            est_tombee = column_autre[1].radio("La candidate est-elle tombée le soir de l'éléction ? ", ["Oui", "Non"])

            submit_button = tab2.form_submit_button("Enregistrer")
def page_prediction():
    # Logo en haut à droite
    col1, col2, col3 = st.columns((1, 4, 1))
    with col3:
        st.image(chemin_logo, use_column_width=True, width=10)
    st.title("Prediction")
    st.write("Vous pouvez choisir parmi deux options.")
    st.write("Premièrement, vous pouvez voir les prédictions de notre année (l'année 2019).")
    st.write("Deuxièmement, vous pouvez entrer manuellement les données des candidates pour Miss France 2025 et obtenir les prédictions.")
    tab1, tab2 = st.tabs(["Option 1", "Option 2"])
    tab1.write("Première option : voir les prédictions de miss france 2019")
    define_tab1(tab1)
    tab2.write("Seconde option : voir les prédictions de miss France 2025 (données à entrer manuellement)")
    define_tab2(tab2)