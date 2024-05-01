import math
import numpy as np
import pandas as pd
from joblib import dump, load
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import StandardScaler, OrdinalEncoder, OneHotEncoder
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from data_split import data_split
import json
from my_model import MyModel
import streamlit as st

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


# Récupération des données
data_missFrance = pd.read_csv('../Databases/data_missFrance.csv', delimiter=';')
data_missFrance_copy = data_missFrance.copy()
data_missFrance = data_missFrance.drop(["audience", "name", "image"], axis=1)
annee_test = 2019
X_train, X_test,y_train,y_test = data_split(data_missFrance, annee_test)
nb_regions = len(set(X_train['region']))

#Preprocessing
#Récupération de la liste des noms des candidates de 2024:
filtered_df = data_missFrance_copy[data_missFrance_copy['annee'] == annee_test]
list_candidate = filtered_df['name'].tolist()

X_train, X_test = tranform_data(X_train, X_test)

#Mise en place des colonnes
columns = []
for i in range(2009, 2025):
    if(i!=annee_test):
        columns.append("year_"+str(i))
for i in range(1,nb_regions+1):
    columns.append("region_" + str(i))
columns+=["age","length","hair_lenght", "hair_color", "eye_color", "skin_color","general_knowledge_test", "has_fallen"]

#print(data_model)
df_X_train = pd.DataFrame.sparse.from_spmatrix(X_train, columns=columns)
df_X_train.to_csv('../Databases/donnee_X_train.csv', index=False)

#Grid Search:

#On a choisi des classifiers qui ont comme paramètres le poids des classes (utile dans notre cas)
models = [DecisionTreeClassifier(),RandomForestClassifier(), SVC(), LogisticRegression()]
dico_decisionTree = {'class_weight':['balanced'], 'max_features': ['sqrt', 'log2'], 'max_depth' : [7, 8, 9], 'random_state' :[0]}
dico_randomForest = {'class_weight':['balanced'], 'n_estimators': [200, 500, 700, 1000], 'max_features': ['sqrt', 'log2'],'max_depth' : [4,5,6,7,8,9,10]}
dico_svc = {'class_weight':['balanced'],'C':[1,2, 3, 4, 5, 10, 20, 50, 100, 200],'gamma':[1,0.1,0.001,0.0001],'kernel':['linear','rbf'], 'probability':[True], 'random_state' :[0]}
dico_logistic = {'class_weight':['balanced'],'C':[0.001, 0.01, 1, 10, 100], 'random_state' :[0], 'max_iter': [1000]}
list_params = [dico_decisionTree, dico_randomForest, dico_svc, dico_logistic]

"""
#à lancer devant le prof
models = [RandomForestClassifier()]
dico_randomForest = {'class_weight':['balanced'], 'n_estimators': [200, 500],'max_depth' : [4,5,6]}
list_params = [dico_randomForest]
"""

#Option 1 : juste prendre 1 Modele
best_score = 0

#On parcourt tous les modèles et on cherche celui qui donne le meilleur score:
for j in range(len(models)):
    # Grid Search
    clf = GridSearchCV(estimator=models[j], param_grid=list_params[j]).fit(X_train, y_train[1])
    score = clf.best_score_
    if (score > best_score):
        best_score = score
        best_model = clf.best_estimator_
        best_params = clf.best_params_
print("Fin option 1; il s'agit de ", best_model)

#Lancement du Modele:
print(best_params)
print(best_model.__class__)
model_info = {
    "model_type": best_model.__class__.__name__,
    "params": best_params
}
with open('data.json', 'w') as f:
    json.dump(model_info, f)
myModel = MyModel([best_model.__class__(**best_params) for i in range(12)])
myModel.fit(X_train, y_train)

prediction = myModel.predict(X_test, list_candidate)
real_rank = give_real_rank(data_missFrance_copy, annee_test)
st.write("prediction : ", prediction)
st.write("vrai classement :", real_rank)
st.write("score de prédiction lea :", evaluate_prediction(prediction, real_rank))