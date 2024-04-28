import math
import numpy as np
import requests
from bs4 import BeautifulSoup
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import SGDClassifier
from sklearn.preprocessing import StandardScaler, OrdinalEncoder, OneHotEncoder



#Récupération des données
data_missFrance = pd.read_csv('../Databases/data_missFrance.csv', delimiter=';')
data_model = data_missFrance.drop(["audience", "name", "image"], axis=1)

#Order for the ordinal encoding
hair_length_order = ["Longs", "Mi-longs", "Courts"]
hair_color_order = ["Noirs", "Bruns", "Chatains", "Roux", "Blonds"]
skin_color_order =["Noire", "Métisse", "Blanche"]
eye_color_order = ["Noirs", "Marrons", "Gris", "Bleus",  "Verts"]
categories = [hair_length_order, hair_color_order, eye_color_order, skin_color_order]

#Preprocessing
ct = ColumnTransformer(
    [
     ("preprocessing_rang", OneHotEncoder(handle_unknown="ignore"), ["rang"]),
     ("preprocessing_age", StandardScaler(), ["age", "taille"]),
     ("preprocessing_OrdinalEncoder", OrdinalEncoder(categories=categories), ['longueur_cheveux', 'couleur_cheveux', 'couleur_yeux', 'couleur_peau']),
     ("preprocessing_general_knowledge_test", SimpleImputer(strategy="constant", fill_value=0), ["laureat_culture_generale"]),
     ("preprocessing_one_hot_encoder", OneHotEncoder(handle_unknown="ignore"), [ "annee", "region"])
    ],
    remainder="passthrough"
)

data_model = ct.fit_transform(data_model)

#Mise en place des colonnes
columns = []
for i in range(1,13):
    columns.append("is_"+str(i))
columns+=["is_unranked", "age","length","hair_lenght", "hair_color", "eye_color", "skin_color","general_knowledge_test"]
for i in range(2009, 2025):
    columns.append("year_"+str(i))
region = 1
while(len(columns)<88):
    columns.append("region_"+str(region))
    region += 1

#print(data_model)
df_new_data = pd.DataFrame.sparse.from_spmatrix(data_model, columns=columns)
#print(df_new_data)


#D'abord on ferait le preprocessing (sans pipeline ?)
#cross validation ici (pour trouver quel est le meilleur modèle)
#for i in range(1,12):
    #faire nos 12 modèles (tester à chaque fois quel est le meilleur modèle / meilleur hyperparamètre ou faire tout le temps le même ?)


class MyModel(object):
    def __init__(self, model= [SGDClassifier(loss="log", penalty="")]*12, mydata=[]):
        self.model = model
        self.mydata = mydata

    def fit(self):
        # Faire nos 12 modèles (tester à chaque fois quel est le meilleur modèle / meilleur hyperparamètre ou faire tout le temps le même ?)
        list_columns_y = ["is_unranked"]+["is_"+str(i) for i in range (1,13)]
        X = self.mydata.drop(columns=list_columns_y)
        for i in range(1,13):
            y=X["is_"+str(i)]
            # Séparation train set & test set
            self.model[i].fit(X,y)
            #grid search


    #Renvoyer la matrice d
    #e prédiction (celle avec toutes les probas)
    def predict(self, X):
        result = np.array()
        for candidate in range(len(X)):
            list_candidate = []
            for i in range(1,13):
                y_pred = self.model[i].predict_proba(X[candidate]) #Renvoie un vecteur de probabilités pour mes deux classes de chacun de mes modèles (oui ou non)
                list_candidate.append(y_pred[0])
            result.append(list_candidate)
        return result





classement_test= np.array([[0.1, 0.2, 0.3, 0.4],[0.3, 0.6, 0.9, 0.2],[0.9, 0.7, 0.8, 0.3],[0.4, 0.3, 0.1, 0.1],[0.8, 0.7, 0.9, 0.6]])
list_candidate = ["Lea", "Ana", "Shirelle", "Jenna", "Shana"]
print(give_rank({}, classement_test, list_candidate))
