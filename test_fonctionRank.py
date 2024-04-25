import math
import numpy as np
import time
import requests
from bs4 import BeautifulSoup
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, KFold, StratifiedKFold, cross_val_score
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import SGDClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier

from sklearn.preprocessing import StandardScaler, OrdinalEncoder, OneHotEncoder
from sklearn.tree import DecisionTreeClassifier

#Récupération des données
data_missFrance = pd.read_csv('data_missFrance.csv', delimiter=';')
data_model = data_missFrance.drop(["audience", "name", "image"], axis=1)
nb_regions = len(set(data_model['region']))

#Encoder personnalisé pour récuperer le top
class Custom_OneHotEncoder(OneHotEncoder):
    def __init__(self, column):
        self.column = column

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        #Vérification que column est bien une colonne de notre dataframe
        new_X = X.drop(columns=[self.column])
        for i in range(1, 13):
            new_X["top_" + str(i)] = (X[self.column] <= i).astype(int)
        return new_X


#Order for the ordinal encoding
hair_length_order = ["Longs", "Mi-longs", "Courts"]
hair_color_order = ["Noirs", "Bruns", "Chatains", "Roux", "Blonds"]
skin_color_order =["Noire", "Métisse", "Blanche"]
eye_color_order = ["Noirs", "Marrons", "Gris", "Bleus",  "Verts"]
categories = [hair_length_order, hair_color_order, eye_color_order, skin_color_order]

# Créer un ColumnTransformer avec votre Custom_OneHotEncoder et d'autres transformations
ct = ColumnTransformer([
    ("preprocessing_one_hot_encoder", OneHotEncoder(handle_unknown="ignore"), ["annee", "region"]),
    ("preprocessing_age", StandardScaler(), ["age", "taille"]),
    ("preprocessing_OrdinalEncoder", OrdinalEncoder(categories=categories), ['longueur_cheveux', 'couleur_cheveux', 'couleur_yeux', 'couleur_peau']),
    ("preprocessing_rang", Custom_OneHotEncoder(column="rang"), ["rang"]),
    ("preprocessing_general_knowledge_test", SimpleImputer(strategy="constant", fill_value=0), ["laureat_culture_generale"]),
],
remainder="passthrough")

ct.fit(data_model)
data_model = ct.transform(data_model)

#Mise en place des colonnes
columns = []
for i in range(1,nb_regions+1):
    columns.append("region_" + str(i))
columns+=["age","length","hair_lenght", "hair_color", "eye_color", "skin_color","general_knowledge_test", "has_fallen"]
for i in range(1,13):
    columns.append("top_"+str(i))
for i in range(2009, 2025):
    columns.append("year_"+str(i))


#print(data_model)
df_new_data = pd.DataFrame.sparse.from_spmatrix(data_model, columns=columns)
df_new_data.to_csv('donnee_avec_preprocessing.csv', index=False)
print(df_new_data.head())
