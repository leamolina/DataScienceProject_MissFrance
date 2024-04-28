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
data_missFrance = pd.read_csv('Databases/data_missFrance.csv', delimiter=';').drop(["audience", "name", "image"], axis=1)
nb_regions = len(set(data_missFrance['region']))


#Séparation X et y (custom_oneEncoder)
def transform_y(df, column):
    new_df = df.drop(columns=[column])  # Enlève la colonne "rang"
    for i in range(1, 13):
        new_df["top_" + str(i)] = (df[column] <= i).astype(int)
    return new_df

df = transform_y(data_missFrance, "rang")
print(df.head())

#Séparation des données :
list_columns_y = ["top_" + str(i) for i in range(1, 13)]
df_copy = df.copy()
X = df.drop(columns=list_columns_y, axis=1).values.tolist() # X = Tout sauf les colonnes de y
y = [df_copy[column].tolist() for column in list_columns_y[:]] # y= Toutes les colonnes de y
X = np.array(X)
y = np.array(y)


#Séparation train & test

#La quinziième colonne de notre dataset correspond à 'year_2024', qui vaut 1 quand on est dans l'année 2024, 0 sinon
#Notre test_set correspond aux données de l'année 2024, notre train_set correspond aux données des années 2009 à 2023
indices_test = np.where(X[:, 9] == 1)[0]
indices_train = np.where(X[:, 9] == 0)[0]

# Sélection des données correspondantes en utilisant les indices
X_test, X_train = X[indices_test], X[indices_train]
y_test = [[] for _ in range(12)]
y_train = [[] for _ in range(12)]
for i in range(12):
    y_train[i] = y[i][indices_train]
    y_test[i] = y[i][indices_test]

# Vérification des dimensions de X et y
print("Dimensions de X_train:", X_train.shape)
print("Dimensions de X_test:", X_test.shape)
for i in range(12):
    print("Dimensions de y_train[", i, "]", y_train[i].shape)
    print("Dimensions de y_test[", i, "]", y_test[i].shape)



#Preprocessing (sans la transformation de y)

#Récupération de la liste des noms des candidates de 2024:
filtered_df = data_missFrance[data_missFrance['annee'] == 2018]
list_candidate = filtered_df['name'].tolist()

#Encoder personnalisé pour récuperer le top
class Custom_OneHotEncoder(OneHotEncoder):
    def __init__(self, column):
        self.column = column

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        #self.column = "rang"
        new_X = X.drop(columns=[self.column]) #Enlève la colonne "rang"
        #Quand i = 1 , X[rang] <=1
        #Quand i = 3 , X[rang] <= 3
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
for i in range(2009, 2025):
    columns.append("year_"+str(i))
for i in range(1,nb_regions+1):
    columns.append("region_" + str(i))
columns+=["age","length","hair_lenght", "hair_color", "eye_color", "skin_color","general_knowledge_test"]
for i in range(1,13):
    columns.append("top_"+str(i))
columns+=["has_fallen"]

#print(data_model)
df_new_data = pd.DataFrame.sparse.from_spmatrix(data_model, columns=columns)
df_new_data.to_csv('donnee_avec_preprocessing.csv', index=False)

class MyModel(object):
    def __init__(self, model= [SGDClassifier(loss="log", penalty="")]*12):
        self.model = model

    def fit(self, X_train, y_train):
        for i in range(12):
            self.model[i].fit(X_train, y_train[i])

    #Renvoyer la matrice de prédiction (celle avec toutes les probas)
    def predict(self, X):
        result = []
        for i in range(12):
            y_pred_real = self.model[i].predict(X)
            #print("voici ce qu'on prédit : ", y_pred_real)
            y_pred = self.model[i].predict_proba(X)
            sublist = []
            for j in range(len(y_pred)):
                sublist.append(y_pred[j][1])
            result.append(sublist)
        return np.array(result).T

    def score(self, X_test, y_test):
        scores = []
        for i in range(12):
            scores.append(self.model[i].score(X_test, y_test[i]))
        return np.array(scores)


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












#Création de notre modèle
myModel = MyModel([best_model.__class__(**best_params) for i in range(12)])
#myModel = MyModel(list_best_models)
#print(best_model.__class__(**best_params))
myModel.fit(X_train, y_train)
prediction_matrix = myModel.predict(X_test)
"""
print(prediction_matrix.shape)
for i in range(len(prediction_matrix)):
    print(prediction_matrix[i])
    print("\n\n\n")"""

def give_rank_5(prediction_matrix, list_candidate):
    scores = []
    for k in range(len(prediction_matrix)):
        sum_proba = 0
        for j in range(12):
            sum_proba += prediction_matrix[k][j]
        scores.append(sum_proba)
    classement = {}
    for i in range(12):
        max_ = np.argmax(scores)
        classement[i+1] = list_candidate[max_]
        scores[max_] = -1
    return classement



def give_rank_ana(prediction_matrix, list_candidate):
    scores = []
    for k in range(len(prediction_matrix)):
        sum_proba = 0
        for j in range(12):
            sum_proba += prediction_matrix[k][j]
        scores.append(sum_proba)
    prediction_matrix = np.array(prediction_matrix)
    new_prediction_matrix = np.array([])
    #On récupère les 12 meilleures miss (scores) pour filtrer les candidates
    classement = []
    list_indices = []
    for i in range(12):
        max_ = np.argmax(scores)
        classement.append(max_)
        scores[max_] = -1
        list_indices.append(max_)
    #On met tous ceux qui ne sont pas dans le top 12 à -1
    for i in range(len(prediction_matrix)):
        if i not in list_indices:
            prediction_matrix[i, :] = -1
    candidates = {}
    #ANAA :  écrit le commentaire pour ça mercii
    for i in range(12):
        index_max = np.argmax(prediction_matrix[:,i])
        candidates[i+1] = list_candidate[index_max]
        prediction_matrix[index_max, :] = -1
        prediction_matrix[:, i] = -1
    return candidates


print("Give rank de lea : \n" )
print(give_rank_5(prediction_matrix, list_candidate))

print("\n\nGive rank de ana : ")
print(give_rank_ana(prediction_matrix, list_candidate))

print("Voici les scores : ")
print(myModel.score(X_test, y_test))

print("Real rank")
real_rank = give_real_rank(data_missFrance, 2018)
prediction_ana = give_rank_ana(prediction_matrix, list_candidate)
prediction_lea = give_rank_5(prediction_matrix, list_candidate)
print("score ana : ", evaluate_prediction(prediction_ana, real_rank))
print("score lea : ", evaluate_prediction(prediction_lea, real_rank))