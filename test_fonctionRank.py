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

annee_test = 2024
# Récupération des données
data_missFrance = pd.read_csv('data_missFrance.csv', delimiter=';')
data_missFrance_copy = data_missFrance.copy()
data_missFrance = data_missFrance.drop(["audience", "name", "image"], axis=1)


# Séparation X et y (custom_oneEncoder)
def transform_y(df, column):
    new_df = df.drop(columns=[column])  # Enlève la colonne "rang"
    for i in range(1, 13):
        new_df["top_" + str(i)] = (df[column] <= i).astype(int)
    return new_df


df = transform_y(data_missFrance, "rang")

# Séparation des données :
list_columns_y = ["top_" + str(i) for i in range(1, 13)]
df_copy = df.copy()
X = df.drop(columns=list_columns_y, axis=1) # X = Tout sauf les colonnes de y
y = [df_copy[column].tolist() for column in list_columns_y[:]]  # y= Toutes les colonnes de y
y = np.array(y)

# Séparation train & test
# Notre test_set correspond aux données de l'année 2022


indices_test = X.index[X['annee'] == annee_test].tolist()
indices_train = X.index[X['annee'] != annee_test].tolist()


# Sélection des données correspondantes en utilisant les indices
X_test = X.iloc[indices_test]
X_train = X.iloc[indices_train]
nb_regions = len(set(X_train['region']))


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


#Preprocessing
#Récupération de la liste des noms des candidates de 2024:
filtered_df = data_missFrance_copy[data_missFrance_copy['annee'] == annee_test]
list_candidate = filtered_df['name'].tolist()

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
    ("preprocessing_general_knowledge_test", SimpleImputer(strategy="constant", fill_value=0), ["laureat_culture_generale"]),
],
remainder="passthrough")

ct.fit(X_train)
X_train = ct.transform(X_train)
X_test = ct.transform(X_test)

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
df_X_train.to_csv('donnee_X_train.csv', index=False)


#Grid Search:
"""
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


#Option 1 : juste prendre 1 modele
start = time.time()
best_score = 0
#On parcourt tous les modèles et on cherche celui qui donne le meilleur score
for j in range(len(models)):
    # Grid Search
    clf = GridSearchCV(estimator=models[j], param_grid=list_params[j]).fit(X_train, y_train[1])
    score = clf.best_score_
    if (score > best_score):
        best_score = score
        best_model = clf.best_estimator_
        best_params = clf.best_params_
end = time.time()
print("Fin option 1 qui a duré " , end-start, "secondes", "il s'agit de ", best_model) #Environs 2.5 secondes

class MyModel(object):
    def __init__(self, model= [SGDClassifier(loss="log", penalty="")]*12):
        self.model = model

    def fit(self, X_train, y_train):
        for i in range(12):
            self.model[i].fit(X_train, y_train[i])


    #Renvoyer la matrice de prédiction (celle avec toutes les probas)
    def prediction_matrix(self, X):
        result = []
        for i in range(12):
            y_pred_real = self.model[i].predict(X)
            # print("voici ce qu'on prédit : ", y_pred_real)
            y_pred = self.model[i].predict_proba(X)
            sublist = []
            for j in range(len(y_pred)):
                sublist.append(y_pred[j][1])
            result.append(sublist)
        return np.array(result).T


    def predict(self,X, list_candidate):
        prediction_matrix = self.prediction_matrix(X)
        scores = []
        for k in range(len(prediction_matrix)):
            sum_proba = 0
            for j in range(12):
                sum_proba += prediction_matrix[k][j]
            scores.append(sum_proba)
        prediction_matrix = np.array(prediction_matrix)
        new_prediction_matrix = np.array([])
        # On récupère les 12 meilleures miss (scores) pour filtrer les candidates
        classement = []
        list_indices = []
        for i in range(12):
            max_ = np.argmax(scores)
            classement.append(max_)
            scores[max_] = -1
            list_indices.append(max_)
        # On met tous ceux qui ne sont pas dans le top 12 à -1
        for i in range(len(prediction_matrix)):
            if i not in list_indices:
                prediction_matrix[i, :] = -1
        candidates = {}
        # ANAA :  écrit le commentaire pour ça mercii
        for i in range(12):
            index_max = np.argmax(prediction_matrix[:, i])
            candidates[i + 1] = list_candidate[index_max]
            prediction_matrix[index_max, :] = -1
            prediction_matrix[:, i] = -1
        return candidates

    def score(self, X_test, y_test):
        scores = []
        for i in range(12):
            scores.append(self.model[i].score(X_test, y_test[i]))
        return np.array(scores)


#Fonctions diverses:
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


#Lancement du modele
myModel = MyModel([best_model.__class__(**best_params) for i in range(12)])
myModel.fit(X_train, y_train)
prediction = myModel.predict(X_test, list_candidate)
real_rank = give_real_rank(data_missFrance_copy, annee_test)
print(prediction)
print(real_rank)
print("score de prédiction :", evaluate_prediction(prediction, real_rank))
print("Score model", myModel.score(X_test, y_test))


"""
scores:
2024: 1739 (Eve pas classée)
2023: 3259 (Indira pas classée) 
2022: 1749 (Diane 9eme)
2021: 2529 (Amandine 10eme)
2020: 2504 (Clémence 5ème)
2019: 1660 (Vaimala 2ème)
2018: 3617 (Maeva 1ère)
2017: 2618 (Alycia 7ème)
2016: 2831 (Iris 4eme)
2015: 2987 (Camille 2ème)
2014: 2478 (Flora 1ère) 
2013: 2089 (Marine 5ème)
2012: 2507 (Delphine 6eme)
2011: 2127 (Laury 3eme)
2010: 2855 (Malika 2ème)
2009: 3292 (Chloé 10ème)
"""