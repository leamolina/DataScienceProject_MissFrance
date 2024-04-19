import math
import numpy as np
import time
import requests
from bs4 import BeautifulSoup
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import SGDClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier

from sklearn.preprocessing import StandardScaler, OrdinalEncoder, OneHotEncoder




#Récupération des données
data_missFrance = pd.read_csv('data_missFrance.csv', delimiter=';')
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

    def fit(self, X_train, y_train):
        for i in range(1,13):
            self.model[i].fit(X_train,y_train[i])



    #Renvoyer la matrice de prédiction (celle avec toutes les probas)
    def predict(self, X):
        result = np.array()
        for candidate in range(len(X)):
            list_candidate = []
            for i in range(1,13):
                y_pred = self.model[i].predict_proba(X[candidate]) #Renvoie un vecteur de probabilités pour mes deux classes de chacun de mes modèles (oui ou non)
                list_candidate.append(y_pred[0])
            result.append(list_candidate)
        return result


def give_rank(initial_rank, pred_rank_matrix, list_candidate):
    ranks = []

    # Base case :
    if len(initial_rank) >= 5 or len(initial_rank) >= len(list_candidate)+1:
        ranks.append(initial_rank)
        return ranks

    else:
        max_rank = np.max(pred_rank_matrix)
        index = np.where(pred_rank_matrix == max_rank)
        if len(index) > 0:
            i = 0
            while(len(index[0])>0):
            #for i in range(len(index[0])):
                personne_index = (index[0][i], index[1][i])

                # La personne est seule sur sa colonne
                nb = sum(index[1] == personne_index[1])
                if nb == 1:
                    new_rank = initial_rank.copy()  # Créer une copie indépendante du classement initial
                    new_rank[list_candidate[personne_index[0]]] = personne_index[1] + 1
                    new_rank["score"]+=max_rank
                    new_pred_rank_matrix = pred_rank_matrix.copy()  # Créer une copie indépendante de la matrice de prédiction
                    new_pred_rank_matrix[personne_index[0], :] = -1
                    new_pred_rank_matrix[:, personne_index[1]] = -1
                    ranks.extend(give_rank(new_rank, new_pred_rank_matrix, list_candidate))
                    index_0 = index[0]
                    index_1 = index[1]
                    index_0 = np.delete(index_0, i)
                    index_1 = np.delete(index_1, i)
                    index = np.array([index_0, index_1])


                # Si plusieurs personnes se trouvent sur la même colonne
                else:
                    j = 0
                    while(len(index[1] > 0 )):
                    #for j in range(len(index[1])):
                        if index[1][j] == personne_index[1]:
                            new_rank = initial_rank.copy() # On crée une copie du dictionnaire qui stocke les rangs
                            new_rank[list_candidate[index[0][j]]] = index[1][j] + 1
                            new_rank["score"] += max_rank
                            new_pred_rank_matrix = pred_rank_matrix.copy()  # On crée une copie de la matrice de prédiction
                            new_pred_rank_matrix[index[0][j], :] = -1 # On met la ligne de la candidate à -1
                            new_pred_rank_matrix[:, index[1][j]] = -1 # On met la colonne du rang à -1
                            ranks.extend(give_rank(new_rank, new_pred_rank_matrix, list_candidate))
                            index_0 = index[0]
                            index_1 = index[1]
                            index_0 = np.delete(index_0, j)
                            index_1 = np.delete(index_1, j)
                            index = np.array([index_0, index_1])
                        j+=1
                i+=1

    if not ranks:
        ranks.append(initial_rank)

    return ranks

#Fontion qui donne le meilleur classement et sans répétitions
def give_best_rank(pred_rank_matrix, list_candidate):
    max_score = 0
    list_dico = give_rank({'score':0}, pred_rank_matrix, list_candidate)
    best_dico = {}
    for dico in list_dico:
        if dico['score'] >= max_score:
            max_score = dico['score']
            best_dico = dico
    return best_dico


classement_test= np.array([[0.1, 0.2, 0.3, 0.4],[0.3, 0.6, 0.9, 0.2],[0.9, 0.7, 0.8, 0.3],[0.4, 0.3, 0.1, 0.1],[0.8, 0.7, 0.9, 0.6]])
list_candidate = ["Lea", "Ana", "Shirelle", "Jenna", "Shana"]
#print(give_best_rank(classement_test, list_candidate))

print(give_rank({'score':0}, classement_test, list_candidate))