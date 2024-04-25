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


def give_rank_bis(pred_rank_matrix, list_candidate):
    rank={}
    for i in range(5):
        list_sum_proba = []
        for k in range(len(list_candidate)):
            sum_proba = 0
            for j in range(i, 5):
                sum_proba+=pred_rank_matrix[k][j]
            for j in range(0,i):
                sum_proba -= pred_rank_matrix[k][j]
            list_sum_proba.append(sum_proba)
        rank[i] = list_sum_proba

    return rank

def give_rank(initial_rank, pred_rank_matrix, list_candidate):
    ranks = []

    # Base case :
    if len(initial_rank) >= 13 or len(initial_rank) >= len(list_candidate)+1:
        ranks.append(initial_rank)
        return ranks

    else:
        max_rank = np.max(pred_rank_matrix)
        index = np.where(pred_rank_matrix == max_rank)
        if len(index) > 0:
            i = 0
            while(len(index)>0 and i<len(index[0])):
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

                    #Suppression de l'indice
                    index_0 = index[0]
                    index_1 = index[1]
                    index_0 = np.delete(index_0, i)
                    index_1 = np.delete(index_1, i)
                    index = np.array([index_0, index_1])

                # Si plusieurs personnes se trouvent sur la même colonne
                else:
                    j = 0
                    while(len(index)>0 and j<len(index[1])):

                        if index[1][j] == personne_index[1]:
                            new_rank = initial_rank.copy() # On crée une copie du dictionnaire qui stocke les rangs
                            new_rank[list_candidate[index[0][j]]] = index[1][j] + 1
                            new_rank["score"] += max_rank
                            new_pred_rank_matrix = pred_rank_matrix.copy()  # On crée une copie de la matrice de prédiction
                            new_pred_rank_matrix[index[0][j], :] = -1 # On met la ligne de la candidate à -1
                            new_pred_rank_matrix[:, index[1][j]] = -1 # On met la colonne du rang à -1
                            ranks.extend(give_rank(new_rank, new_pred_rank_matrix, list_candidate))

                            #Suppression ici de l'indice de la candidate
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

#Fontion qui parcourt toutes les combinaisons e classement et renvoie la plus probable (celle dont la somme des probabilités est la plus élevée)
def give_best_rank(pred_rank_matrix, list_candidate):
    max_score = 0
    list_dico = give_rank({'score':0}, pred_rank_matrix, list_candidate)
    best_dico = {}
    for dico in list_dico:
        if dico['score'] >= max_score:
            max_score = dico['score']
            best_dico = dico
    return best_dico


list_candidate = ["Ana", "Shirelle", "Léa", "Jenna", "Shana"]
pred_rank_matrix = [[0.1, 0.2, 0.3, 0.4, 0.5], [0.6, 0.7, 0.8, 0.1, 0.2], [0.4, 0.3, 0.1, 0.1, 0.2], [0.5, 0.2, 0.9, 0.8, 0.3], [0.7, 0.1, 0.3, 0.3, 0.1]]
print(give_rank_bis(pred_rank_matrix, list_candidate))

