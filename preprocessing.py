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
            print("voici le score : ", self.model[i].score(X_train, y_train[i]))

    def fitBis(self, X_train, y_train):
        i = 0
        while (i < 12):
            # Mélange les données
            sfk = StratifiedKFold(n_splits=12)
            for train_index, test_index in sfk.split(X_train, y_train[i]):
                X_train_split, X_test_split = X_train[train_index], X_train[test_index]
                y_train_split, y_test_split = y_train[i][train_index], y_train[i][test_index]

                self.model[i].fit(X_train_split, y_train_split)
                print("Voici le score : ", self.model[i].score(X_test_split, y_test_split))
                i += 1


    #Renvoyer la matrice de prédiction (celle avec toutes les probas)

    def predictBis(self, X):
        result = []
        for i in range(12):
            y_pred_real = self.model[i].predict(X)
            print("voici ce qu'on prédit : ", y_pred_real)
            y_pred = self.model[i].predict_proba(X)
            sublist = []
            for j in range(len(y_pred)):
                sublist.append(y_pred[j][1])
            result.append(sublist)
        return np.array(result).T

    def predict(self, X):
        result = np.array([])
        for candidate in range(len(X)):
            list_candidate = []
            for i in range(12):
                y_pred = self.model[i].predict_proba(X[candidate]) #Renvoie un vecteur de probabilités pour mes deux classes de chacun de mes modèles (oui ou non)
                list_candidate.append(y_pred[1])
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

#Séparation des données :
list_columns_y = ["top_" + str(i) for i in range(1, 13)]
df_new_data_copy = df_new_data.copy()
X = df_new_data.drop(columns=list_columns_y, axis=1).values.tolist() # X = Tout sauf les colonnes de y
y = [df_new_data_copy[column].tolist() for column in list_columns_y[:]] # y= Toutes les colonnes de y
X = np.array(X)
y = np.array(y)


#La quinziième colonne de notre dataset correspond à 'year_2024', qui vaut 1 quand on est dans l'année 2024, 0 sinon
#Notre test_set correspond aux données de l'année 2024, notre train_set correspond aux données des années 2009 à 2023
indices_test = np.where(X[:, 14] == 1)[0]
indices_train = np.where(X[:, 14] == 0)[0]

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



#Grid Search
"""
DecisionTreeClassifier —> weight & predict_proba (parameters)
RandomForestClassifier —> weight & predict_proba (parameters)
SVC —> weight & predict_proba (parameters) 
LogisticRegression —> weight & predict_proba (parameters)
"""


#On a choisi des classifiers qui ont comme paramètres le poids des classes (utile dans notre cas)
models = [DecisionTreeClassifier(),RandomForestClassifier(), SVC(), LogisticRegression()]
dico_decisionTree = {'class_weight':['balanced'], 'max_features': ['sqrt', 'log2'], 'max_depth' : [7, 8, 9], 'random_state' :[0]}
dico_randomForest = {'class_weight':['balanced'], 'n_estimators': [200, 500, 700, 1000], 'max_features': ['sqrt', 'log2'],'max_depth' : [4,5,6,7,8,9,10]}
dico_svc = {'class_weight':['balanced'],'C':[1,2, 3, 4, 5, 10, 20, 50, 100, 200],'gamma':[1,0.1,0.001,0.0001],'kernel':['linear','rbf'], 'probability':[True], 'random_state' :[0]}
dico_logistic = {'class_weight':['balanced'],'C':[0.001, 0.01, 1, 10, 100], 'random_state' :[0]}
list_params = [dico_decisionTree, dico_randomForest, dico_svc, dico_logistic]


""""#à lancer devant le prof
models = [RandomForestClassifier()]
dico_randomForest = {'class_weight':['balanced'], 'n_estimators': [200, 500],'max_depth' : [4,5,6]}
list_params = [dico_randomForest]"""

#Option 1 : juste prendre 1 modele
start = time.time()
best_score = 0
y = df_new_data["top_1"]
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

"""
#Option 2 : tester sur tous les modèles
start = time.time()
best_score = 0
for j in range(len(models)):
    sum_scores = 0
    for i in range(1, 13):
        # Grid Search
        clf = GridSearchCV(estimator=models[j], param_grid=list_params[j]).fit(X_train, y_train[i-1])
        score = clf.best_score_
        sum_scores += score
    sum_scores = sum_scores/12
    print("on est dans le modele ", models[j], "et le best_score est ",sum_scores)
    if (sum_scores > best_score):
        best_score = sum_scores
        best_model = clf.best_estimator_
        best_params = clf.best_params_
end = time.time()
print("Fin option 2 qui a duré ", end-start, "secondes", "et le modele est ", best_model) #Environs 28 secondes --> Maintenant c'est passé à 14 secondes : Catastrophe j'ai l'impression qu'il apprend le modèle par coeur
"""
"""
#Option 3 : tester sur tous les modèles & renvoyer une liste de 12 modèles
start = time.time()
best_score = 0
list_best_models = []
for i  in range(1,13):
    sum_scores = 0
    for j in range(len(models)):
        # Grid Search
        clf = GridSearchCV(estimator=models[j], param_grid=list_params[j]).fit(X_train, y_train[1])
        score = clf.best_score_
        if (score > best_score):
            best_score = score
            best_model = clf.best_estimator_
            best_params = clf.best_params_
    list_best_models.append(best_model)

end = time.time()
print("Fin option 3 qui a duré ", end-start, "secondes", "et le modele est ", list_best_models) #Environs 28 secondes --> Maintenant c'est passé à 14 secondes : Catastrophe j'ai l'impression qu'il apprend le modèle par coeur
"""


classement_test= np.array([[0.1, 0.2, 0.3, 0.4],[0.3, 0.6, 0.9, 0.2],[0.9, 0.7, 0.8, 0.3],[0.4, 0.3, 0.1, 0.1],[0.8, 0.7, 0.9, 0.6]])
list_candidate = ["Lea", "Ana", "Shirelle", "Jenna", "Shana"]
#print(give_best_rank(classement_test, list_candidate))



#Création de notre modèle
myModel = MyModel([best_model.__class__(**best_params) for i in range(12)])
#myModel = MyModel(list_best_models)
#print(best_model.__class__(**best_params))
myModel.fit(X_train, y_train)
prediction_matrix = myModel.predictBis(X_test)
print(prediction_matrix.shape)
for i in range(len(prediction_matrix)):
    print(prediction_matrix[i])
    print("\n\n\n")


