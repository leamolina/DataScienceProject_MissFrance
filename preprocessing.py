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

#Récupération de la liste des noms des candidates de 2024:
filtered_df = data_missFrance[data_missFrance['annee'] == 2023]
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
            print("voici le score : ", self.model[i].score(X_train, y_train[i]))

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



#Séparation des données :
list_columns_y = ["top_" + str(i) for i in range(1, 13)]
df_new_data_copy = df_new_data.copy()
X = df_new_data.drop(columns=list_columns_y, axis=1).values.tolist() # X = Tout sauf les colonnes de y
y = [df_new_data_copy[column].tolist() for column in list_columns_y[:]] # y= Toutes les colonnes de y
X = np.array(X)
y = np.array(y)

#2024 = 15
#2023 = 1
#2022 = 12


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
#On a choisi des classifiers qui ont comme paramètres le poids des classes (utile dans notre cas)
models = [DecisionTreeClassifier(),RandomForestClassifier(), SVC(), LogisticRegression()]
dico_decisionTree = {'class_weight':['balanced'], 'max_features': ['sqrt', 'log2'], 'max_depth' : [7, 8, 9], 'random_state' :[0]}
dico_randomForest = {'class_weight':['balanced'], 'n_estimators': [200, 500, 700, 1000], 'max_features': ['sqrt', 'log2'],'max_depth' : [4,5,6,7,8,9,10]}
dico_svc = {'class_weight':['balanced'],'C':[1,2, 3, 4, 5, 10, 20, 50, 100, 200],'gamma':[1,0.1,0.001,0.0001],'kernel':['linear','rbf'], 'probability':[True], 'random_state' :[0]}
dico_logistic = {'class_weight':['balanced'],'C':[0.001, 0.01, 1, 10, 100], 'random_state' :[0]}
list_params = [dico_decisionTree, dico_randomForest, dico_svc, dico_logistic]
"""

#à lancer devant le prof
models = [RandomForestClassifier()]
dico_randomForest = {'class_weight':['balanced'], 'n_estimators': [200, 500],'max_depth' : [4,5,6]}
list_params = [dico_randomForest]


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


print("maintenant on passe à la vraie prédiction")

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


"""
Analyse des classements par année


2024 : 

Léa:
{1: 'Ève Gilles'(1), 2: 'Ravahere Silloux' (7), 3: 'Adélina Blanc'(3), 4: 'Noémie Le Bras'(unrank), 
5: 'Clémence Ménard'(9), 6: 'Agathe Toullieu'(unrank), 7: 'Maxime Teissier'(5), 8: 'Lola Turpin'(unrank), 
9: 'Emma Grousset'(unrank), 10: 'Karla Bchir'(12), 11: 'Mélanie Odules'(unrank), 12: 'Chléo Modestine'(unrank)}

Ana: 
{1: 'Ève Gilles'(1), 2: 'Ravahere Silloux'(7), 3: 'Adélina Blanc'(3), 4: 'Agathe Toullieu' (unrank), 
5: 'Mélanie Odules'(unrank), 6: 'Maxime Teissier'(5), 7: 'Karla Bchir'(12), 8: 'Clémence Ménard'(9), 
9: 'Noémie Le Bras'(unrank), 10: 'Lola Turpin'(unrank), 11: 'Chléo Modestine'(unrank), 12: 'Emma Grousset'(unrank)}


2023 :

Léa:
{1: 'Indira Ampiot' (1), 2: 'Herenui Tuheiava' (unrank), 3: 'Agathe Cauet'(2), 4: 'Chana Goyons'(unrank), 
5: 'Chiara Fontaine'(13), 6: 'Sarah Aoutar'(7), 7: 'Coraline Larasle'(unrank), 8: 'Alissia Ladevèze'(5),
9: 'Camille Sedira'(unrank), 10: 'Lara Lebretton'(unrank), 11: 'Emma Guibert'(12), 12: 'Adèle Bonnamour'(unrank)

Ana:
{1: 'Indira Ampiot'(1), 2: 'Herenui Tuheiava'(unrank), 3: 'Chana Goyons'(unrank), 4: 'Chiara Fontaine'(13), 
5: 'Agathe Cauet'(2), 6: 'Sarah Aoutar'(7), 7: 'Adèle Bonnamour'(unrank), 8: 'Emma Guibert'(12), 
9: 'Lara Lebretton'(unrank), 10: 'Camille Sedira'(unrank), 11: 'Coraline Larasle' (unrank), 12: 'Alissia Ladevèze'(5)}


Léa 

{1: 'Camille Sedira', 2: 'Orianne Galvez-Soto', 3: 'Alissia Ladevèze', 4: 'Lara Lebretton', 5: 'Enora Moal', 6: 'Coraline Larasle', 7: 'Solène Scholer', 8: 'Orianne Meloni', 9: 'Flavy Barla', 10: 'Marion Navarro', 11: 'Indira Ampiot', 12: 'Shaïna Robin'}


Give rank de ana : 
{1: 'Indira Ampiot', 2: 'Alissia Ladevèze', 3: 'Coraline Larasle', 4: 'Lara Lebretton', 5: 'Flavy Barla', 6: 'Orianne Galvez-Soto', 7: 'Camille Sedira', 8: 'Marion Navarro', 9: 'Shaïna Robin', 10: 'Solène Scholer', 11: 'Orianne Meloni', 12: 'Enora Moal'}

"""