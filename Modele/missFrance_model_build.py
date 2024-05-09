import pickle
import time

import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, KFold
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

from data_split import data_split
from column_transformer import column_transformer
from multiModelTop12Predictor import MultiModelTop12Predictor

# Récupération des données
data_missFrance = pd.read_csv('../Databases/data_missFrance.csv', delimiter=';')
data_missFrance_copy = data_missFrance.copy()
data_missFrance = data_missFrance.drop(['audience', 'name', 'image'], axis=1)
annee_test = 2019
X_train, X_test, y_train, y_test = data_split(data_missFrance, annee_test)
nb_regions = len(set(X_train['region']))

# Preprocessing
# Récupération de la liste des noms des candidates de 2024:
filtered_df = data_missFrance_copy[data_missFrance_copy['annee'] == annee_test]
list_candidate = filtered_df['name'].tolist()

# Récupération du column transformer
column_transformer(data_missFrance, X_train)
path_ct = 'train/column_transformer.pkl'
ct = pickle.load(open(path_ct, 'rb'))
X_train = ct.transform(X_train)


# Grid Search:

# On a choisi des classifiers qui ont comme paramètres le poids des classes (utile dans notre cas)
models = [DecisionTreeClassifier(), RandomForestClassifier(), SVC(), LogisticRegression()]
dico_decisionTree = {'class_weight': ['balanced'], 'max_features': ['sqrt', 'log2'], 'max_depth': [7, 8, 9], 'random_state': [0]}
dico_randomForest = {'class_weight': ['balanced'], 'n_estimators': [200, 500, 700, 1000], 'max_features': ['sqrt', 'log2'], 'max_depth': [4, 5, 6, 7, 8, 9, 10]}
dico_svc = {'class_weight': ['balanced'], 'C': [1, 2, 3, 4, 5, 10, 20, 50, 100, 200], 'gamma': [1, 0.1, 0.001, 0.0001], 'kernel': ['linear', 'rbf'], 'probability': [True], 'random_state': [0]}
dico_logistic = {'class_weight': ['balanced'], 'C': [0.001, 0.01, 1, 10, 100], 'random_state': [0], 'max_iter': [1000]}
list_params = [dico_decisionTree, dico_randomForest, dico_svc, dico_logistic]

# Option 1 : juste prendre 1 Modele
time_start = time.time()
best_score = 0

# On parcourt tous les modèles et on cherche celui qui donne le meilleur score:
for j in range(len(models)):
    # Grid Search
    cv = KFold(n_splits=5, shuffle=True)
    clf = GridSearchCV(estimator=models[j], param_grid=list_params[j], cv=cv).fit(X_train, y_train[1])
    score = clf.best_score_
    if score > best_score:
        best_score = score
        best_model = clf.best_estimator_
        best_params = clf.best_params_
time_end = time.time()
print('Fin option 1; il s\'agit de ', best_model)
print('L\'option 1 a duré ', time_end - time_start, ' secondes')

# Lancement du Modele:
model = MultiModelTop12Predictor([best_model.__class__(**best_params) for i in range(12)])
model.fit(X_train, y_train)

print('Dump model')
model.dump_model('train/model_')
