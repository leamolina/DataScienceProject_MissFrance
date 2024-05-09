import time

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import OneHotEncoder, StandardScaler, OrdinalEncoder

from data_split import data_split
from multiModelTop12Predictor import MultiModelTop12Predictor


def give_real_rank_bis(df, annee):
    filtered_df = df[df['annee'] == annee]
    rank = {}
    for i in range(1, 13):
        miss = filtered_df.loc[df['rang'] == i, 'name'].tolist()[0]
        rank[i] = miss
    return rank


# Récupération des données
data_missFrance = pd.read_csv('../Databases/data_missFrance.csv', delimiter=';')
data_missFrance_copy = data_missFrance.copy()
data_missFrance = data_missFrance.drop(['audience', 'name', 'image'], axis=1)

for annee_test in [2018,2019, 2020]:
    X_train, X_test,y_train,y_test = data_split(data_missFrance, annee_test)
    nb_regions = len(set(X_train['region']))

    #Preprocessing
    #Récupération de la liste des noms des candidates de 2024:
    filtered_df = data_missFrance_copy[data_missFrance_copy['annee'] == annee_test]
    list_candidate = filtered_df['name'].tolist()

    #Récupération du column transformer
    # Order for the ordinal encoding
    hair_length_order = ['Longs', 'Mi-longs', 'Courts']
    hair_color_order = ['Noirs', 'Bruns', 'Chatains', 'Roux', 'Blonds']
    skin_color_order = ['Noire', 'Métisse', 'Blanche']
    eye_color_order = ['Noirs', 'Marrons', 'Gris', 'Bleus', 'Verts']
    categories = [hair_length_order, hair_color_order, eye_color_order, skin_color_order]

    # Créer un ColumnTransformer avec votre Custom_OneHotEncoder et d'autres transformations
    ct = ColumnTransformer([
        ('preprocessing_one_hot_encoder', OneHotEncoder(handle_unknown='ignore'), ['annee', 'region']),
        ('preprocessing_age', StandardScaler(), ['age', 'taille']),
        ('preprocessing_OrdinalEncoder', OrdinalEncoder(categories=categories),
         ['longueur_cheveux', 'couleur_cheveux', 'couleur_yeux', 'couleur_peau']),
        ('preprocessing_general_knowledge_test', SimpleImputer(strategy='constant', fill_value=0),
         ['laureat_culture_generale']),
    ],
        remainder='passthrough')

    ct.fit(X_train)

    X_train = ct.transform(X_train)
    X_test = ct.transform(X_test)

    #Mise en place des colonnes
    columns = []
    for i in range(2009, 2025):
        if(i!=annee_test):
            columns.append('year_'+str(i))
    for i in range(1,nb_regions+1):
        columns.append('region_' + str(i))
    columns+=['age','length','hair_lenght', 'hair_color', 'eye_color', 'skin_color','general_knowledge_test', 'has_fallen']


    #Grid Search:
    #à lancer devant le prof
    models = [RandomForestClassifier()]
    dico_randomForest = {'class_weight':['balanced'], 'n_estimators': [200, 500],'max_depth' : [4,5,6]}
    list_params = [dico_randomForest]


    #Option 1 : juste prendre 1 Modele
    time_start = time.time()
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
    time_end = time.time()
    print('Fin option 1; il s\'agit de ', best_model)
    print('L\'option 1 a duré ', time_end - time_start, ' secondes')

    #Lancement du Modele:
    model = MultiModelTop12Predictor([best_model.__class__(**best_params) for i in range(12)])
    model.fit(X_train, y_train)
    prediction = model.predict(X_test, list_candidate)
    real_rank = give_real_rank_bis(data_missFrance_copy, annee_test)
    print("pred", prediction)
    print("real", real_rank)
    fract_cp = model.fraction_of_concordant_pairs(real_rank, prediction)
    rec_rank = model.reciprocal_rank(real_rank, prediction)
    ap_k = model.ap_at_k(real_rank, prediction, 12)
    print("Score pour l'année ", annee_test, " : ", (fract_cp+ap_k+rec_rank)/3)
    print('fract_cp', fract_cp)
    print('rec_rank', rec_rank)
    print('ap_k', ap_k)
    print("Balanced accuracy score : ", model.balanced_accuracy_score(X_test, y_test))
    print("\n\n\n")

print('Coucou')