import pickle

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler, OrdinalEncoder

from data_split import data_split

# Récupération des données
data_missFrance = pd.read_csv('../Databases/data_missFrance.csv', delimiter=';')
data_missFrance_copy = data_missFrance.copy()
data_missFrance = data_missFrance.drop(['audience', 'name', 'image'], axis=1)

annee_test = 2019
X_train, X_test, y_train, y_test = data_split(data_missFrance, annee_test)

# Preprocessing

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
    ('preprocessing_OrdinalEncoder', OrdinalEncoder(categories=categories), ['longueur_cheveux', 'couleur_cheveux',
                                                                             'couleur_yeux', 'couleur_peau']),
    ('preprocessing_general_knowledge_test', SimpleImputer(strategy='constant', fill_value=0), ['laureat_culture_generale']),
    ],
    remainder='passthrough')

ct.fit(X_train)

# Dump du column transformer
name_file_ct = f'train/column_transformer.pkl'
pickle.dump(ct, open(name_file_ct, 'wb'))
