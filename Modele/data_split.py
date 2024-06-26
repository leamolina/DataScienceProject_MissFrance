import numpy as np


# Séparation X et y (et ajout des colonnes de top)
def transform_data(df, column):
    new_df = df.drop(columns=[column])  # Enlève la colonne "rang"
    for i in range(1, 13):
        new_df['top_' + str(i)] = (df[column] <= i).astype(int)
    return new_df


# Séparation de jeu de données d'entraînement et de test (year_test = jeu de données d'entraînement)
def data_split(data_missFrance, year_test):

    df = transform_data(data_missFrance, 'rang')
    # Séparation des données :
    list_columns_y = ['top_' + str(i) for i in range(1, 13)]
    df_copy = df.copy()
    X = df.drop(columns=list_columns_y, axis=1)  # X = Tout sauf les colonnes de y
    y = [df_copy[column].tolist() for column in list_columns_y[:]]  # y= Toutes les colonnes de y
    y = np.array(y)

    # Séparation train & test
    # Notre test_set correspond aux données de l'année annee_test
    indices_test = X.index[X['annee'] == year_test].tolist()
    indices_train = X.index[X['annee'] != year_test].tolist()

    # Sélection des données correspondantes en utilisant les indices
    X_test = X.iloc[indices_test]
    X_train = X.iloc[indices_train]
    y_test = [[] for _ in range(12)]
    y_train = [[] for _ in range(12)]
    for i in range(12):
        y_train[i] = y[i][indices_train]
        y_test[i] = y[i][indices_test]
    return X_train, X_test, y_train, y_test
