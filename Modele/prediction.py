import math

from joblib import dump, load
import pandas as pd
import numpy as np
from data_split import data_split


#Fonctions diverses :
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

if __name__ == '__main__':

    # Récupération des données
    data_missFrance = pd.read_csv('../Databases/data_missFrance.csv', delimiter=';')
    data_missFrance_copy = data_missFrance.copy()
    data_missFrance = data_missFrance.drop(["audience", "name", "image"], axis=1)

    annee_test = 2019
    X_train, X_test, y_train, y_test = data_split(data_missFrance, annee_test)
    filtered_df = data_missFrance_copy[data_missFrance_copy['annee'] == annee_test]
    list_candidate = filtered_df['name'].tolist()

    myModel = load('myModelRanking.joblib')
    ct = load('column_transformer.joblib')
    X_test = ct.transform(X_test)

    prediction = myModel.predict(X_test, list_candidate)
    real_rank = give_real_rank(data_missFrance_copy, annee_test)
    print("prediction : ", prediction)
    print("vrai classement :", real_rank)
    print("score de prédiction lea :", evaluate_prediction(prediction, real_rank))

    """
    scores (Ana / Léa) :
    2024: 1725 (Eve pas classée) / 1765 (Eve pas classée)
    2023: 3259 (Indira pas classée) / 3235 (Indira pas classée)
    2022: 1780 (Diane 9eme) / 1741 (Diane 7ème) 
    2021: 2509 (Amandine 10eme) / 2563 (Amandine 10eme)
    2020: 2529 (Clémence 5ème) / 2555 (Clémence 1ere)
    2019: 1658 (Vaimala 2ème) / 1666 (Vaimala 1ere)
    2018: 2891 (Maeva 1ère) / 2868 (Maeva 1ère)
    2017: 2634 (Alycia 7ème) / 2615 (Alycia 8ème)
    2016: 2870 (Iris 5eme) / 2859 (Iris 3eme)
    2015: 2987 (Camille 2ème) /2994 (Camille 3eme)
    2014: 2447 (Flora 5eme) / 2465 (Flora 3eme)
    2013: 2472 (Marine 5ème) / 2515 (Marine 7eme)
    2012: 2530 (Delphine 7eme) / 2501 (Delphine 5eme)
    2011: 2074 (Laury 3eme) / 2046 (Laury 2eme)
    2010: 2831 (Malika 2ème) 2841 (Malika 5eme)
    2009: 3281 (Chloé 9ème)  / 3298 (Chloé 10eme)

    somme des scores:
    Ana = 40477
    Léa = 40527
    """
