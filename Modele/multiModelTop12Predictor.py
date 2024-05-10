import pickle

import numpy as np
from sklearn.metrics import balanced_accuracy_score


class MultiModelTop12Predictor(object):
    def __init__(self, model=None):
        self.model = model

    # Entraînement des 12 modèles
    def fit(self, X_train, y_train):
        for i in range(12):
            self.model[i].fit(X_train, y_train[i])

    # Renvoi de la matrice de prédiction (qui contient les probabilités de chaque top)
    def prediction_matrix(self, X):
        result = []
        for i in range(12):
            y_pred = self.model[i].predict_proba(X)
            sublist = []
            for j in range(len(y_pred)):
                sublist.append(y_pred[j][1])
            result.append(sublist)
        return np.array(result).T

    # Prédiction du classement à partir de la matrice de prédiction
    def predict(self, X, list_candidate):
        prediction_matrix = self.prediction_matrix(X)  # Récupération de la matrice de prédiction
        scores = []  # On attribue un score pour chaque candidate : la somme des probabilités obtenues grâce à la matrice de prédiction
        for k in range(len(prediction_matrix)):
            sum_proba = 0
            for j in range(12):
                sum_proba += prediction_matrix[k][j]
            scores.append(sum_proba)
        ranking = {}
        for i in range(min(12, len(list_candidate))):
            max_ = np.argmax(scores)
            ranking[i + 1] = list_candidate[max_]
            scores[max_] = -1
        return ranking

    # Permet de sauvegarder les 12 modèles dans des fichier Pickle (pour ne pas avoir à ré-entraîner le model avant chaque utilisaton)
    def dump_model(self, path):
        for i in range(12):
            name_file_model = path+str(i)+'.pkl'
            pickle.dump(self.model[i], open(name_file_model, 'wb'))

    # Moyenne de la balanced_accuracy_score de chaque modèle
    def balanced_accuracy_score(self, X_test, y_test):
        sum_ = 0
        # Construction de y_pred
        y_pred = []
        for i in range(12):
            pred = list(self.model[i].predict(X_test))
            y_pred.append(pred)
        for i in range(12):
            sum_ += balanced_accuracy_score(y_test[i], y_pred[i])
        return sum_/12

    # Taux de paires concordantes
    def fraction_of_concordant_pairs(self, real_rank, predicted_rank):
        nb_of_concordant_pairs = 0
        nb_of_pairs = 66
        for i in range(1, 13):
            candidate_i = predicted_rank[i]
            # Recherche de la candidate dans le réel classement
            cand = [candidate for rank, candidate in real_rank.items() if candidate == candidate_i]
            if len(cand) > 0:
                real_rank_i = [rank for rank, candidate in real_rank.items() if candidate == candidate_i][0]
                pred_rank_i = i
                for j in range(i+1, 13):
                    candidate_j = predicted_rank[j]
                    # Recherche de la candidate j dans le réel classement
                    cand = [candidate for rank, candidate in real_rank.items() if candidate == candidate_j]
                    if len(cand) == 0:
                        nb_of_concordant_pairs += 1
                    else:
                        real_rank_j = [rank for rank, candidate in real_rank.items() if candidate == candidate_j][0]
                        pred_rank_j = j
                        if (real_rank_i > real_rank_j and pred_rank_i > pred_rank_j) or (real_rank_i < real_rank_j and pred_rank_i < pred_rank_j):
                            nb_of_concordant_pairs += 1

        if nb_of_pairs > 0:
            return nb_of_concordant_pairs / nb_of_pairs
        else:
            return 0

    # Rang réciproque : distance qui sépare la vraie gagnante à son rang prédit
    def reciprocal_rank(self, real_rank, predicted_rank):
        winner = real_rank[1]
        predicted_rank_winner = [rank for rank, candidate in predicted_rank.items() if candidate == winner]
        # Si la candidate apparaît dans la prédiction du top 12
        if len(predicted_rank_winner) > 0:
            return 1/predicted_rank_winner[0]
        # Si la candidate n'apparaît pas dans la prédiction du top 12
        else:
            return 0

    # Le nombre de candidates qui appartiennent bien au top i (dans notre prédiction) / i
    def precision_at_i(self, real_rank, predicted_rank, i):
        nb_well_predicted = 0
        for j in range(1, i+1):
            candidate = predicted_rank[j]
            candidate_real_rank = [rank for rank, cand in real_rank.items() if cand == candidate and rank <= i]
            if len(candidate_real_rank) > 0:
                nb_well_predicted += 1
        return nb_well_predicted/i

    # Moyenne pondérée entre les precision (en donnant un poids plus important aux premiers rangs)
    def ap_at_k(self, real_rank, predicted_rank, k):
        sum_ = 0
        sum_weight = 0
        for i in range(1, k+1):
            weight = (k+1)-i
            sum_weight += weight
            sum_ += self.precision_at_i(real_rank, predicted_rank, i) * weight
        return (1/sum_weight)*sum_
