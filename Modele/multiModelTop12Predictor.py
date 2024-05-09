import math
import pickle

import numpy as np
from sklearn.metrics import balanced_accuracy_score


class MultiModelTop12Predictor(object):
    def __init__(self, model=None):
        self.model = model

    def fit(self, X_train, y_train):
        for i in range(12):
            self.model[i].fit(X_train, y_train[i])

    # Renvoyer la matrice de prédiction (celle avec toutes les probabilités)
    def prediction_matrix(self, X):
        result = []
        for i in range(12):
            y_pred = self.model[i].predict_proba(X)
            sublist = []
            for j in range(len(y_pred)):
                sublist.append(y_pred[j][1])
            result.append(sublist)
        return np.array(result).T

    def predict(self, X, list_candidate):
        prediction_matrix = self.prediction_matrix(X)
        scores = []
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

    def score(self, X_test, y_test):
        scores = []
        for i in range(12):
            scores.append(self.model[i].score(X_test, y_test[i]))
        return np.array(scores)

    def evaluate_prediction(self, X_test, list_candidate, real_rank):
        prediction = self.predict(X_test, list_candidate)
        sum = 0
        for (key, value) in prediction.items():
            if value in real_rank.keys():
                diff = key - real_rank[value]
            else:
                diff = 20
            sum += math.pow(diff, 2)
        return sum

    def dump_model(self, path):
        for i in range(12):
            name_file_model = path+str(i)+'.pkl'
            pickle.dump(self.model[i], open(name_file_model, 'wb'))

    def balanced_accuracy_score(self, X_test, y_test):
        sum = 0
        # Construction de y_pred
        y_pred = []
        for i in range(12):
            pred = list(self.model[i].predict(X_test))
            y_pred.append(pred)
        for i in range(12):
            sum += balanced_accuracy_score(y_test[i], y_pred[i])
        return sum/12

    # On utilise le raisonnement de la Kendall-Tau distance (c'est un score : il faut le maximiser)
    def fraction_of_concordant_pairs(self, real_rank, predicted_rank):
        nb_of_concordant_pairs = 0
        nb_of_pairs = 66
        for i in range(1, 13):
            candidate_i = predicted_rank[i]
            # Recherche de la candidate dans le réel rang
            cand = [candidate for rank, candidate in real_rank.items() if candidate == candidate_i]
            if len(cand) > 0:
                real_rank_i = [rank for rank, candidate in real_rank.items() if candidate == candidate_i][0]
                pred_rank_i = i
                for j in range(i+1, 13):
                    candidate_j = predicted_rank[j]
                    # Recherche de la candidate j dans le réel rang
                    cand = [candidate for rank, candidate in real_rank.items() if candidate == candidate_j]
                    if len(cand) == 0:
                        nb_of_concordant_pairs+=1
                    else:
                        real_rank_j = [rank for rank, candidate in real_rank.items() if candidate == candidate_j][0]
                        pred_rank_j = j
                        if (real_rank_i > real_rank_j and pred_rank_i > pred_rank_j) or (real_rank_i < real_rank_j and pred_rank_i < pred_rank_j):
                            nb_of_concordant_pairs += 1

        if nb_of_pairs > 0:
            return nb_of_concordant_pairs / nb_of_pairs
        else:
            return 0

    def reciprocal_rank(self, real_rank, predicted_rank):
        winner = real_rank[1]
        rank_winner = [rank for rank, candidate in predicted_rank.items() if candidate == winner]
        if len(rank_winner) > 0:
            return 1/rank_winner[0]
        else:
            return 0

    # Le nombre de candidates qui appartiennent bien au top i (dans notre prédiction) / i : cette métrique est à maximiser
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
        sum = 0
        sum_weight = 0
        for i in range(1, k+1):
            weight = (k+1)-i
            sum_weight += weight
            sum += self.precision_at_i(real_rank, predicted_rank, i) * weight
        return (1/sum_weight)*sum
