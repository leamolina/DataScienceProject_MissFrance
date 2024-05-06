import numpy as np
import pickle
import math

from sklearn.utils.validation import check_is_fitted


class MultiModelTop12Predictor(object):
    def __init__(self, model= None):
        self.model = model

    def fit(self, X_train, y_train):
        for i in range(12):
            self.model[i].fit(X_train, y_train[i])

    #Renvoyer la matrice de prédiction (celle avec toutes les probabilités)
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
        #Remplacer cette ligne par
        for i in range(min(12, len(list_candidate))):
            #for i in range(12):
            max_ = np.argmax(scores)
            ranking[i + 1] = list_candidate[max_]
            scores[max_] = -1
        return ranking

    def score(self, X_test, y_test):
        scores = []
        for i in range(12):
            scores.append(self.model[i].score(X_test, y_test[i]))
        return np.array(scores)

    def evaluate_prediction(self, X_test, list_candidate,real_rank) :
        prediction = self.predict(X_test, list_candidate)
        sum = 0
        for (key, value) in prediction.items():
            if (value in real_rank.keys()):
                diff = key - real_rank[value]
            else:
                diff = 20
            sum += math.pow(diff, 2)
        return sum

    def dump_model(self, path):
        for i in range(12):
            name_file_model = path+str(i)+'.pkl'
            pickle.dump(self.model[i], open(name_file_model, 'wb'))