import numpy as np

class MyModel(object):
    def __init__(self, model= None):
        self.model = model

    def fit(self, X_train, y_train):
        for i in range(12):
            self.model[i].fit(X_train, y_train[i])

    #Renvoyer la matrice de prédiction (celle avec toutes les probabilités)
    def prediction_matrix(self, X):
        result = []
        for i in range(12):
            y_pred_real = self.model[i].predict(X)
            # print("voici ce qu'on prédit : ", y_pred_real)
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
        classement = {}
        for i in range(12):
            max_ = np.argmax(scores)
            classement[i + 1] = list_candidate[max_]
            scores[max_] = -1
        return classement

    def score(self, X_test, y_test):
        scores = []
        for i in range(12):
            scores.append(self.model[i].score(X_test, y_test[i]))
        return np.array(scores)