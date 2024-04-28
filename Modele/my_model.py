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

    def predict(self,X, list_candidate):
        prediction_matrix = self.prediction_matrix(X)
        scores = []
        for k in range(len(prediction_matrix)):
            sum_proba = 0
            for j in range(12):
                sum_proba += prediction_matrix[k][j]
            scores.append(sum_proba)
        prediction_matrix = np.array(prediction_matrix)
        new_prediction_matrix = np.array([])
        # On récupère les 12 meilleures miss (scores) pour filtrer les candidates
        classement = []
        list_indices = []
        for i in range(12):
            max_ = np.argmax(scores)
            classement.append(max_)
            scores[max_] = -1
            list_indices.append(max_)
        # On met tous ceux qui ne sont pas dans le top 12 à -1
        for i in range(len(prediction_matrix)):
            if i not in list_indices:
                prediction_matrix[i, :] = -1
        candidates = {}
        # ANAA :  écrit le commentaire pour ça mercii
        for i in range(12):
            index_max = np.argmax(prediction_matrix[:, i])
            candidates[i + 1] = list_candidate[index_max]
            prediction_matrix[index_max, :] = -1
            prediction_matrix[:, i] = -1
        return candidates

    def score(self, X_test, y_test):
        scores = []
        for i in range(12):
            scores.append(self.model[i].score(X_test, y_test[i]))
        return np.array(scores)