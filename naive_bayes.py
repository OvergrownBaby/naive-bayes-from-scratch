import pandas as pd
import numpy as np

class bernoulliNB():
    def __init__(self):
        self.model = None

    def fit(self, X: pd.DataFrame, y: pd.Series):
        labels = y.unique()
        features = X.columns

        ret = {}
        for label in labels:
            p_label = (y == label).mean()
            dataset = X[y == label]
            likelihoods = {}
            for feature in features:
                likelihoods[feature] = (dataset[feature] == 1).mean()
            ret[label] = (p_label, likelihoods)

        self.model = ret

    def predict(self, X: pd.DataFrame):
        def get_posterior(label, feature_vec):
            # print([feature for feature in feature_vec.index])
            # print([feature for feature in feature_vec.columns()])
            return self.model[label][0] * np.prod([self.model[label][1][feature] if feature_vec[feature] == 1 else 1 - self.model[label][1][feature] for feature in feature_vec.index])
        
        y_pred = []
        for idx, row in X.iterrows():
            get_posterior(1, row.drop('text'))
            y_pred.append(max(self.model.keys(), key=lambda label: get_posterior(label, row.drop('text'))))
        # return max(self.model.keys(), key=lambda label: get_posterior(label, X.columns.drop('text')))
        return y_pred

    def score(self, X, y):
        y_pred = self.predict(X)
        return np.mean(y_pred == y)