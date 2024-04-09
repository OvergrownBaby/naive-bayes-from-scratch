import pandas as pd
import numpy as np
from abc import ABC
import numpy as np
from scipy import sparse

class MultinomialNB():
    def __init__(self, log=False, alpha=1):
        self.log = log
        self.alpha = alpha
        self.model = {}  # {label: (class_prob, feature_probs)}

    def fit(self, X, y):
        labels = np.unique(y)
        class_probs = {label: (y == label).mean() for label in labels}

        num_of_features = X.shape[1]
        
        for label in labels:
            subset = X[y == label]
            total_count = subset.sum(axis=0)
            feature_probs = (total_count + self.alpha) / (total_count.sum() + self.alpha * num_of_features)
            if isinstance(feature_probs, np.matrix):
                feature_probs = feature_probs.tolist()[0]
            self.model[label] = (class_probs[label], feature_probs)
        return self.model

    def get_posterior(self, label, feature_vec):
        class_prob = self.model[label][0]
        feature_probs = self.model[label][1]

        # Convert sparse feature vector to a dense format
        if isinstance(feature_vec, sparse._csr.csr_matrix):
            feature_vec = feature_vec.toarray()[0]
        
        log_prob = np.log(class_prob)
        for i in range(len(feature_vec)):
            feature_prob = feature_probs[i] if i < len(feature_probs) else self.alpha
            log_prob += feature_vec[i] * np.log(feature_prob)
        
        return log_prob


    def predict(self, X):
        y_pred = []
        for i in range(X.shape[0]):
            if isinstance(X, pd.DataFrame):
                row = X.iloc[i]
            elif isinstance(X, sparse._csr.csr_matrix):
                row = X.getrow(i)
            y_pred.append(max(self.model.keys(), key=lambda label: self.get_posterior(label, row)))
        return np.array(y_pred)

# todo: add alpha parameter for laplace smoothing
class BernoulliNB():
    def __init__(self, log=False, laplace_smoothing=False):
        self.log = log
        self.laplace_smoothing = laplace_smoothing

        self.model = {} # {label: (class_prob, feature_probs)}
        
    def fit(self, X: pd.DataFrame, y: pd.Series):
        labels = y.unique()
        class_probs = y.value_counts(normalize=True)

        for label in labels:
            subset = X[y == label]
            if self.laplace_smoothing:
                feature_probs = subset.sum().add(1) / (len(subset) + 2)
            else:
                feature_probs = subset.mean()
            self.model[label] = (class_probs[label], feature_probs.to_dict())

        # print(self.model)

    def get_posterior(self, label, feature_vec: pd.DataFrame):

        if self.log:
            # log(p(class)) + log(p(x_1|class)) + ... + log(p(x_n|class))
            return np.log(self.model[label][0]) + np.sum([
                np.log(self.model[label][1][feature]) if feature_vec[feature] == 1 
                else np.log1p(-self.model[label][1][feature]) 
                for feature in feature_vec.index
            ])
        else:
            # p(class) * p(x_1|class) * ... * p(x_n|class)
            return self.model[label][0] * np.prod([
                self.model[label][1][feature] if feature_vec[feature] == 1 
                else 1 - self.model[label][1][feature] 
                for feature in feature_vec.index
            ])


    def predict(self, X: pd.DataFrame):
        y_pred = []
        for _, row in X.iterrows():
            y_pred.append(max(self.model.keys(), key=lambda label: self.get_posterior(label, row)))
        return y_pred