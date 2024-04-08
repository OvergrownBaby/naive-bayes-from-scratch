import pandas as pd
import numpy as np
from abc import ABC

class MultinomialNB():
    def __init__(self, log=False, alpha=1):
        self.log = log
        self.alpha = alpha

        self.model = {} # {label: (class_prob, feature_counts, feature_probs)}

    def fit(self, X: pd.DataFrame, y: pd.Series):
        labels = y.unique()
        class_probs = y.value_counts(normalize=True)

        num_of_keywords = len(X.columns)
        alpha = self.alpha
        
        for label in labels:
            subset = X[y == label]
            total_count = sum(subset.sum())
            feature_probs = (subset.sum() + alpha) / (total_count + alpha*num_of_keywords)
            self.model[label] = (class_probs[label], feature_probs.to_dict())

    def get_posterior(self, label, feature_vec: pd.DataFrame):
        """
        argmax p(C|X) = p(C)p(X|C)\n
        where p(X|C) = m_coeff * p(x_1|C)^x_1 * ... * p(x_n|C)^x_n\n
        but after log transform and optimization, \n
        p(X|C) = p(x_1|C)^x_1 + ... + p(x_n|C)^x_n
        where p(x_i|C) = (count of word i in class + a)/(total words in class)
        """
        class_prob = self.model[label][0]
        feature_probs = self.model[label][1]
        
        # log(p(class)) + log(p(x_1|class)) + ... + log(p(x_n|class))
        return np.log(class_prob) + np.sum([
            np.log(feature_probs[feature]) if feature_vec[feature] == 1 
            else np.log1p(-feature_probs[feature]) 
            for feature in feature_vec.index
        ])

    def predict(self, X: pd.DataFrame):
        y_pred = []
        for _, row in X.iterrows():
            y_pred.append(max(self.model.keys(), key=lambda label: self.get_posterior(label, row)))
        return y_pred
    

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