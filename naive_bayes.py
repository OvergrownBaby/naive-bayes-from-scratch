import pandas as pd
import numpy as np

# todo: vectorize likelihood calculations

class MultinomialNB():
    def __init__(self):
        self.model = {}

    def fit(self, X: pd.DataFrame, y: pd.Series):
        labels = y.unique()
        features = X.columns

        
        for label in labels:
            subset = X[y == label]
            likelihoods = {}
            class_prob = len(subset) / len(X)
            for feature in features:
                likelihoods[feature] = subset[feature].mean()
            self.model[label] = (class_prob, likelihoods)

        

    def predict():
        pass


class BernoulliNB():
    def __init__(self, log_likelihood=False):
        self.log_likelihood = log_likelihood
        print("Model: Bernoulli Naive Bayes")
        print("Log Likelihood: ", self.log_likelihood)

        self.model = {}
        

    def fit(self, X: pd.DataFrame, y: pd.Series):
        labels = y.unique()
        features = X.columns

        ### uses log likelihood to avoid underflow + laplace smoothing
        if self.log_likelihood:
            class_probs = np.log(y.value_counts(normalize=True))

            for label in labels:
                subset = X[y == label]
                log_likelihoods = {}
                # Use Laplace smoothing to avoid log(0)
                for feature in features:
                    feature_count = subset[feature].sum()
                    log_likelihoods[feature] = np.log((feature_count + 1) / (len(subset) + 2))
                self.model[label] = (class_probs[label], log_likelihoods)

        ### base implementation
        else:
            class_probs = y.value_counts(normalize=True)

            for label in labels:
                subset = X[y == label]
                likelihoods = {}
                for feature in features:
                    likelihoods[feature] = (subset[feature] == 1).mean()
                self.model[label] = (class_probs[label], likelihoods)


    def predict(self, X: pd.DataFrame):
        def get_posterior(label, feature_vec):
            if self.log_likelihood:
                return self.model[label][0] + np.sum(
                    [self.model[label][1][feature] if feature_vec[feature] == 1 
                    else np.log1p(-np.exp(self.model[label][1][feature])) 
                    for feature in feature_vec.index])
            else:
                return self.model[label][0] * np.prod(
                    [self.model[label][1][feature] if feature_vec[feature] == 1 
                    else 1 - self.model[label][1][feature] 
                    for feature in feature_vec.index])
        
        y_pred = []
        for _, row in X.iterrows():
            y_pred.append(max(self.model.keys(), key=lambda label: get_posterior(label, row)))
        return y_pred