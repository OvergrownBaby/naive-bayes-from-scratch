import pandas as pd
import numpy as np

class BernoulliNB():
    def __init__(self, log_likelihood=False):
        self.model = {}
        self.log_likelihood = log_likelihood
        print("Model Parameters:")
        print("Log Likelihood: ", self.log_likelihood)

    def fit(self, X: pd.DataFrame, y: pd.Series):
        labels = y.unique()
        features = X.columns

        ### uses log likelihood to avoid underflow + laplace smoothing
        if self.log_likelihood:
            # Compute the log prior probability for each class
            log_prior = np.log(y.value_counts(normalize=True))
            
            # Compute the log likelihood for each feature given each class
            
            for label in labels:
                subset = X[y == label]
                log_likelihood = {}
                # Use Laplace smoothing to avoid log(0)
                for feature in features:
                    feature_count = subset[feature].sum()
                    log_likelihood[feature] = np.log((feature_count + 1) / (len(subset) + 2))
                self.model[label] = (log_prior[label], log_likelihood)

        ### base implementation
        else:
            for label in labels:
                p_label = (y == label).mean()
                dataset = X[y == label]
                likelihoods = {}
                for feature in features:
                    likelihoods[feature] = (dataset[feature] == 1).mean()
                self.model[label] = (p_label, likelihoods)


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

    def score(self, X, y):
        y_pred = self.predict(X)
        return np.mean(y_pred == y)